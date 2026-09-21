import {
  Box3,
  BufferGeometry,
  Color,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  type Material,
  Matrix4,
  Mesh,
  MeshStandardMaterial,
  type Object3D,
  Points,
  PointsMaterial,
  Raycaster,
  Sphere,
  Vector2,
  Vector3,
} from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from "three-mesh-bvh";

import type { AssemblyDto } from "../ipc/bindings/AssemblyDto";
import type { PairDetailDto } from "../ipc/bindings/PairDetailDto";
import { fragmentColour, groupColour, UNPAIRED_COLOUR } from "./colours";
import { type Disc, packDiscs } from "./layout";
import { rowsToMatrix4 } from "./matrix";
import {
  contactColour,
  ghostMatrix,
  PAIR_A_COLOUR,
  PAIR_B_COLOUR,
  SEAM_COLOUR,
  separationOffset,
} from "./pair";
import { principalAxes } from "./principal";
import {
  disposeObject,
  FROM,
  isMesh,
  materialsOf,
  sphereOf,
  Stage,
  themeName,
  tokenColour,
  viewDirection,
} from "./stage";

/**
 * Picking against 600 000 faces, once and for the whole window (A §7.2). `acceleratedRaycast`
 * falls back to three.js's own raycast for a geometry that has no bounds tree, so the «Вход»
 * viewer is unaffected by this; only the meshes this file builds a tree for go the fast way.
 */
BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
Mesh.prototype.raycast = acceleratedRaycast;

/**
 * How many GLBs are fetched at once. The asset protocol goes through the shell's own HTTP
 * handler, one thread per request, and a collection of 155 asking all at once would starve the
 * very run whose result is being drawn; four keeps the pipe full without taking it over.
 */
const CONCURRENCY = 4;

/** The air between two spread-out groups, as a fraction of the largest group's radius. */
const GAP_FRACTION = 0.15;

/** Above this many names on screen at once nothing can be read; A §7.2's «Подписи» then narrows. */
const LABEL_LIMIT = 60;

/**
 * How wide and how tall one name is on screen, near enough to test two of them for a collision:
 * the labels are one 10 px line of the same eight-character shape, so an estimate spares a layout
 * read per name per frame that a measurement of the real element would cost.
 */
const LABEL_CHAR = 6.2;
const LABEL_LINE = 14;

/** How much light the hovered and the selected fragment give off by themselves. */
const HOVER_LIFT = 0.1;
const SELECT_LIFT = 0.26;

/** How far a pointer may travel between down and up and still count as a click and not an orbit. */
const CLICK_SLOP = 4;

/**
 * How big one contact point or one seam voxel is drawn, in CSS pixels, and flat: a seam of five
 * thousand samples read at arm's length is a texture, and a point that shrinks with distance
 * turns the far half of it into nothing. Three pixels is the smallest dot that still has a hue.
 */
const POINT_SIZE = 3;

/** How much of the fragment under it a ghost lets through (A §7.2: «translucent»). */
const GHOST_OPACITY = 0.42;

/** The pose of the fragment a pair is shown *in the frame of*: A stands still, B moves (R §0). */
const IDENTITY = new Matrix4();

/** The fragment the pointer is on, the one that is chosen, or neither. */
type Emphasis = "none" | "hover" | "select";

const LIFT: Record<Emphasis, number> = { none: 0, hover: HOVER_LIFT, select: SELECT_LIFT };

/** Every fragment of the run's input snapshot that has a display GLB, and what the run made of them. */
export interface AssemblyInput {
  fragments: { name: string; url: string }[];
  assembly: AssemblyDto;
}

/** «Цвет: скан / фрагменты / группы» (A §7.2). */
export type ColourMode = "scan" | "fragment" | "group";

/** «Все группы разнесены» ↔ «одна группа» (A §7.2). */
export type LayoutMode = "spread" | "single";

/** What the React shell around the viewer has to hear about. */
export interface ViewerEvents {
  /** The fragment under the cursor and where the cursor is, in the container's own coordinates. */
  onHover(name: string | null, at: { x: number; y: number } | null): void;
  /** A click on a fragment, or on the background. */
  onSelect(name: string | null): void;
  /** How many of the display meshes are in; `loaded === total` is «the assembly is whole». */
  onProgress(loaded: number, total: number): void;
}

/** One mesh of one fragment, and what it was wearing when it arrived. */
interface MeshSkin {
  mesh: Mesh;
  /** The material the GLB came with, in the shape the mesh had it in. */
  scan: Material | Material[];
  /** Those of them that can glow, with the emissive they came with. */
  lights: { material: MeshStandardMaterial; was: Color }[];
}

/** One fragment in the scene: where it hangs, what it is wearing, and how big it is. */
interface Member {
  readonly name: string;
  /** Its place in the input list — which colour «Цвет: фрагменты» gives it. */
  readonly index: number;
  readonly url: string;
  /** Carries «Разъединить»'s push; its child is the GLB, whose own matrix is the pose. */
  readonly holder: Group;
  object: Object3D | null;
  skins: MeshSkin[];
  /** Its box in its group's frame with no explode; empty until its mesh has arrived. */
  readonly box: Box3;
  /** Which group of the current assembly it belongs to, or −1 for none. */
  group: number;
  /** Whether the current assembly gave it a pose at all. */
  placed: boolean;
}

/** One assembly group: a node of its own, so that spreading the groups moves one matrix each. */
interface GroupNode {
  readonly node: Group;
  readonly members: Member[];
  /** A group of one is not a vessel — it belongs in A §7.2's «unassembled tray». */
  singleton: boolean;
  hidden: boolean;
  /** Its centre and radius in its own frame, with the current explode applied. */
  readonly centre: Vector3;
  radius: number;
}

/** What «Пара» is showing (A §8.3): the two fragments, and the matrix that puts B into A's frame. */
interface Pair {
  readonly a: Member;
  readonly b: Member;
  /** `p_a = T · p_b`, the candidate's own matrix (R §0). */
  readonly pose: Matrix4;
  /** What tells a new pair from the same pair asked for twice — a re-render must not re-frame. */
  readonly key: string;
}

/** What the camera was doing before «Пара» took the viewport over, so that leaving gives it back. */
interface Parked {
  position: Vector3;
  up: Vector3;
  target: Vector3;
  near: number;
  far: number;
}

/** Reusable scratch, so that a hover or a frame allocates nothing. */
const TMP_BOX = new Box3();
const TMP_V = new Vector3();

/**
 * A whole run in 3D (A §7.2), with no React anywhere in it.
 *
 * Every fragment's display mesh is loaded **once**; the run is laid over them as one matrix each,
 * which is what makes milestone 5's draft reassembly a `setAssembly` and not a reload. The engine
 * centres every group on the origin (R §8.2), so the groups would otherwise sit inside one
 * another: `layout.ts` packs them onto a plane, and «Разъединить» pushes a group's own fragments
 * out of its centroid.
 *
 * What it shares with the «Вход» viewer — the canvas, the renderer, the camera, the lights, the
 * draw-on-demand flag and the disposal — is [`Stage`]'s.
 */
export class AssemblyViewer {
  private readonly stage: Stage;
  private readonly events: ViewerEvents;
  private readonly loader = new GLTFLoader();

  /** Everything the run holds, so that one `remove` empties the viewport. */
  private readonly root = new Group();

  private readonly members = new Map<string, Member>();
  private order: Member[] = [];
  private groups: GroupNode[] = [];

  /** Every mesh and every GLB root back to its fragment, for picking. */
  private readonly owner = new Map<Object3D, Member>();

  /** The materials of «Цвет: фрагменты» and «Цвет: группы», one per colour and emphasis. */
  private readonly palette = new Map<string, MeshStandardMaterial>();

  /**
   * «Пара» (A §8.3), and the node it hangs on. The node never moves, so its frame **is** A's
   * frame — which is the frame `PairDetail` gives its points in, and why they can be children of
   * it with no matrix of their own.
   */
  private pair: Pair | null = null;
  private readonly pairNode = new Group();
  private separation = 0;

  /** The seam over that pair, the two clouds that draw it, and the theme they were coloured for. */
  private detail: PairDetailDto | null = null;
  private contactCloud: Points | null = null;
  private seamCloud: Points | null = null;
  private detailTheme: string | null = null;
  /** One material for both clouds: the colours are per vertex, so the size is all they differ in. */
  private readonly pointMaterial = new PointsMaterial({
    size: POINT_SIZE,
    sizeAttenuation: false,
    vertexColors: true,
  });

  /** A candidate's partner where the candidate would put it (A §7.2), and what every ghost wears. */
  private readonly ghostNode = new Group();
  private ghostMaterial: MeshStandardMaterial | null = null;
  /**
   * The group put back on screen for as long as the ghost hangs on a fragment inside it, or
   * `null` when the ghost needed nothing shown.
   *
   * A §7.2 hides the tray of groups of one by default, and an unpaired fragment is exactly the
   * one whose candidates a reviewer hovers — «и куда он тогда встанет?» is a question about a
   * piece that is not standing anywhere yet. Without this the answer was a viewport in which
   * nothing happened, because the anchor the ghost is measured from was not drawn.
   */
  private ghostGroup: number | null = null;

  /** Where the camera stood before a pair took the viewport; `null` outside «Пара». */
  private parked: Parked | null = null;

  private assembly: AssemblyDto | null = null;
  private colourMode: ColourMode = "scan";
  private layoutMode: LayoutMode = "spread";
  private single: number | null = null;
  private unassembled = false;
  private explode = 0;
  private labelsOn = false;
  private selected: string | null = null;
  private hovered: string | null = null;

  private readonly raycaster = new Raycaster();
  private readonly ndc = new Vector2();
  /** Where the pointer is, in the container's coordinates, and whether it has moved since the last pick. */
  private pointer: { x: number; y: number } | null = null;
  private pointerMoved = false;
  private picking = 0;
  private down: { x: number; y: number } | null = null;

  /** The names drawn over the scene (`L`), and the elements they are drawn with. */
  private readonly labelLayer: HTMLDivElement;
  private readonly labelPool: HTMLDivElement[] = [];

  /** Takes every listener off in one call, whatever they were put on. */
  private readonly listeners = new AbortController();

  /** Counts `load()` calls, so a slow earlier load cannot pour its meshes into a later one. */
  private loads = 0;
  private arrived = 0;
  /** Set the first time the user grabs the controls: nothing re-frames the camera under them. */
  private userMoved = false;
  private disposed = false;

  /**
   * The element the viewport lives in — not a canvas. [`Stage`] makes the canvas itself and takes
   * it away again in `dispose()`, because a canvas React kept across a remount would be handed
   * back to the next viewer with the first one's WebGL context on it, lost. The container is also
   * what «Подписи» hangs its names on.
   */
  constructor(container: HTMLElement, events: ViewerEvents) {
    this.events = events;
    this.stage = new Stage(container, {
      onRender: () => {
        this.placeLabels();
        this.syncDetailTheme();
      },
      onUserMove: () => {
        this.userMoved = true;
      },
    });
    this.stage.scene.add(this.root);
    // Both stand at the origin for good: a ghost carries its own world matrix and the pair's
    // node is A's frame, which is the frame `PairDetail`'s points already come in.
    this.pairNode.visible = false;
    this.ghostNode.matrixAutoUpdate = false;
    this.root.add(this.pairNode, this.ghostNode);
    this.raycaster.firstHitOnly = true;

    // The label layer is positioned against the container, so the container has to be a
    // containing block. It is the viewer's own element (`AssemblyView` gives it nothing else to
    // do), and a caller who forgot the `relative` would otherwise see the names in the corner.
    if (getComputedStyle(this.stage.container).position === "static") {
      this.stage.container.style.position = "relative";
    }
    this.labelLayer = document.createElement("div");
    this.labelLayer.style.cssText = "position:absolute;inset:0;overflow:hidden;pointer-events:none";
    this.labelLayer.hidden = true;
    this.stage.container.appendChild(this.labelLayer);

    this.watchPointer();
  }

  /**
   * Fetches every fragment's display mesh, at most [`CONCURRENCY`] at a time, and shows each one
   * as it arrives rather than after the last: a collection of 155 takes seconds, and an empty
   * viewport for those seconds reads as «there is nothing here». Resolves when all of them are
   * in, however many of them failed — one GLB that will not parse is one fragment missing from
   * the assembly, not a broken run.
   */
  async load(input: AssemblyInput): Promise<void> {
    const token = ++this.loads;
    this.clear();
    this.order = input.fragments.map((fragment, index) => ({
      name: fragment.name,
      index,
      url: fragment.url,
      holder: new Group(),
      object: null,
      skins: [],
      box: new Box3(),
      group: -1,
      placed: false,
    }));
    for (const member of this.order) {
      this.members.set(member.name, member);
    }
    this.setAssembly(input.assembly);

    const total = this.order.length;
    this.events.onProgress(0, total);
    if (total === 0) {
      return;
    }
    const queue = [...this.order];
    let done = 0;
    const worker = async (): Promise<void> => {
      for (;;) {
        const member = queue.shift();
        if (member === undefined || this.stale(token)) {
          return;
        }
        try {
          const gltf = await this.loader.loadAsync(member.url);
          if (this.stale(token)) {
            disposeObject(gltf.scene);
            return;
          }
          this.attach(member, gltf.scene);
        } catch {
          // A missing file, a scope the asset protocol refuses, a GLB the parser chokes on: this
          // fragment is one of many and the rest of the assembly is still worth looking at.
        }
        done += 1;
        this.events.onProgress(done, total);
      }
    };
    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, total) }, () => worker()));
    if (!this.stale(token) && !this.userMoved) {
      this.fit();
    }
  }

  /** Whether a load has been overtaken — by a newer `load()`, or by the viewer going away. */
  private stale(token: number): boolean {
    return this.disposed || token !== this.loads;
  }

  /**
   * A new set of poses and groups over the meshes already loaded (A §7.2) — which is all a draft
   * reassembly is, and why milestone 5's slider will not reload a byte.
   */
  setAssembly(assembly: AssemblyDto): void {
    this.assembly = assembly;
    for (const group of this.groups) {
      this.root.remove(group.node);
    }
    this.groups = assembly.groups.map((group) => ({
      node: new Group(),
      members: [],
      singleton: group.members.length < 2,
      hidden: false,
      centre: new Vector3(),
      radius: 0,
    }));
    for (const group of this.groups) {
      this.root.add(group.node);
    }

    const of = new Map<string, number>();
    assembly.groups.forEach((group, index) => {
      for (const name of group.members) {
        of.set(name, index);
      }
    });

    for (const member of this.order) {
      member.group = of.get(member.name) ?? -1;
      this.groups[member.group]?.members.push(member);
    }
    this.rehang();
  }

  /** «Цвет: скан / фрагменты / группы» (`C`). */
  setColourMode(mode: ColourMode): void {
    if (this.colourMode === mode) {
      return;
    }
    this.colourMode = mode;
    this.repaint();
    this.stage.invalidate();
  }

  /** «Все группы» ↔ «Одна группа»; the group is the one the tree asked to show alone. */
  setLayout(mode: LayoutMode, group?: number): void {
    this.layoutMode = mode;
    this.single = mode === "single" ? (group ?? this.single ?? 0) : null;
    this.relayout();
  }

  /** The tray of groups of one: off by default, because 81 loose sherds are clutter (A §7.2). */
  setUnassembledVisible(on: boolean): void {
    if (this.unassembled === on) {
      return;
    }
    this.unassembled = on;
    this.relayout();
  }

  /** The eye beside a group's row in the tree. */
  setGroupVisible(group: number, on: boolean): void {
    const found = this.groups[group];
    if (found === undefined || found.hidden === !on) {
      return;
    }
    found.hidden = !on;
    this.relayout();
  }

  /** «Разъединить», 0…1: members pushed from their group's centroid by that much of its radius. */
  setExplode(amount: number): void {
    const want = Number.isFinite(amount) ? Math.min(Math.max(amount, 0), 1) : 0;
    if (this.explode === want) {
      return;
    }
    this.explode = want;
    this.relayout();
  }

  /**
   * «Пара» (A §8.3): the two fragments of one candidate alone in the viewport, A at the identity
   * in grey and B at the candidate's pose in orange — the two colours of the engine's own review
   * images, so that the screen and `review/<a>__<b>.png` read as one join and not as two.
   *
   * Everything else goes away and nothing is reloaded: the two fragments keep the display meshes
   * the run already put on the GPU and are simply given another matrix and another material,
   * which is what lets a reviewer walk a queue of thirty-nine pairs without waiting once.
   *
   * `null` puts the run back exactly as it was — the layout, the colour mode, what was hidden,
   * and the camera, which is parked on the way in: a pair is framed on the origin, and coming
   * back to an assembly spread over a plane at that zoom would look like an empty viewport.
   *
   * The pose comes from `candidates.json` over IPC. One that is not four rows of four finite
   * numbers, or a name the collection does not have, leaves the run on screen rather than
   * putting a `NaN` into the scene graph.
   */
  showPair(pair: { a: string; b: string; pose: number[][] } | null): void {
    if (pair === null) {
      this.leavePair();
      return;
    }
    const a = this.members.get(pair.a);
    const b = this.members.get(pair.b);
    if (a === undefined || b === undefined || a === b) {
      this.leavePair();
      return;
    }
    const key = `${pair.a}\u0000${pair.b}\u0000${JSON.stringify(pair.pose)}`;
    if (this.pair?.key === key) {
      return;
    }
    let pose: Matrix4;
    try {
      pose = rowsToMatrix4(pair.pose);
    } catch {
      this.leavePair();
      return;
    }
    if (this.pair === null) {
      this.park();
    }
    // A ghost belongs to the assembly it is a ghost against; in «Пара» there is no assembly.
    this.clearGhost();
    this.pair = { a, b, pose, key };
    this.rehang();
    this.drawDetail();
    // Every pair is framed, whether or not the user has moved the camera before: the whole of
    // this screen is «look at this one join», and the next one is somewhere else.
    this.fit();
  }

  /**
   * The seam of the placement on screen (A §8.3): B's fracture samples coloured by R §6.1's own
   * classes — green under the tight limit, amber under the gap limit, red beyond — and R §6.2's
   * seam voxels in white, as two point clouds in A's frame.
   *
   * A pair whose working mesh has no triangle has no surfaces and so no seam at all
   * (`sherd_core::review::seam_view`), and an answer may arrive after the reviewer has clicked
   * the next row: an empty cloud and a seam belonging to another pair are both drawn as nothing,
   * neither as an error.
   */
  setPairDetail(detail: PairDetailDto | null): void {
    this.detail = detail;
    this.drawDetail();
  }

  /** «Разъединить» for a pair, 0…1: B pushed off A along the line between the two centroids. */
  setPairSeparation(amount: number): void {
    const want = Number.isFinite(amount) ? Math.min(Math.max(amount, 0), 1) : 0;
    if (this.separation === want) {
      return;
    }
    this.separation = want;
    if (this.pair !== null) {
      this.relayout();
    }
  }

  /**
   * A ghost of `name` where a candidate would put it (A §7.2): the fragment drawn translucent at
   * `world(anchor) · pose`, or at `world(anchor) · pose⁻¹` when the candidate names the two
   * fragments the other way round — which is what the inspector's rows show on hover (A §7.3).
   *
   * It is a clone of the display mesh and not a second load: the geometry is shared with the
   * fragment it ghosts and is never this method's to give back. There is no ghost while «Пара»
   * is on, because there is nothing there for it to be a ghost *against*.
   *
   * The anchor's **group is put on screen** for as long as the ghost hangs, if it was not there
   * already: a ghost stands at `world(anchor) · pose`, so a piece nobody can see is a ghost
   * nobody can make sense of — and A §7.2 keeps the tray of unpaired fragments hidden by
   * default, which is precisely where the fragments whose candidates get hovered are.
   * [`clearGhost`] puts it back.
   */
  setGhost(ghost: { name: string; anchor: string; pose: number[][]; flip: boolean } | null): void {
    this.clearGhost();
    this.stage.invalidate();
    if (ghost === null || this.pair !== null) {
      return;
    }
    const member = this.members.get(ghost.name);
    const anchor = this.members.get(ghost.anchor);
    const object = member?.object ?? null;
    const on = anchor?.object ?? null;
    if (anchor === undefined || object === null || on === null) {
      return;
    }
    if (!this.shows(anchor) && this.layoutMode === "spread") {
      this.ghostGroup = anchor.group;
      // The reveal goes through the layout and not straight onto the node, so the group is
      // packed and placed with the others rather than left wherever it last stood — a tray
      // that has never been shown stands at the origin, over the block.
      this.relayout();
    }
    if (!this.shows(anchor)) {
      return;
    }
    on.updateWorldMatrix(true, false);
    let where: Matrix4;
    try {
      where = ghostMatrix(on.matrixWorld, ghost.pose, ghost.flip);
    } catch {
      return; // a candidate's pose is data from outside; no ghost is better than a wrong one
    }
    const clone = object.clone();
    // The GLB's own hierarchy keeps its matrices; only the root's is the run's pose, and the
    // ghost's place is the node's instead.
    clone.matrixAutoUpdate = false;
    clone.matrix.identity();
    clone.matrixWorldNeedsUpdate = true;
    const material = this.ghost();
    clone.traverse((child) => {
      if (isMesh(child)) {
        child.material = material;
      }
    });
    this.ghostNode.matrix.copy(where);
    this.ghostNode.matrixWorldNeedsUpdate = true;
    this.ghostNode.add(clone);
  }

  /** The chosen fragment, from the tree or from a click in the viewport. */
  select(name: string | null): void {
    if (this.selected === name) {
      return;
    }
    const before = this.selected;
    this.selected = name;
    this.repaintOne(before);
    this.repaintOne(name);
    this.stage.invalidate();
  }

  /** Puts one fragment in the middle of the viewport without turning the camera round it. */
  flyTo(name: string): void {
    const member = this.members.get(name);
    if (member === undefined || member.object === null || !this.shows(member)) {
      return;
    }
    const direction = this.stage.camera.position.clone().sub(this.stage.controls.target);
    if (direction.lengthSq() < 1e-12) {
      direction.copy(FROM);
    }
    this.stage.frame(sphereOf(member.object), direction.normalize(), this.stage.camera.up.clone());
  }

  /**
   * Frames everything that is visible (`F`, A §7.1). The direction is the principal axes of what
   * is on screen, as in the «Вход» viewer: a block of groups lying in the XY plane is met from
   * above at a tilt, and a single upright group from the side.
   */
  fit(): void {
    const corners: number[] = [];
    for (const member of this.order) {
      if (!this.shows(member) || member.box.isEmpty()) {
        continue;
      }
      const world = this.worldBox(member, TMP_BOX);
      for (const x of [world.min.x, world.max.x]) {
        for (const y of [world.min.y, world.max.y]) {
          for (const z of [world.min.z, world.max.z]) {
            corners.push(x, y, z);
          }
        }
      }
    }
    if (corners.length === 0) {
      return;
    }
    // The corners of every fragment's own box, not of one box around them all: a block of groups
    // has air in its corners, and both the framing and the direction must be about the sherds.
    const { from, up } = viewDirection(principalAxes(corners), true);
    this.stage.framePoints(corners, from, up);
  }

  /** «Подписи» (`L`): every visible fragment's name at its centroid, while they can be read. */
  setLabels(on: boolean): void {
    if (this.labelsOn === on) {
      return;
    }
    this.labelsOn = on;
    this.labelLayer.hidden = !on;
    this.stage.invalidate();
  }

  /** «Снимок PNG»: the current frame as a data URL, drawn on this tick so the buffer still holds it. */
  screenshot(): string {
    this.stage.renderNow();
    return this.stage.canvas.toDataURL("image/png");
  }

  /** Gives the GPU everything back — 155 display meshes is the biggest thing the window holds. */
  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.listeners.abort();
    if (this.picking !== 0) {
      cancelAnimationFrame(this.picking);
      this.picking = 0;
    }
    this.clear();
    for (const material of this.palette.values()) {
      material.dispose();
    }
    this.palette.clear();
    this.pointMaterial.dispose();
    this.ghostMaterial?.dispose();
    this.ghostMaterial = null;
    this.labelLayer.remove();
    this.stage.dispose();
  }

  /** Everything one load put in the scene, out of it and off the GPU. */
  private clear(): void {
    // Before the geometries go: a ghost is a clone that shares them, and one left hanging would
    // be drawn out of buffers the driver has already been given back.
    this.clearGhost();
    this.releasePoints();
    this.pair = null;
    this.detail = null;
    this.parked = null;
    for (const member of this.order) {
      this.release(member);
      member.holder.removeFromParent();
    }
    for (const group of this.groups) {
      group.node.removeFromParent();
    }
    this.groups = [];
    this.order = [];
    this.members.clear();
    this.owner.clear();
    this.arrived = 0;
    this.hovered = null;
  }

  /** Whether a fragment is one of the two «Пара» is showing. */
  private inPair(member: Member): boolean {
    return this.pair !== null && (member === this.pair.a || member === this.pair.b);
  }

  /**
   * Who hangs under what, given the pair: the two fragments of «Пара» under its node, everything
   * else under its group's, and a fragment the assembly gave no group to under nothing at all.
   *
   * `Object3D.add` takes a child off its previous parent, so this is also what moves a fragment
   * out of a group and back into it.
   */
  private reparent(): void {
    for (const member of this.order) {
      if (this.inPair(member)) {
        this.pairNode.add(member.holder);
        continue;
      }
      const group = this.groups[member.group];
      if (group === undefined) {
        member.holder.removeFromParent();
      } else {
        group.node.add(member.holder);
      }
    }
  }

  /**
   * After the assembly or the pair changed: where every fragment hangs, what matrix it wears,
   * how big it is, what colour it is, and where the whole lot stands. The order matters — a box
   * is measured through the parents a fragment has *now*, so the reparenting comes first.
   */
  private rehang(): void {
    this.reparent();
    for (const member of this.order) {
      member.placed = this.pose(member);
      this.measure(member);
    }
    this.repaint();
    this.relayout();
  }

  /** Back out of «Пара» to the run, camera and all. */
  private leavePair(): void {
    if (this.pair === null) {
      return;
    }
    this.pair = null;
    this.rehang();
    this.drawDetail();
    this.unpark();
  }

  /** Remembers where the camera is, so that leaving «Пара» is not a new view of the assembly. */
  private park(): void {
    this.parked = {
      position: this.stage.camera.position.clone(),
      up: this.stage.camera.up.clone(),
      target: this.stage.controls.target.clone(),
      near: this.stage.camera.near,
      far: this.stage.camera.far,
    };
  }

  /** And puts it back. */
  private unpark(): void {
    const parked = this.parked;
    this.parked = null;
    if (parked === null) {
      return;
    }
    this.stage.camera.position.copy(parked.position);
    this.stage.camera.up.copy(parked.up);
    this.stage.camera.near = parked.near;
    this.stage.camera.far = parked.far;
    this.stage.camera.updateProjectionMatrix();
    this.stage.controls.target.copy(parked.target);
    this.stage.controls.update();
    this.stage.invalidate();
  }

  /**
   * One fragment's geometries, their bounds trees and the materials **its own GLB brought** —
   * back to the GPU.
   *
   * The palette's are not its own. `paint` hangs one shared `MeshStandardMaterial` per colour and
   * emphasis on every mesh of every fragment, and those live as long as the viewer does; freeing
   * them here would free them once per mesh — a dozen times over for one material — and, worse,
   * leave the next assembly wearing a material whose GPU program has already been given back.
   * So the GLB's own material goes on again first, and the palette is disposed exactly once, in
   * [`dispose`](AssemblyViewer.dispose).
   */
  private release(member: Member): void {
    const object = member.object;
    if (object === null) {
      return;
    }
    for (const skin of member.skins) {
      skin.mesh.material = skin.scan;
    }
    object.traverse((child) => {
      if (isMesh(child)) {
        child.geometry.disposeBoundsTree();
      }
    });
    disposeObject(object);
    object.removeFromParent();
    member.object = null;
    member.skins = [];
  }

  /** One fragment's mesh has arrived: hang it, index it for picking, and show it at once. */
  private attach(member: Member, object: Object3D): void {
    member.object = object;
    member.holder.add(object);
    this.owner.set(object, member);
    object.traverse((child) => {
      if (isMesh(child)) {
        // A tree per geometry is what makes hovering 600 000 faces cost nothing (A §7.2).
        child.geometry.computeBoundsTree();
        this.owner.set(child, member);
        member.skins.push({
          mesh: child,
          scan: child.material,
          lights: materialsOf(child)
            .filter((material) => material instanceof MeshStandardMaterial)
            .map((material) => ({ material, was: material.emissive.clone() })),
        });
      }
    });
    member.placed = this.pose(member);
    this.measure(member);
    this.paint(member);
    this.arrived += 1;
    this.relayout();
    // The view widens as the collection arrives, so that the first sherd in is not a close-up
    // with the other hundred and fifty outside the frame for the next few seconds. The moment
    // the user touches the controls this stops: nothing moves a camera a person is holding —
    // except in «Пара», where the two fragments *are* the screen and one of them just arrived.
    if (!this.userMoved || this.inPair(member)) {
      this.fit();
    }
  }

  /**
   * Puts the run's own matrix on a fragment — the only route from a file's rows into the scene
   * (`matrix.ts`). A pose that is not sixteen finite numbers leaves the fragment unplaced rather
   * than putting a `NaN` into the scene graph, where it would take the whole group's bounding box
   * with it; the poses come from `assembly.json` and are data from outside.
   */
  private pose(member: Member): boolean {
    // «Пара» overrides the run for its two fragments and for nothing else (A §8.3): A stands at
    // the identity, which makes the scene's frame A's frame, and B wears the candidate's matrix.
    const pair = this.pair;
    if (pair !== null && (member === pair.a || member === pair.b)) {
      if (member.object === null) {
        return true;
      }
      member.object.matrixAutoUpdate = false;
      member.object.matrix.copy(member === pair.a ? IDENTITY : pair.pose);
      member.object.matrixWorldNeedsUpdate = true;
      return true;
    }
    const rows = this.assembly?.poses[member.name];
    if (rows === undefined || member.object === null) {
      return rows !== undefined;
    }
    try {
      member.object.matrixAutoUpdate = false;
      member.object.matrix.copy(rowsToMatrix4(rows));
      member.object.matrixWorldNeedsUpdate = true;
      return true;
    } catch {
      return false;
    }
  }

  /**
   * A fragment's box in its group's frame, with no explode. Both nodes above it are pure
   * translations, so the world box less their offsets is exactly that — and having it as a box
   * means the layout and «Разъединить» are arithmetic, with no traversal of the scene per frame.
   */
  private measure(member: Member): void {
    member.box.makeEmpty();
    if (member.object === null || !member.placed) {
      return;
    }
    member.object.updateWorldMatrix(true, true);
    const world = TMP_BOX.setFromObject(member.object);
    if (world.isEmpty()) {
      return;
    }
    member.box.copy(world).translate(this.offset(member, TMP_V).negate());
  }

  /**
   * How far the two pure translations above a fragment have carried it: its own «Разъединить»
   * push and its group's place on the plane. «Пара»'s node never moves, so a fragment under it
   * carries only its own push — which is the whole of the separation slider.
   */
  private offset(member: Member, into: Vector3): Vector3 {
    into.copy(member.holder.position);
    if (this.inPair(member)) {
      return into;
    }
    const parent = this.groups[member.group];
    return parent === undefined ? into : into.add(parent.node.position);
  }

  /** A fragment's box where the world sees it now. */
  private worldBox(member: Member, into: Box3): Box3 {
    return into.copy(member.box).translate(this.offset(member, TMP_V));
  }

  /**
   * Where every group stands and how far its fragments are pushed apart. Three passes, all on
   * stored boxes: what a group is without the explode, where the explode puts its members, and
   * where `packDiscs` puts the groups it ends up making.
   */
  private relayout(): void {
    const pair = this.pair;
    if (pair !== null) {
      this.layoutPair(pair);
      return;
    }
    this.pairNode.visible = false;
    // What is visible at all, first: a hidden group takes no space on the plane.
    for (const [index, group] of this.groups.entries()) {
      // A group shown for a ghost ([`ghostGroup`]) is laid out like any other, so the tray it
      // belongs to is packed and placed rather than left at the origin over the block. Only in
      // «Все группы»: «Одна группа» centres every group on the origin, and a second one revealed
      // there would stand inside the one the user asked to be alone with.
      const revealed = this.layoutMode === "spread" && this.ghostGroup === index;
      group.node.visible =
        revealed ||
        (!group.hidden &&
          (!group.singleton || this.unassembled) &&
          (this.layoutMode === "spread" || this.single === index));
      for (const member of group.members) {
        member.holder.visible = member.object !== null && member.placed;
      }
    }

    for (const group of this.groups) {
      // The group as the engine left it, which is what «Разъединить» pushes out of.
      const base = new Box3();
      for (const member of group.members) {
        if (member.holder.visible && !member.box.isEmpty()) {
          base.union(member.box);
        }
      }
      const sphere = base.isEmpty() ? new Sphere(new Vector3(), 0) : base.getBoundingSphere(new Sphere());
      const full = new Box3();
      for (const member of group.members) {
        member.holder.position.set(0, 0, 0);
        if (!member.holder.visible || member.box.isEmpty()) {
          continue;
        }
        if (this.explode > 0 && sphere.radius > 0) {
          member.box.getCenter(TMP_V).sub(sphere.center);
          if (TMP_V.lengthSq() > 1e-12) {
            member.holder.position.copy(TMP_V.normalize().multiplyScalar(this.explode * sphere.radius));
          }
        }
        full.union(TMP_BOX.copy(member.box).translate(member.holder.position));
      }
      const grown = full.isEmpty() ? new Sphere(new Vector3(), 0) : full.getBoundingSphere(new Sphere());
      group.centre.copy(grown.center);
      group.radius = grown.radius;
    }

    const shown = this.groups.map((group, id) => ({ group, id })).filter((row) => row.group.node.visible);
    const gap = GAP_FRACTION * shown.reduce((most, row) => Math.max(most, row.group.radius), 0);
    const block = shown.filter((row) => !row.group.singleton);
    const tray = shown.filter((row) => row.group.singleton);
    const discs = (rows: typeof shown): Disc[] => rows.map((row) => ({ id: row.id, radius: row.group.radius }));
    const above = packDiscs(discs(block), gap);
    const below = packDiscs(discs(tray), gap);
    // The tray hangs under the block, far enough down that the two read as two things (A §7.2).
    const drop = height(above, block) / 2 + height(below, tray) / 2 + 2 * gap;

    for (const [index, group] of this.groups.entries()) {
      if (this.layoutMode === "single") {
        group.node.position.copy(group.centre).negate();
        continue;
      }
      const place =
        above.find((row) => row.id === index) ??
        (() => {
          const found = below.find((row) => row.id === index);
          return found === undefined ? undefined : { id: found.id, x: found.x, y: found.y - drop };
        })();
      if (place === undefined) {
        continue;
      }
      group.node.position.set(place.x - group.centre.x, place.y - group.centre.y, -group.centre.z);
    }
    this.stage.invalidate();
  }

  /**
   * The same for «Пара» (A §8.3), where there is no plane to pack: the groups go dark, the two
   * fragments stand in the candidate's own arrangement, and «Разъединить» pushes B off A along
   * the line between their centroids — the only thing the slider does here, since a pair is not
   * a group being opened up but two pieces being taken apart.
   */
  private layoutPair(pair: Pair): void {
    this.pairNode.visible = true;
    for (const group of this.groups) {
      group.node.visible = false;
    }
    for (const member of this.order) {
      member.holder.position.set(0, 0, 0);
      member.holder.visible = this.inPair(member) && member.object !== null && member.placed;
    }
    if (this.separation > 0 && !pair.a.box.isEmpty() && !pair.b.box.isEmpty()) {
      // How far a full slider pushes: the radius of the larger of the two pieces — the same
      // scale «Разъединить» uses over a group, so the two sliders of the app feel like one.
      const radius = (box: Box3): number => box.getBoundingSphere(new Sphere()).radius;
      const span = Math.max(radius(pair.a.box), radius(pair.b.box));
      const from = pair.a.box.getCenter(new Vector3());
      const to = pair.b.box.getCenter(new Vector3());
      pair.b.holder.position.copy(separationOffset(from, to, span, this.separation));
    }
    this.stage.invalidate();
  }

  /** What every fragment is wearing, from scratch — a colour mode or a new assembly changed it. */
  private repaint(): void {
    for (const member of this.order) {
      this.paint(member);
    }
  }

  /** The same for one fragment, by name, when the selection or the hover moved on or off it. */
  private repaintOne(name: string | null): void {
    const member = name === null ? undefined : this.members.get(name);
    if (member !== undefined) {
      this.paint(member);
    }
  }

  /**
   * Colour modes swap **materials** and never vertex data (A §7.2): «скан» puts back exactly what
   * the engine wrote into the GLB, the other two hang one shared material per colour on every
   * mesh of the fragment. The chosen fragment glows, the hovered one glows less — which is a
   * different material in the palette modes and the fragment's own emissive in «скан», where the
   * materials belong to that one fragment and can be lifted in place.
   */
  private paint(member: Member): void {
    const lift = LIFT[this.emphasisOf(member)];
    // «Пара» is not a fourth colour mode but the review images' own two colours, and it wins
    // over whichever mode the «Сборка» screen was left in (A §8.3).
    const pair = this.pair;
    if (pair !== null && (member === pair.a || member === pair.b)) {
      const material = this.shared(member === pair.a ? PAIR_A_COLOUR : PAIR_B_COLOUR, lift);
      for (const skin of member.skins) {
        skin.mesh.material = material;
      }
      return;
    }
    if (this.colourMode === "scan") {
      for (const skin of member.skins) {
        skin.mesh.material = skin.scan;
        for (const light of skin.lights) {
          light.material.emissive.copy(light.was).addScalar(lift);
        }
      }
      return;
    }
    const group = this.groups[member.group];
    const colour =
      this.colourMode === "fragment"
        ? fragmentColour(member.index)
        : group === undefined || group.singleton
          ? UNPAIRED_COLOUR
          : groupColour(member.group);
    const material = this.shared(colour, lift);
    for (const skin of member.skins) {
      skin.mesh.material = material;
    }
  }

  /** Is this the chosen fragment, the one under the cursor, or neither? */
  private emphasisOf(member: Member): Emphasis {
    if (this.selected === member.name) {
      return "select";
    }
    return this.hovered === member.name ? "hover" : "none";
  }

  /** One material per colour and lift, kept for the viewer's life: twelve hues, three states. */
  private shared(colour: string, lift: number): MeshStandardMaterial {
    const key = `${colour}|${String(lift)}`;
    const known = this.palette.get(key);
    if (known !== undefined) {
      return known;
    }
    const material = new MeshStandardMaterial({
      color: new Color(colour),
      roughness: 0.9,
      metalness: 0,
      // The display meshes are open shells cut out of a pot: a single-sided one is half a hole.
      side: DoubleSide,
      vertexColors: false,
    });
    if (lift > 0) {
      // Lifted in the fragment's own hue rather than in white, so an emphasised piece stays the
      // colour its group is and does not read as a fourth colour mode.
      material.emissive.copy(material.color).multiplyScalar(lift * 1.4);
    }
    this.palette.set(key, material);
    return material;
  }

  /** Whether a fragment is on screen at all: its group is shown, and it has a mesh and a pose. */
  private shows(member: Member): boolean {
    if (!member.holder.visible) {
      return false;
    }
    return this.inPair(member) ? this.pairNode.visible : (this.groups[member.group]?.node.visible ?? false);
  }

  /**
   * The seam over the pair on screen (A §8.3), rebuilt from scratch: two clouds of a few
   * thousand points each are cheaper to make again than to diff, and they change only when the
   * reviewer picks another row or the palette changes under them.
   *
   * A seam that names another pair is dropped rather than drawn. `PairDetail` is answered
   * asynchronously by the session, and a reviewer walking the queue with `A` and `X` will have
   * moved on by the time a slow one lands; the wrong seam on the right pair is the one mistake
   * on this screen a person cannot see.
   */
  private drawDetail(): void {
    this.releasePoints();
    // Before the early returns: a seam that has just been taken off the screen is a change, and
    // a viewer that draws on demand has to be told about it as much as about one that arrived.
    this.stage.invalidate();
    const pair = this.pair;
    const detail = this.detail;
    if (pair === null || detail === null || detail.a !== pair.a.name || detail.b !== pair.b.name) {
      return;
    }
    this.detailTheme = themeName();
    const cache = new Map<string, Color>();
    const of = (token: string, fallback: number): Color => {
      const known = cache.get(token);
      if (known !== undefined) {
        return known;
      }
      const colour = tokenColour(token, fallback);
      cache.set(token, colour);
      return colour;
    };
    this.contactCloud = this.cloud(detail.contact, (index) => {
      const row = contactColour(detail.contact_class[index] ?? Number.NaN);
      return of(row.token, row.fallback);
    });
    const white = new Color(SEAM_COLOUR);
    this.seamCloud = this.cloud(detail.seam, () => white);
  }

  /**
   * One cloud of points in A's frame. Everything here came over IPC as `f32` triples: a triple
   * that is not three finite numbers is left out rather than written into the buffer, where one
   * `NaN` would take the whole cloud's bounding sphere — and with it the framing — with it.
   */
  private cloud(points: readonly (readonly number[])[], colourAt: (index: number) => Color): Points | null {
    const xyz: number[] = [];
    const rgb: number[] = [];
    points.forEach((point, index) => {
      const [x, y, z] = point;
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        return;
      }
      const colour = colourAt(index);
      xyz.push(x ?? 0, y ?? 0, z ?? 0);
      rgb.push(colour.r, colour.g, colour.b);
    });
    if (xyz.length === 0) {
      return null;
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new Float32BufferAttribute(xyz, 3));
    geometry.setAttribute("color", new Float32BufferAttribute(rgb, 3));
    const cloud = new Points(geometry, this.pointMaterial);
    // The pair's node stands at the origin and never moves, so it *is* A's frame — which is the
    // frame `sherd_core::review::seam_view` gives these points in. No matrix of their own.
    this.pairNode.add(cloud);
    return cloud;
  }

  /** The two clouds' geometries back to the GPU; the material they wear is the viewer's own. */
  private releasePoints(): void {
    for (const cloud of [this.contactCloud, this.seamCloud]) {
      if (cloud !== null) {
        cloud.removeFromParent();
        cloud.geometry.dispose();
      }
    }
    this.contactCloud = null;
    this.seamCloud = null;
    this.detailTheme = null;
  }

  /**
   * The three contact colours are tokens of `styles.css` and the two themes give them different
   * values, so a theme switched with a pair on screen has to be caught. Same rule as the stage's
   * clear colour: only when the attribute that decides it changed, never per frame.
   */
  private syncDetailTheme(): void {
    if (this.detailTheme !== null && this.detailTheme !== themeName()) {
      this.drawDetail();
    }
  }

  /**
   * A ghost's clone off the scene. It shares the fragment's geometry — nothing here is its own.
   *
   * And the group [`setGhost`] put on screen for it goes back to being hidden: what the user
   * arranged (the eye of a row, «Без пары») is theirs, and a hover may not leave it changed.
   */
  private clearGhost(): void {
    this.ghostNode.clear();
    if (this.ghostGroup !== null) {
      this.ghostGroup = null;
      this.relayout();
    }
  }

  /**
   * What every ghost wears: B's orange of the review images, translucent and writing no depth,
   * so that the fragment it is proposed against reads through it rather than fighting it.
   */
  private ghost(): MeshStandardMaterial {
    this.ghostMaterial ??= new MeshStandardMaterial({
      color: new Color(PAIR_B_COLOUR),
      roughness: 0.9,
      metalness: 0,
      side: DoubleSide,
      transparent: true,
      opacity: GHOST_OPACITY,
      depthWrite: false,
    });
    return this.ghostMaterial;
  }

  /**
   * The pointer, at most once a frame and only when it moved (A §7.2: hovering must stay smooth
   * over 600 000 faces). Three gestures on one canvas: the hover, a click that is not the end of
   * an orbit, and a double click that flies to what is under it.
   */
  private watchPointer(): void {
    const { signal } = this.listeners;
    const canvas = this.stage.canvas;
    const local = (event: PointerEvent | MouseEvent): { x: number; y: number } => {
      const rect = canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    canvas.addEventListener(
      "pointermove",
      (event) => {
        this.pointer = local(event);
        this.pointerMoved = true;
        this.schedulePick();
      },
      { signal },
    );
    canvas.addEventListener(
      "pointerleave",
      () => {
        this.pointer = null;
        this.hover(null, null);
      },
      { signal },
    );
    canvas.addEventListener(
      "pointerdown",
      (event) => {
        this.down = event.button === 0 ? { x: event.clientX, y: event.clientY } : null;
      },
      { signal },
    );
    canvas.addEventListener(
      "pointerup",
      (event) => {
        const from = this.down;
        this.down = null;
        if (from === null || Math.hypot(event.clientX - from.x, event.clientY - from.y) > CLICK_SLOP) {
          return;
        }
        const at = local(event);
        const hit = this.pick(at.x, at.y);
        const name = hit?.name ?? null;
        this.select(name);
        this.events.onSelect(name);
      },
      { signal },
    );
    canvas.addEventListener(
      "dblclick",
      (event) => {
        const at = local(event);
        const hit = this.pick(at.x, at.y);
        if (hit !== null) {
          this.flyTo(hit.name);
        }
      },
      { signal },
    );
  }

  /** One pick a frame at most, and none at all while the pointer is still. */
  private schedulePick(): void {
    if (this.disposed || this.picking !== 0) {
      return;
    }
    this.picking = requestAnimationFrame(() => {
      this.picking = 0;
      if (this.disposed || !this.pointerMoved || this.pointer === null) {
        return;
      }
      this.pointerMoved = false;
      const at = this.pointer;
      this.hover(this.pick(at.x, at.y)?.name ?? null, at);
    });
  }

  /** What the tooltip is told, and the emphasis that goes with it. */
  private hover(name: string | null, at: { x: number; y: number } | null): void {
    if (this.hovered !== name) {
      const before = this.hovered;
      this.hovered = name;
      this.repaintOne(before);
      this.repaintOne(name);
      this.stage.invalidate();
    }
    this.events.onHover(name, name === null ? null : at);
  }

  /** The fragment under a point of the container, or `null` for the background. */
  private pick(x: number, y: number): Member | null {
    const rect = this.stage.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return null;
    }
    this.ndc.set((x / rect.width) * 2 - 1, -(y / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.ndc, this.stage.camera);
    // three.js's raycaster does not look at `visible`, so only what is shown is offered to it.
    const targets: Object3D[] = [];
    for (const member of this.order) {
      if (member.object !== null && this.shows(member)) {
        targets.push(member.object);
      }
    }
    const hit = this.raycaster.intersectObjects(targets, true)[0];
    if (hit === undefined) {
      return null;
    }
    for (let node: Object3D | null = hit.object; node !== null; node = node.parent) {
      const member = this.owner.get(node);
      if (member !== undefined) {
        return member;
      }
    }
    return null;
  }

  /**
   * «Подписи» (A §7.2), re-placed after every frame the stage drew, which is the only moment the
   * camera can have moved. Over [`LABEL_LIMIT`] names nothing is readable, so the layer narrows
   * to the selected group and, failing that, shows nothing at all.
   *
   * Under that many, the names still have to be kept off one another: two fragments a hand's width
   * apart in the pot are a few pixels apart on screen, and two names printed over each other say
   * less than one name does. The one in front wins, which is the one whose fragment the eye is on
   * — except for the chosen fragment, whose name is placed before every other, because the one
   * name a person who has just clicked a fragment is looking for is that one.
   */
  private placeLabels(): void {
    if (!this.labelsOn) {
      return;
    }
    const shown = this.order.filter((member) => this.shows(member));
    const group = this.selected === null ? this.single : (this.members.get(this.selected)?.group ?? null);
    let list = shown;
    if (list.length > LABEL_LIMIT) {
      list = group === null ? [] : shown.filter((member) => member.group === group);
    }
    if (list.length > LABEL_LIMIT) {
      list = [];
    }

    const width = this.stage.container.clientWidth;
    const height = this.stage.container.clientHeight;
    const wanted: { name: string; x: number; y: number; depth: number }[] = [];
    for (const member of list) {
      this.worldBox(member, TMP_BOX).getCenter(TMP_V).project(this.stage.camera);
      if (TMP_V.z > 1) {
        continue; // behind the camera
      }
      wanted.push({
        name: member.name,
        x: (TMP_V.x * 0.5 + 0.5) * width,
        y: (-TMP_V.y * 0.5 + 0.5) * height,
        depth: TMP_V.z,
      });
    }
    const chosen = this.selected;
    const first = (name: string): number => (name === chosen ? 0 : 1);
    wanted.sort((a, b) => first(a.name) - first(b.name) || a.depth - b.depth);

    const taken: { left: number; right: number; top: number; bottom: number }[] = [];
    let used = 0;
    for (const want of wanted) {
      const half = (want.name.length * LABEL_CHAR) / 2;
      const box = {
        left: want.x - half,
        right: want.x + half,
        top: want.y - LABEL_LINE / 2,
        bottom: want.y + LABEL_LINE / 2,
      };
      const clash = taken.some(
        (other) => box.left < other.right && other.left < box.right && box.top < other.bottom && other.top < box.bottom,
      );
      if (clash) {
        continue;
      }
      taken.push(box);
      const label = this.labelAt(used);
      label.textContent = want.name;
      label.style.left = `${String(want.x)}px`;
      label.style.top = `${String(want.y)}px`;
      label.hidden = false;
      used += 1;
    }
    for (let i = used; i < this.labelPool.length; i += 1) {
      const spare = this.labelPool[i];
      if (spare !== undefined) {
        spare.hidden = true;
      }
    }
  }

  /** The `n`-th label element, made on first use and reused for every frame after it. */
  private labelAt(n: number): HTMLDivElement {
    const known = this.labelPool[n];
    if (known !== undefined) {
      return known;
    }
    const label = document.createElement("div");
    // Colours from the tokens of `styles.css`; the halo is the viewport's own colour, which is
    // what makes a name readable over a pale sherd as well as over the background.
    label.style.cssText =
      "position:absolute;transform:translate(-50%,-50%);white-space:nowrap;font-size:10px;" +
      "color:var(--viewport-text);text-shadow:0 0 3px var(--viewport),0 1px 2px var(--viewport)";
    this.labelLayer.appendChild(label);
    this.labelPool.push(label);
    return label;
  }
}

/** How tall a packed block came out, discs and all — what the tray has to clear. */
function height(placed: readonly { id: number; y: number }[], rows: { id: number; group: GroupNode }[]): number {
  let [low, high] = [Infinity, -Infinity];
  for (const place of placed) {
    const radius = rows.find((row) => row.id === place.id)?.group.radius ?? 0;
    low = Math.min(low, place.y - radius);
    high = Math.max(high, place.y + radius);
  }
  return high > low ? high - low : 0;
}
