import {
  Box3,
  Color,
  DirectionalLight,
  HemisphereLight,
  MathUtils,
  Mesh,
  type Material,
  type Object3D,
  PerspectiveCamera,
  Scene,
  Sphere,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { PrincipalAxes } from "./principal";

/**
 * The floor both viewers stand on (A §7.2): a canvas of one's own, a renderer, a camera that
 * orbits, the lighting a sherd's relief needs, and the dirty flag that makes the window draw when
 * something changed and not otherwise.
 *
 * It exists because «Вход» shows one fragment and «Сборка» shows a hundred and fifty-five, and
 * everything above — what is loaded, what it is coloured with, what is picked — differs between
 * them while none of this does. Two copies of it would drift, and the half that drifts is always
 * the disposal.
 */

/**
 * A narrow enough field of view that a fragment reads as a solid object rather than as a
 * fish-eyed one, and wide enough that framing does not park the camera in another postcode.
 */
const FOV = 35;

/**
 * Where the camera stands when there is no face to be shown from, and the side of a wall it
 * prefers when there is one: above, in front and to the right, with `up = +Z`, because the scans
 * come out of the scanner Z-up.
 */
export const FROM = new Vector3(1, -1, 0.8).normalize();

/**
 * How far off face-on a fragment is shown, towards its long side. A sherd met by a fixed
 * direction is usually met edge-on, which shows nothing; met exactly face-on its relief is flat.
 * The engine's own previews frame a fragment by its principal axes for the same reason.
 */
const TILT = MathUtils.degToRad(25);

/** A little air around the bounding sphere, so nothing touches the edge of the viewport. */
const MARGIN = 1.15;

/** The viewport's colour if the stylesheet has not loaded yet — `--viewport` of `styles.css`. */
const VIEWPORT_FALLBACK = 0x26262a;

/**
 * `child instanceof Mesh` on its own narrows to `Mesh<any, any, any>` — the class is generic and
 * `instanceof` fills its parameters with `any` — and nothing typed `any` is allowed to escape
 * into the rest of the window. Naming the default instantiation here keeps that one spot.
 */
export function isMesh(object: Object3D): object is Mesh {
  return object instanceof Mesh;
}

/** A mesh's materials, whichever of the two shapes three.js gives them in. */
export function materialsOf(mesh: Mesh): Material[] {
  return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
}

/**
 * Everything a loaded scene holds on the GPU. three.js frees neither geometries nor materials
 * when an object leaves the scene graph, and a workspace of 155 fragments walked through twice
 * would otherwise be two hundred buffers the driver is still holding.
 */
export function disposeObject(object: Object3D): void {
  object.traverse((child) => {
    if (isMesh(child)) {
      child.geometry.dispose();
      for (const material of materialsOf(child)) {
        material.dispose();
      }
    }
  });
}

/** The bounding sphere of an object as the world sees it; empty when it holds no geometry. */
export function sphereOf(object: Object3D): Sphere {
  return new Box3().setFromObject(object).getBoundingSphere(new Sphere());
}

/** The clear colour, from the palette of `styles.css` — the viewport is dark in both themes. */
function viewportColour(): Color {
  const fallback = new Color(VIEWPORT_FALLBACK);
  if (typeof document === "undefined") {
    return fallback;
  }
  const raw = getComputedStyle(document.documentElement).getPropertyValue("--viewport").trim();
  if (raw === "") {
    return fallback;
  }
  try {
    return new Color(raw);
  } catch {
    return fallback;
  }
}

/**
 * Which way to look at something, given the principal axes of what is being looked at (or `null`
 * for «could not tell»): down its wall's normal, tilted towards its long side. A block of groups
 * lying in the XY plane has its least-variance axis through Z and so is looked at from above at
 * that same tilt, which is what A §7.2's spread layout wants.
 *
 * `across` decides which way round the thing then stands. One fragment is shown on its long side,
 * as the «Вход» viewer has always drawn it. A block of groups is *packed* half again as wide as
 * it is tall (`layout.ts`), for a viewport that is wider than it is tall, and must be met that
 * way round: its long axis lies across the screen and the remaining axis is up. That third axis
 * is square to the view by construction, so unlike `+Z` it can never be the degenerate «up» that
 * a camera looking straight down cannot use.
 */
export function viewDirection(axes: PrincipalAxes | null, across = false): { from: Vector3; up: Vector3 } {
  const from = FROM.clone();
  const up = new Vector3(0, 0, 1);
  if (axes === null) {
    return { from, up };
  }
  const normal = new Vector3(...axes.normal);
  const major = new Vector3(...axes.major);
  if (normal.dot(FROM) < 0) {
    normal.negate();
  }
  from.copy(normal).multiplyScalar(Math.cos(TILT)).addScaledVector(major, Math.sin(TILT)).normalize();
  if (across) {
    const sideways = new Vector3().crossVectors(normal, major);
    if (sideways.lengthSq() > 1e-12) {
      up.copy(sideways.normalize());
    }
    // Looking down the scanner's own vertical there is no "up" left in it: the long side then.
  } else if (Math.abs(from.z) > 0.9) {
    up.copy(major);
  }
  return { from, up };
}

/** What a viewer wants to be told about its own stage. */
export interface StageEvents {
  /** After every frame the stage drew — where an overlay that follows the camera is re-placed. */
  onRender?: (() => void) | undefined;
  /** The first time the user grabs the controls; a viewer then stops re-framing under them. */
  onUserMove?: (() => void) | undefined;
}

/**
 * **It draws when something changed and not otherwise.** A free-running loop would keep a GPU
 * busy that the engine may be using for the very run whose progress the window is showing, so
 * every source of change — the controls, a load, a resize, a colour mode — asks for one frame
 * through `invalidate()` and nothing asks for the next one.
 *
 * **It makes its own canvas** inside the element it is given, and takes it away again in
 * `dispose()`. A canvas outlives the WebGL context `dispose()` gives back, and a second viewer
 * built on the same element would be handed that same lost context by `getContext` — three.js
 * reads a shader precision off it before anything can check, and throws. Owning the canvas means
 * a viewer can never be handed another viewer's dead context, which is exactly what React's
 * StrictMode does on every mount in development.
 */
export class Stage {
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly canvas: HTMLCanvasElement;

  /** The element the canvas lives in: what the viewport's size is read from, and what an overlay hangs on. */
  readonly container: HTMLElement;

  private readonly renderer: WebGLRenderer;
  private readonly observer: ResizeObserver;
  private readonly events: StageEvents;

  /** The handle of the frame already asked for, or `0` for «nothing is pending» — the dirty flag. */
  private pending = 0;

  /** The `data-theme` the clear colour was last read for; `null` until the first frame. */
  private themeSeen: string | null = null;

  private ended = false;

  constructor(container: HTMLElement, events: StageEvents = {}) {
    this.container = container;
    this.events = events;
    const canvas = document.createElement("canvas");
    // The renderer is told not to write a width and a height into the style (see `resize()`), so
    // the canvas is stretched over its container here, once. Three declarations and no class: the
    // element belongs to this file and nothing else may lay it out.
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    container.appendChild(canvas);
    this.canvas = canvas;
    // `low-power` asks a laptop with two GPUs for the integrated one: this is a scene under a
    // hemisphere light, and the discrete card may be busy being the engine's backend.
    this.renderer = new WebGLRenderer({ canvas, antialias: true, powerPreference: "low-power" });
    this.renderer.outputColorSpace = SRGBColorSpace;
    // Above 2 the cost is quadratic and nobody can see it.
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.camera = new PerspectiveCamera(FOV, 1, 0.1, 1000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.copy(FROM);

    this.scene.add(new HemisphereLight(0xffffff, 0x3a3a40, 1.1));
    // A child of the camera, so the relief of a fracture reads from every angle: a light fixed to
    // the scene leaves half the orbit flat, and flat is exactly where a join cannot be judged.
    const key = new DirectionalLight(0xffffff, 1.6);
    key.position.set(0.6, 0.8, 1);
    this.camera.add(key);
    // A light only lights what is in the scene graph with it, camera children included.
    this.scene.add(this.camera);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    // Damping would keep asking for frames after the mouse stopped; see the class's note.
    this.controls.enableDamping = false;
    this.controls.addEventListener("change", () => {
      this.invalidate();
    });
    this.controls.addEventListener("start", () => {
      this.events.onUserMove?.();
    });

    this.observer = new ResizeObserver(() => {
      this.resize();
    });
    this.observer.observe(container);
    this.resize();
  }

  /** Whether `dispose()` has been called: an arriving load must not touch a dead renderer. */
  get disposed(): boolean {
    return this.ended;
  }

  /** Asks for one frame, and for one only: the handle is the flag. */
  invalidate(): void {
    if (this.ended || this.pending !== 0) {
      return;
    }
    this.pending = requestAnimationFrame(() => {
      this.render();
    });
  }

  /**
   * Draws at once, on this tick. Only «Снимок PNG» needs it: a canvas without
   * `preserveDrawingBuffer` has nothing readable in it unless it was drawn in the same turn of
   * the event loop as `toDataURL`, and keeping that buffer alive costs every other frame.
   */
  renderNow(): void {
    if (this.ended) {
      return;
    }
    if (this.pending !== 0) {
      cancelAnimationFrame(this.pending);
    }
    this.render();
  }

  /**
   * Puts a sphere in front of the camera from `from`, with `up` above. The near and far planes
   * are derived from the sphere and not fixed: the scans are in millimetres with coordinates in
   * the hundreds, and a `near` of 0.1 against a `far` of 1000 has no depth precision left there.
   */
  frame(sphere: Sphere, from: Vector3, up: Vector3): void {
    // An empty box answers with a radius of −1, and a mesh with a broken vertex with NaN.
    const radius = Number.isFinite(sphere.radius) && sphere.radius > 0 ? sphere.radius : 1;
    const centre = sphere.center;
    if (!Number.isFinite(centre.x) || !Number.isFinite(centre.y) || !Number.isFinite(centre.z)) {
      centre.set(0, 0, 0);
    }
    const distance = (radius / Math.sin(MathUtils.degToRad(FOV) / 2)) * MARGIN;
    this.camera.near = radius / 100;
    this.camera.far = radius * 100;
    this.camera.up.copy(up);
    this.camera.position.copy(centre).addScaledVector(from, distance);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(centre);
    this.controls.update();
    this.invalidate();
  }

  /**
   * The same, for a cloud of points (`x, y, z, x, y, z, …`) rather than a sphere.
   *
   * A run spread over a plane is flat, wide and made of a dozen clusters with air between them.
   * Its bounding *sphere* is its diagonal, and its bounding *box* has corners where nothing is,
   * so framing by either leaves the assembly at half the size of the viewport and throws away
   * the shape `layout.ts` worked to pack. This measures the points themselves along the camera's
   * own three axes and fits the two that are seen, which is also what centres the view: the
   * middle of what is on screen, not the middle of a box around it.
   */
  framePoints(points: ArrayLike<number>, from: Vector3, up: Vector3): void {
    const forward = from.clone();
    if (forward.lengthSq() < 1e-12) {
      forward.copy(FROM);
    }
    forward.normalize();
    const right = new Vector3().crossVectors(up, forward);
    if (right.lengthSq() < 1e-12) {
      // `up` looks straight down the view: any direction square to it will do as «right».
      right.crossVectors(new Vector3(0, 0, 1), forward);
      if (right.lengthSq() < 1e-12) {
        right.set(1, 0, 0);
      }
    }
    right.normalize();
    const above = new Vector3().crossVectors(forward, right).normalize();

    const low = [Infinity, Infinity, Infinity];
    const high = [-Infinity, -Infinity, -Infinity];
    const point = new Vector3();
    for (let i = 0; i + 2 < points.length; i += 3) {
      point.set(points[i] ?? NaN, points[i + 1] ?? NaN, points[i + 2] ?? NaN);
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) {
        continue;
      }
      [right, above, forward].forEach((axis, k) => {
        const along = point.dot(axis);
        low[k] = Math.min(low[k] ?? Infinity, along);
        high[k] = Math.max(high[k] ?? -Infinity, along);
      });
    }
    const span = (k: number): number => ((high[k] ?? 0) - (low[k] ?? 0)) / 2;
    const mid = (k: number): number => ((high[k] ?? 0) + (low[k] ?? 0)) / 2;
    if (!Number.isFinite(span(0)) || !Number.isFinite(span(1)) || !Number.isFinite(span(2))) {
      return;
    }

    const tangent = Math.tan(MathUtils.degToRad(FOV) / 2);
    const aspect = this.camera.aspect > 0 ? this.camera.aspect : 1;
    // Far enough back that both the seen extents fit at the *near* face of what is being framed,
    // which is what the half-depth on top is for: a block met at a tilt is deeper than it looks.
    const distance = Math.max(span(1) / tangent, span(0) / (tangent * aspect)) * MARGIN + span(2);
    const scale = Math.max(span(0), span(1), span(2), 1);
    const centre = new Vector3()
      .addScaledVector(right, mid(0))
      .addScaledVector(above, mid(1))
      .addScaledVector(forward, mid(2));

    this.camera.near = scale / 100;
    this.camera.far = distance + span(2) + scale * 4;
    this.camera.up.copy(above);
    this.camera.position.copy(centre).addScaledVector(forward, distance);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(centre);
    this.controls.update();
    this.invalidate();
  }

  /** Gives the GPU everything back: the window may live for hours after the last fragment. */
  dispose(): void {
    if (this.ended) {
      return;
    }
    this.ended = true;
    if (this.pending !== 0) {
      cancelAnimationFrame(this.pending);
      this.pending = 0;
    }
    this.observer.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    // Without this the context lingers until the garbage collector feels like it, and a browser
    // gives out only a handful of them.
    this.renderer.forceContextLoss();
    // And the canvas goes with the context it carries: a lost context is never given back by
    // `getContext`, so an element left behind here would be a trap for the next viewer.
    this.canvas.remove();
  }

  private render(): void {
    this.pending = 0;
    if (this.ended) {
      return;
    }
    this.syncClearColour();
    this.renderer.render(this.scene, this.camera);
    this.events.onRender?.();
  }

  /**
   * The palette has one `--viewport` per theme and the user can switch themes with the viewer
   * open. Reading a custom property costs a style recalculation, so it happens only when the
   * attribute that decides it has actually changed — never on every frame of an orbit.
   */
  private syncClearColour(): void {
    const theme = document.documentElement.dataset.theme ?? "";
    if (theme === this.themeSeen) {
      return;
    }
    this.themeSeen = theme;
    this.renderer.setClearColor(viewportColour());
  }

  /**
   * The canvas is sized by its container through CSS; `false` keeps the renderer from writing a
   * width and a height back onto it and fighting the layout.
   */
  private resize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) {
      return;
    }
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }
}
