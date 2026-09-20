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
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

/**
 * A narrow enough field of view that a fragment reads as a solid object rather than as a
 * fish-eyed one, and wide enough that `fit()` does not park the camera in another postcode.
 */
const FOV = 35;

/**
 * Where the camera stands when a fragment is fitted: above, in front and to the right, with
 * `up = +Z`. The scans come out of the scanner Z-up, and a fracture face read straight on shows
 * no relief at all.
 */
const FROM = new Vector3(1, -1, 0.8).normalize();

/** A little air around the bounding sphere, so nothing touches the edge of the viewport. */
const MARGIN = 1.15;

/**
 * How many loaded scenes stay in memory. Stepping through the neighbours of a fragment in the
 * «Вход» list is the common move (A §7.3), and a re-parse of a GLB is tens of milliseconds and a
 * visible blink; eight of them is a few tens of megabytes at the display-mesh budget of A §7.2.
 */
const CACHE_LIMIT = 8;

/** The viewport's colour if the stylesheet has not loaded yet — `--viewport` of `styles.css`. */
const VIEWPORT_FALLBACK = 0x26262a;

/**
 * `child instanceof Mesh` on its own narrows to `Mesh<any, any, any>` — the class is generic and
 * `instanceof` fills its parameters with `any` — and nothing typed `any` is allowed to escape
 * into the rest of the window. Naming the default instantiation here keeps that one spot.
 */
function isMesh(object: Object3D): object is Mesh {
  return object instanceof Mesh;
}

/** A mesh's materials, whichever of the two shapes three.js gives them in. */
function materialsOf(mesh: Mesh): Material[] {
  return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
}

/**
 * Everything a loaded scene holds on the GPU. three.js frees neither geometries nor materials
 * when an object leaves the scene graph, and a workspace of 155 fragments walked through twice
 * would otherwise be two hundred buffers the driver is still holding.
 */
function disposeObject(object: Object3D): void {
  object.traverse((child) => {
    if (isMesh(child)) {
      child.geometry.dispose();
      for (const material of materialsOf(child)) {
        material.dispose();
      }
    }
  });
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
 * One fragment in 3D (A §7.2), with no React anywhere in it: a canvas, a camera that orbits, and
 * the display meshes the engine wrote into `fragments/`, fetched through Tauri's asset protocol.
 *
 * **It draws when something changed and not otherwise.** A free-running loop would keep a GPU
 * busy that the engine may be using for the very run whose progress the window is showing, so
 * every source of change — the controls, a load, a resize, the wireframe toggle — asks for one
 * frame through `invalidate()` and nothing asks for the next one.
 */
export class FragmentViewer {
  private readonly canvas: HTMLCanvasElement;
  private readonly renderer: WebGLRenderer;
  private readonly camera: PerspectiveCamera;
  private readonly scene = new Scene();
  private readonly controls: OrbitControls;
  private readonly loader = new GLTFLoader();
  private readonly observer: ResizeObserver;

  /** The last `CACHE_LIMIT` scenes by URL, oldest first — a `Map` keeps what was inserted when. */
  private readonly cache = new Map<string, Object3D>();

  /** What is on screen; `null` between a `show(null)` and the next fragment. */
  private current: Object3D | null = null;

  /** The handle of the frame already asked for, or `0` for «nothing is pending» — the dirty flag. */
  private frame = 0;

  /** Counts `show()` calls, so a slow earlier load cannot replace what a later one put up. */
  private loads = 0;

  private wireframe = false;

  /** The `data-theme` the clear colour was last read for; `null` until the first frame. */
  private themeSeen: string | null = null;

  private disposed = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    // `low-power` asks a laptop with two GPUs for the integrated one: this is a single fragment
    // under a hemisphere light, and the discrete card may be busy being the engine's backend.
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

    this.observer = new ResizeObserver(() => {
      this.resize();
    });
    const parent = canvas.parentElement;
    if (parent !== null) {
      this.observer.observe(parent);
    }
    this.resize();
  }

  /**
   * Shows the GLB at `url`, or nothing for `null`. Resolves when it is on screen; a newer call
   * wins over an older one still loading, and the loser's work is kept in the cache rather than
   * thrown away — the user who clicked past a fragment usually comes back to it.
   */
  async show(url: string | null): Promise<void> {
    const token = ++this.loads;
    if (url === null) {
      this.display(null);
      return;
    }

    const cached = this.cache.get(url);
    const object = cached ?? (await this.loader.loadAsync(url)).scene;
    this.remember(url, object);
    if (token !== this.loads || this.disposed) {
      return;
    }
    this.display(object);
    this.fit();
  }

  /** The wireframe toggle of the «Вход» mode's viewport (A §7.3). */
  setWireframe(on: boolean): void {
    this.wireframe = on;
    if (this.current !== null) {
      this.applyWireframe(this.current);
    }
    this.invalidate();
  }

  /**
   * Puts the whole fragment on screen (`F`, A §7.1). The near and far planes are derived from the
   * object and not fixed: the scans are in millimetres with coordinates in the hundreds, and a
   * `near` of 0.1 against a `far` of 1000 has no depth precision left at that distance.
   */
  fit(): void {
    const object = this.current;
    if (object === null) {
      return;
    }
    const sphere = new Box3().setFromObject(object).getBoundingSphere(new Sphere());
    // An empty box answers with a radius of −1, and a mesh with a broken vertex with NaN.
    const radius = Number.isFinite(sphere.radius) && sphere.radius > 0 ? sphere.radius : 1;
    const distance = (radius / Math.sin(MathUtils.degToRad(FOV) / 2)) * MARGIN;

    this.camera.near = radius / 100;
    this.camera.far = radius * 100;
    this.camera.up.set(0, 0, 1);
    this.camera.position.copy(sphere.center).addScaledVector(FROM, distance);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(sphere.center);
    this.controls.update();
    this.invalidate();
  }

  /** Gives the GPU everything back: the window may live for hours after the last fragment. */
  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    if (this.frame !== 0) {
      cancelAnimationFrame(this.frame);
      this.frame = 0;
    }
    this.observer.disconnect();
    this.controls.dispose();
    for (const object of this.cache.values()) {
      disposeObject(object);
    }
    this.cache.clear();
    this.current = null;
    this.renderer.dispose();
    // Without this the context lingers until the garbage collector feels like it, and a browser
    // gives out only a handful of them.
    this.renderer.forceContextLoss();
  }

  /** Swaps what the scene holds; the outgoing object stays in the cache, alive and undisposed. */
  private display(object: Object3D | null): void {
    if (this.current !== null) {
      this.scene.remove(this.current);
    }
    this.current = object;
    if (object !== null) {
      this.applyWireframe(object);
      this.scene.add(object);
    }
    this.invalidate();
  }

  /** Newest last, oldest evicted — and what is on screen is never evicted at all. */
  private remember(url: string, object: Object3D): void {
    if (this.disposed) {
      disposeObject(object);
      return;
    }
    this.cache.delete(url);
    this.cache.set(url, object);
    for (const [key, evicted] of this.cache) {
      if (this.cache.size <= CACHE_LIMIT) {
        break;
      }
      // The displayed scene is usually the newest, but not always: a load that lost the race is
      // remembered after the one that won it, and freeing the winner's buffers under the camera
      // would empty the viewport. One entry over the limit is the cheaper of the two.
      if (evicted === this.current) {
        continue;
      }
      this.cache.delete(key);
      disposeObject(evicted);
    }
  }

  /**
   * The GLBs come with the materials the engine chose — vertex-coloured white for a scan, plain
   * clay for one without colour, both double-sided and non-metallic — so nothing here replaces
   * them; the toggle only flips the one flag on whatever is there.
   */
  private applyWireframe(object: Object3D): void {
    const on = this.wireframe;
    object.traverse((child) => {
      if (isMesh(child)) {
        for (const material of materialsOf(child)) {
          if ("wireframe" in material) {
            material.wireframe = on;
          }
        }
      }
    });
  }

  /** Asks for one frame, and for one only: the handle is the flag. */
  private invalidate(): void {
    if (this.disposed || this.frame !== 0) {
      return;
    }
    this.frame = requestAnimationFrame(() => {
      this.render();
    });
  }

  private render(): void {
    this.frame = 0;
    if (this.disposed) {
      return;
    }
    this.syncClearColour();
    this.renderer.render(this.scene, this.camera);
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
   * The canvas is sized by its parent through CSS; `false` keeps the renderer from writing a
   * width and a height back onto it and fighting the layout.
   */
  private resize(): void {
    const parent = this.canvas.parentElement;
    const width = parent?.clientWidth ?? 0;
    const height = parent?.clientHeight ?? 0;
    if (width === 0 || height === 0) {
      return;
    }
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }
}
