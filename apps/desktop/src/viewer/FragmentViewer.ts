import { type Object3D } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { type PrincipalAxes, principalAxes } from "./principal";
import { disposeObject, isMesh, materialsOf, sphereOf, Stage, viewDirection } from "./stage";

/**
 * How many loaded scenes stay in memory. Stepping through the neighbours of a fragment in the
 * «Вход» list is the common move (A §7.3), and a re-parse of a GLB is tens of milliseconds and a
 * visible blink; eight of them is a few tens of megabytes at the display-mesh budget of A §7.2.
 */
const CACHE_LIMIT = 8;

/**
 * One fragment in 3D (A §7.2), with no React anywhere in it: a canvas, a camera that orbits, and
 * the display meshes the engine wrote into `fragments/`, fetched through Tauri's asset protocol.
 *
 * The renderer, the camera, the lights, the dirty flag and the disposal are [`Stage`]'s and are
 * shared with the assembly's viewer; what is here is only what showing *one* fragment adds: a
 * small cache of parsed scenes, the wireframe toggle, and framing by principal axes.
 */
export class FragmentViewer {
  private readonly stage: Stage;
  private readonly loader = new GLTFLoader();

  /** The last `CACHE_LIMIT` scenes by URL, oldest first — a `Map` keeps what was inserted when. */
  private readonly cache = new Map<string, Object3D>();

  /**
   * The loads under way, by URL. Two `show()`s of one fragment before either resolved — the
   * arrow keys run back and forth over a slow GLB, StrictMode mounts the pane twice — would
   * otherwise parse it twice and put two scenes in the cache under one key, of which only the
   * second is ever disposed. The first would be geometry the driver holds until the window closes.
   */
  private readonly loading = new Map<string, Promise<Object3D>>();

  /** What is on screen; `null` between a `show(null)` and the next fragment. */
  private current: Object3D | null = null;

  /** Counts `show()` calls, so a slow earlier load cannot replace what a later one put up. */
  private loads = 0;

  private wireframe = false;

  private disposed = false;
  private readonly axes = new WeakMap<Object3D, PrincipalAxes | null>();

  constructor(container: HTMLElement) {
    this.stage = new Stage(container);
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
    const object = cached ?? (await this.load(url));
    this.remember(url, object);
    if (token !== this.loads || this.disposed) {
      return;
    }
    this.display(object);
    this.fit();
  }

  /**
   * One parse of a GLB at a time, whatever asks for it: callers that arrive while a URL is in
   * flight are handed that same promise, and the entry is forgotten as soon as it settles — a
   * load that failed must be retryable, and one that succeeded is in the cache from then on.
   */
  private load(url: string): Promise<Object3D> {
    const already = this.loading.get(url);
    if (already !== undefined) {
      return already;
    }
    const loading = this.loader
      .loadAsync(url)
      .then((gltf) => gltf.scene)
      .finally(() => {
        this.loading.delete(url);
      });
    this.loading.set(url, loading);
    return loading;
  }

  /** The wireframe toggle of the «Вход» mode's viewport (A §7.3). */
  setWireframe(on: boolean): void {
    this.wireframe = on;
    if (this.current !== null) {
      this.applyWireframe(this.current);
    }
    this.stage.invalidate();
  }

  /** Puts the whole fragment on screen (`F`, A §7.1), face-on to its wall and tilted. */
  fit(): void {
    const object = this.current;
    if (object === null) {
      return;
    }
    const { from, up } = viewDirection(this.axesOf(object));
    this.stage.frame(sphereOf(object), from, up);
  }

  /** A loaded fragment's principal axes, worked out once: `fit()` is also the `F` key. */
  private axesOf(object: Object3D): PrincipalAxes | null {
    const known = this.axes.get(object);
    if (known !== undefined) {
      return known;
    }
    let found: PrincipalAxes | null = null;
    object.traverse((child) => {
      if (found === null && isMesh(child)) {
        found = principalAxes(child.geometry.getAttribute("position").array);
      }
    });
    this.axes.set(object, found);
    return found;
  }

  /** Gives the GPU everything back: the window may live for hours after the last fragment. */
  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    for (const object of this.cache.values()) {
      disposeObject(object);
    }
    this.cache.clear();
    // A load still in flight is not waited for; it resolves into `remember`, which disposes of
    // what it was handed once `disposed` is set.
    this.loading.clear();
    this.current = null;
    this.stage.dispose();
  }

  /** Swaps what the scene holds; the outgoing object stays in the cache, alive and undisposed. */
  private display(object: Object3D | null): void {
    if (this.current !== null) {
      this.stage.scene.remove(this.current);
    }
    this.current = object;
    if (object !== null) {
      this.applyWireframe(object);
      this.stage.scene.add(object);
    }
    this.stage.invalidate();
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
}
