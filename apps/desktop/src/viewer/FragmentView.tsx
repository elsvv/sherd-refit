import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { toCommandError } from "../ipc/api";
import { FragmentViewer } from "./FragmentViewer";

/** What to show, how to show it, and when the user asked for it to be fitted again. */
interface FragmentViewProps {
  /** The asset-protocol URL of a display mesh, or `null` for an empty viewport. */
  url: string | null;
  wireframe: boolean;
  /** `useFitSignal`'s counter: every increment is one press of `F` or of «Вписать» (A §7.1). */
  fitSignal: number;
}

/**
 * The React side of the viewer, and the whole of it (A §7.2: `viewer/` is three.js with no React
 * in it). This component owns one `FragmentViewer` for its lifetime and does nothing but tell it
 * what changed — React never sees the scene, and the scene never re-renders React.
 *
 * A `<canvas>` cannot report what it is showing, so the two things the user needs in words —
 * that a model is on its way, and that it did not arrive — are drawn over it as a line of text.
 *
 * The canvas itself is the viewer's, not React's: the effect hands over an empty `<div>` that
 * React never puts a child into, and each viewer makes and removes its own canvas inside it. A
 * canvas React kept across StrictMode's mount → cleanup → mount would come back to the second
 * viewer with the first one's WebGL context on it, lost, and three.js throws on a lost context.
 */
export default function FragmentView({ url, wireframe, fitSignal }: FragmentViewProps) {
  const { t } = useTranslation();
  const host = useRef<HTMLDivElement | null>(null);
  const [viewer, setViewer] = useState<FragmentViewer | null>(null);
  const [loading, setLoading] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  // One viewer per mounted component. It is kept in state and not in a ref on purpose: the
  // effects below have to run again once it exists, and a ref changing tells React nothing.
  useEffect(() => {
    const element = host.current;
    if (element === null) {
      return undefined;
    }
    const created = new FragmentViewer(element);
    setViewer(created);
    return () => {
      setViewer(null);
      created.dispose();
    };
  }, []);

  useEffect(() => {
    if (viewer === null) {
      return undefined;
    }
    let gone = false;
    setFailure(null);
    setLoading(url !== null);
    void viewer.show(url).then(
      () => {
        if (!gone) {
          setLoading(false);
        }
      },
      (e: unknown) => {
        // A missing file, a scope the asset protocol refuses, a GLB the parser chokes on: the
        // fragment is one of many and the window carries on with the rest of the collection.
        if (!gone) {
          setLoading(false);
          setFailure(toCommandError(e).message);
        }
      },
    );
    return () => {
      gone = true;
    };
  }, [viewer, url]);

  useEffect(() => {
    viewer?.setWireframe(wireframe);
  }, [viewer, wireframe]);

  // `fitSignal` is read for its change and not for its value; a new viewer starts fitted anyway.
  useEffect(() => {
    viewer?.fit();
  }, [viewer, fitSignal]);

  let message: string | null = null;
  if (failure !== null) {
    message = t("viewer.failed", { message: failure });
  } else if (loading) {
    message = t("viewer.loading");
  }

  return (
    <div className="relative h-full w-full">
      <div ref={host} className="h-full w-full" />
      {message === null ? null : (
        // `pointer-events-none`: the orbit keeps working under a line that is only there to read.
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center px-6 text-center text-xs text-viewport-text">
          {message}
        </div>
      )}
    </div>
  );
}
