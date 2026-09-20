/**
 * The window's root. For now it is the product name and nothing else: the frame, its panes and
 * the modes arrive in tasks 5 to 8 (A §5). It is here so that the scaffold has something to
 * build, and so that `pnpm dev` shows the tokens of `styles.css` actually applied.
 */
export default function App() {
  return (
    <div className="flex h-full items-center justify-center bg-bg text-text">
      <h1 className="text-2xl font-semibold tracking-tight">Sherd Refit</h1>
    </div>
  );
}
