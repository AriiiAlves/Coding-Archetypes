/**
 * Loaded on demand by 08_performance.tsx through React.lazy.
 * Vite puts this file (and anything only it imports) in a separate JS chunk,
 * downloaded the first time it renders. Lazy modules must have a default export.
 */
export default function LazyPanel() {
  return (
    <p>
      This component was downloaded only when you asked for it. Open the browser's Network tab
      to see the separate chunk. Use this for heavy, rarely used parts: charts, editors, admin pages.
    </p>
  );
}
