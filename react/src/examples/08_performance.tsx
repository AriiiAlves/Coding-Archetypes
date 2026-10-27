/**
 * 08 - PERFORMANCE: keep the UI responsive.
 *
 * React re-renders a component when its state changes, and re-renders all of its
 * children by default. That is usually fast enough: measure before optimizing
 * (React DevTools > Profiler, "Highlight updates when components render").
 *
 * What's covered:
 *   - memo: skip re-rendering a child whose props didn't change
 *   - useMemo: cache an expensive calculation
 *   - useCallback: keep a function's identity stable for memoized children
 *   - useTransition / useDeferredValue: keep typing smooth during heavy updates
 *   - lazy + Suspense: split code and load it on demand
 *
 * The React Compiler (babel-plugin-react-compiler) adds memo/useMemo/useCallback
 * automatically at build time. With it enabled, most manual memoization below is
 * unnecessary. It is still worth knowing what it does.
 */
import { Suspense, lazy, memo, useCallback, useDeferredValue, useMemo, useRef, useState, useTransition } from "react";

import { Card } from "./01_components_props";

const LazyPanel = lazy(() => import("./08_lazy_panel")); // declared at module level, never inside a component

// A large list to make re-renders measurable.
const ITEMS = Array.from({ length: 20_000 }, (_, i) => ({ id: i, label: `Item ${i.toString().padStart(5, "0")}` }));

// Deliberately slow: stands in for heavy formatting, charts, big tables.
function slowFilter(query: string) {
  const start = performance.now();
  while (performance.now() - start < 30) { /* burn 30 ms */ }
  return ITEMS.filter((item) => item.label.includes(query)).slice(0, 200);
}

// Shows how many times a component rendered: handy while learning.
// In development, StrictMode renders twice, so counts go up in steps of 2.
function RenderCount({ label }: { label: string }) {
  const count = useRef(0);
  count.current += 1;
  return <small className="muted">{label} rendered {count.current}×</small>;
}

// --------------------------------------------------------------------------- //
// 1. memo + useCallback + useMemo
// --------------------------------------------------------------------------- //
type RowListProps = { items: { id: number; label: string }[]; onSelect: (id: number) => void };

// memo: re-render only when props change (compared with Object.is, one level deep).
const RowList = memo(function RowList({ items, onSelect }: RowListProps) {
  return (
    <div>
      <RenderCount label="RowList" />
      <ul className="scroll-box">
        {items.map((item) => (
          <li key={item.id}><button className="link" onClick={() => onSelect(item.id)}>{item.label}</button></li>
        ))}
      </ul>
    </div>
  );
});

function MemoDemo() {
  const [query, setQuery] = useState("00");
  const [selected, setSelected] = useState<number | null>(null);
  const [unrelated, setUnrelated] = useState(0);

  // Without useMemo, slowFilter runs on EVERY render, including "unrelated" clicks.
  const items = useMemo(() => slowFilter(query), [query]);

  // Without useCallback, a new function is created on every render, so memo(RowList)
  // would see a "changed" prop and re-render anyway.
  const handleSelect = useCallback((id: number) => setSelected(id), []);

  return (
    <div className="stack">
      <div className="row">
        <input value={query} onChange={(e) => setQuery(e.target.value)} aria-label="filter" />
        <button className="secondary" onClick={() => setUnrelated((n) => n + 1)}>Unrelated state: {unrelated}</button>
        <RenderCount label="Parent" />
      </div>
      <p className="muted">Selected: {selected ?? "none"} · Click “Unrelated state”: the parent re-renders, RowList doesn't.</p>
      <RowList items={items} onSelect={handleSelect} />
    </div>
  );
}

// --------------------------------------------------------------------------- //
// 2. useTransition: mark a state update as non-urgent. React keeps the input
//    responsive and renders the slow part in the background, interruptibly.
// --------------------------------------------------------------------------- //
function TransitionDemo() {
  const [text, setText] = useState("");          // urgent: the input itself
  const [query, setQuery] = useState("");        // non-urgent: what the slow list uses
  const [isPending, startTransition] = useTransition();
  const items = useMemo(() => slowFilter(query), [query]);

  return (
    <div className="stack">
      <input
        value={text}
        onChange={(e) => {
          setText(e.target.value);                         // update now
          startTransition(() => setQuery(e.target.value)); // update when there's time
        }}
        placeholder="Type fast: the input stays smooth"
        aria-label="transition filter"
      />
      <p className="muted">{isPending ? "Updating list…" : `${items.length} results`}</p>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// 3. useDeferredValue: the same idea when you don't own the state setter
//    (e.g. the value comes from props). The deferred copy lags behind.
// --------------------------------------------------------------------------- //
const SlowResults = memo(function SlowResults({ query }: { query: string }) {
  const items = slowFilter(query);
  return <p className="muted">{items.length} results for “{query}”</p>;
});

function DeferredDemo() {
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const stale = query !== deferredQuery;
  return (
    <div className="stack">
      <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Deferred filter" aria-label="deferred filter" />
      <div style={{ opacity: stale ? 0.5 : 1 }}>
        <SlowResults query={deferredQuery} />
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// 4. Code splitting with lazy + Suspense
// --------------------------------------------------------------------------- //
function LazyDemo() {
  const [show, setShow] = useState(false);
  return (
    <div className="stack">
      <button onClick={() => setShow(true)} disabled={show}>Load panel</button>
      {show && (
        <Suspense fallback={<p className="muted">Loading panel…</p>}>
          <LazyPanel />
        </Suspense>
      )}
    </div>
  );
}

export default function PerformanceExample() {
  return (
    <>
      <h1>08 · Performance</h1>
      <Card title="memo, useMemo, useCallback"><MemoDemo /></Card>
      <Card title="useTransition"><TransitionDemo /></Card>
      <Card title="useDeferredValue"><DeferredDemo /></Card>
      <Card title="lazy + Suspense (code splitting)"><LazyDemo /></Card>
      <Card title="Other tips">
        <ul>
          <li>Keep state as close as possible to where it's used, so fewer components re-render.</li>
          <li>Pass components as <code>children</code>: they don't re-render when the wrapper's state changes.</li>
          <li>Render very long lists with virtualization (TanStack Virtual, react-window).</li>
          <li>Check bundle size with <code>npm run build</code>: Vite prints each chunk's size.</li>
        </ul>
      </Card>
    </>
  );
}
