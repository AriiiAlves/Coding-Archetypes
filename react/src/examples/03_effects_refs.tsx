/**
 * 03 - EFFECTS & REFS: talking to the world outside React.
 *
 * useEffect runs code AFTER render to synchronize with an external system:
 * the DOM, timers, subscriptions, the network, browser APIs.
 * useRef holds a value that survives re-renders WITHOUT causing one, and
 * gives you direct access to DOM nodes.
 *
 * What's covered:
 *   - effect dependencies: [] (once), [a, b] (when a or b change), none (every render)
 *   - cleanup functions (timers, listeners, subscriptions)
 *   - refs to DOM nodes (focus, scroll, measure)
 *   - refs as mutable "instance variables" (interval ids, previous values)
 *   - when NOT to use an effect
 */
import { useEffect, useRef, useState, type ComponentProps } from "react";

import { Card } from "./01_components_props";

// --------------------------------------------------------------------------- //
// 1. Sync with a browser API whenever a value changes.
// --------------------------------------------------------------------------- //
function DocumentTitle() {
  const [clicks, setClicks] = useState(0);

  useEffect(() => {
    const previous = document.title;
    document.title = `Clicked ${clicks} times`;
    return () => {
      document.title = previous; // cleanup: restore when the component unmounts
    };
  }, [clicks]); // re-run only when `clicks` changes. List EVERY value from the component used inside.

  return <button onClick={() => setClicks((c) => c + 1)}>Update tab title ({clicks})</button>;
}

// --------------------------------------------------------------------------- //
// 2. Subscriptions need cleanup, or you leak listeners on every mount.
//    (StrictMode mounts twice in development to catch exactly this.)
// --------------------------------------------------------------------------- //
function WindowSize() {
  const [size, setSize] = useState({ w: window.innerWidth, h: window.innerHeight });

  useEffect(() => {
    const onResize = () => setSize({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []); // [] = subscribe on mount, unsubscribe on unmount

  return <p>Window: {size.w} × {size.h} (resize the browser)</p>;
}

// --------------------------------------------------------------------------- //
// 3. A stopwatch: the interval id lives in a ref because it is not displayed,
//    and changing it must not trigger a render.
// --------------------------------------------------------------------------- //
function Stopwatch() {
  const [elapsedMs, setElapsedMs] = useState(0);
  const [running, setRunning] = useState(false);
  const intervalRef = useRef<number | null>(null);
  const startedAtRef = useRef(0);

  useEffect(() => {
    if (!running) return;
    startedAtRef.current = Date.now() - elapsedMs;
    intervalRef.current = window.setInterval(() => setElapsedMs(Date.now() - startedAtRef.current), 50);
    return () => {
      if (intervalRef.current !== null) window.clearInterval(intervalRef.current);
    };
    // elapsedMs is read only when starting; re-running on each tick would restart the interval.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  return (
    <div className="row">
      <output>{(elapsedMs / 1000).toFixed(2)}s</output>
      <button onClick={() => setRunning((r) => !r)}>{running ? "Stop" : "Start"}</button>
      <button className="secondary" onClick={() => { setRunning(false); setElapsedMs(0); }}>Reset</button>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// 4. DOM refs: focus an input and scroll an element into view.
//    In React 19, `ref` is a regular prop: function components can accept it
//    directly, without forwardRef.
// --------------------------------------------------------------------------- //
function TextInput({ ref, ...props }: ComponentProps<"input">) {
  return <input ref={ref} {...props} />;
}

function FocusDemo() {
  const inputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLParagraphElement>(null);

  return (
    <div className="stack">
      <div className="row">
        <TextInput ref={inputRef} placeholder="I can be focused from a button" />
        <button onClick={() => inputRef.current?.focus()}>Focus input</button>
        <button className="secondary" onClick={() => bottomRef.current?.scrollIntoView({ behavior: "smooth" })}>
          Scroll to bottom
        </button>
      </div>
      <div className="scroll-box">
        {Array.from({ length: 20 }, (_, i) => <p key={i}>Line {i + 1}</p>)}
        <p ref={bottomRef}><strong>The end.</strong></p>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// 5. You might NOT need an effect.
//    - Derived data: compute it during render (see 02), don't copy props into state with an effect.
//    - User events: do the work in the event handler, not in an effect that watches state.
//    - Resetting state when a prop changes: give the component a `key` instead.
//    - Fetching data: prefer a data library or router loaders (see 05 and 09).
// --------------------------------------------------------------------------- //
function Greeting({ name }: { name: string }) {
  // ❌ const [upper, setUpper] = useState(""); useEffect(() => setUpper(name.toUpperCase()), [name]);
  // ✅ an extra render is avoided and the value is never stale:
  const upper = name.toUpperCase();
  return <p>Hello, {upper}</p>;
}

export default function EffectsRefsExample() {
  const [showWatch, setShowWatch] = useState(true);
  return (
    <>
      <h1>03 · Effects & refs</h1>
      <Card title="Effect with dependencies"><DocumentTitle /></Card>
      <Card title="Subscription with cleanup"><WindowSize /></Card>
      <Card
        title="Ref for a timer id"
        footer={<button className="link" onClick={() => setShowWatch((s) => !s)}>{showWatch ? "Unmount" : "Mount"} stopwatch</button>}
      >
        {showWatch ? <Stopwatch /> : <p className="muted">Unmounted: the effect cleanup cleared the interval.</p>}
      </Card>
      <Card title="DOM refs"><FocusDemo /></Card>
      <Card title="No effect needed"><Greeting name="Ada" /></Card>
    </>
  );
}
