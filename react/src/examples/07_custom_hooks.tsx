/**
 * 07 - CUSTOM HOOKS: reuse stateful logic across components.
 *
 * A custom hook is a function whose name starts with `use` and that calls other
 * hooks. It shares LOGIC, not state: every component that calls it gets its own
 * independent copy of the state.
 *
 * Rules of hooks: call them at the top level of a component or another hook,
 * never inside conditions, loops or nested functions (React relies on call order).
 *
 * What's covered (copy any of these into src/hooks/ in your app):
 *   - useToggle          small state helpers
 *   - useLocalStorage    state persisted across reloads
 *   - useDebounce        delay a fast-changing value (search boxes)
 *   - useMediaQuery      subscribe to an external store with useSyncExternalStore
 *   - usePrevious        remember the value from the last render
 */
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

import { SEARCH_DEBOUNCE_MS, STORAGE_PREFIX } from "../config";
import { Card } from "./01_components_props";

// --------------------------------------------------------------------------- //
// useToggle: wrap a common state pattern behind a clear name.
// --------------------------------------------------------------------------- //
export function useToggle(initial = false) {
  const [on, setOn] = useState(initial);
  // useCallback keeps the function identity stable, so it is safe in deps arrays
  // and as a prop to memoized children (see 08).
  const toggle = useCallback(() => setOn((v) => !v), []);
  return [on, toggle, setOn] as const; // `as const` = a typed tuple, like useState
}

// --------------------------------------------------------------------------- //
// useLocalStorage: same API as useState, but survives page reloads.
// --------------------------------------------------------------------------- //
export function useLocalStorage<T>(key: string, initialValue: T) {
  const storageKey = STORAGE_PREFIX + key;

  // Lazy initializer: the function runs only on the first render, not on every render.
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = window.localStorage.getItem(storageKey);
      return stored !== null ? (JSON.parse(stored) as T) : initialValue;
    } catch {
      return initialValue; // storage blocked (private mode) or corrupted JSON
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(value));
    } catch {
      /* quota exceeded or storage blocked: keep working in memory */
    }
  }, [storageKey, value]);

  return [value, setValue] as const;
}

// --------------------------------------------------------------------------- //
// useDebounce: returns `value` only after it stopped changing for `delayMs`.
// --------------------------------------------------------------------------- //
export function useDebounce<T>(value: T, delayMs = SEARCH_DEBOUNCE_MS): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id); // each new value cancels the previous timer
  }, [value, delayMs]);
  return debounced;
}

// --------------------------------------------------------------------------- //
// useMediaQuery: useSyncExternalStore is the correct way to read a value that
// lives outside React (browser APIs, store libraries). It handles concurrent
// rendering correctly, which a useEffect + useState version does not.
// --------------------------------------------------------------------------- //
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const mql = window.matchMedia(query);
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    },
    [query],
  );
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches, // read the current value (client)
    () => false,                             // value during server rendering
  );
}

// --------------------------------------------------------------------------- //
// usePrevious: the value `value` had before its most recent change.
// --------------------------------------------------------------------------- //
export function usePrevious<T>(value: T): T | undefined {
  const [current, setCurrent] = useState(value);
  const [previous, setPrevious] = useState<T | undefined>(undefined);
  // Adjusting state during render (instead of in an effect) is the recommended
  // pattern for "react to a prop change": React re-renders immediately.
  if (!Object.is(value, current)) {
    setPrevious(current);
    setCurrent(value);
  }
  return previous;
}

// --------------------------------------------------------------------------- //
// Demo: the hooks used together
// --------------------------------------------------------------------------- //
const FRUITS = ["apple", "apricot", "banana", "blueberry", "cherry", "grape", "kiwi", "lemon", "mango", "melon", "orange", "peach", "pear", "plum"];

function SearchDemo() {
  const [query, setQuery] = useLocalStorage("fruit-query", "");
  const debouncedQuery = useDebounce(query);
  const renders = useRef(0);
  renders.current += 1; // reading/writing ref.current during render is fine only for debug counters like this

  // The expensive work (here: filtering, in real apps: a network request) only
  // runs when the debounced value changes, not on every keystroke.
  const results = FRUITS.filter((f) => f.includes(debouncedQuery.toLowerCase()));

  return (
    <div className="stack">
      <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search fruit (saved in localStorage)" aria-label="search" />
      <p className="muted">
        typed: “{query}” · searched: “{debouncedQuery}” {query !== debouncedQuery && "(waiting…)"}
      </p>
      <p>{results.join(", ") || "No matches"}</p>
    </div>
  );
}

function HooksDemo() {
  const [open, toggleOpen] = useToggle();
  const [count, setCount] = useState(0);
  const previous = usePrevious(count);
  const isWide = useMediaQuery("(min-width: 900px)");
  const prefersDark = useMediaQuery("(prefers-color-scheme: dark)");

  return (
    <div className="stack">
      <div className="row">
        <button onClick={toggleOpen}>{open ? "Hide" : "Show"} details</button>
        {open && <span>Details are visible.</span>}
      </div>
      <div className="row">
        <button onClick={() => setCount((c) => c + 1)}>count: {count}</button>
        <span className="muted">previous: {previous ?? "—"}</span>
      </div>
      <p>Viewport is {isWide ? "wide (≥ 900px)" : "narrow"} · OS theme: {prefersDark ? "dark" : "light"}</p>
    </div>
  );
}

export default function CustomHooksExample() {
  return (
    <>
      <h1>07 · Custom hooks</h1>
      <Card title="useLocalStorage + useDebounce"><SearchDemo /></Card>
      <Card title="useToggle, usePrevious, useMediaQuery"><HooksDemo /></Card>
    </>
  );
}
