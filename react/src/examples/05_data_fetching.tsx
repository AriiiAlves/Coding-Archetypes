/**
 * 05 - DATA FETCHING: loading data from an API with loading and error states.
 *
 * Three patterns, from manual to declarative:
 *   A. fetch in useEffect, with AbortController and explicit loading/error state
 *   B. the same logic extracted into a reusable useFetch hook
 *   C. React 19 `use(promise)` + <Suspense> for loading + an Error Boundary for errors
 *
 * In real apps, a data library (TanStack Query, SWR) or router loaders (see 09)
 * add caching, deduplication, retries and refetching. The patterns below show
 * what they do for you, and are fine for small apps.
 */
import { Component, Suspense, use, useEffect, useState, type ReactNode } from "react";

import { API_BASE_URL, REQUEST_TIMEOUT_MS } from "../config";
import { Card } from "./01_components_props";

export type User = { id: number; name: string; email: string; company: { name: string } };
type Post = { id: number; title: string; body: string };

// --------------------------------------------------------------------------- //
// A typed fetch helper: one place for the base URL, timeout and error handling.
// fetch() only rejects on network failure, so check `res.ok` for HTTP errors.
// --------------------------------------------------------------------------- //
export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  const res = await fetch(`${API_BASE_URL}${path}`, {
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
  return res.json() as Promise<T>;
}

// --------------------------------------------------------------------------- //
// A. Manual fetch in an effect
// --------------------------------------------------------------------------- //
export function UserList() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Abort the request if the component unmounts (or the effect re-runs)
    // before it finishes: avoids setting state on an unmounted component and
    // race conditions where an old response overwrites a newer one.
    const controller = new AbortController();
    getJson<User[]>("/users", controller.signal)
      .then(setUsers)
      .catch((e: unknown) => {
        if (controller.signal.aborted) return; // expected on cleanup, not an error
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  if (loading) return <p className="muted">Loading users…</p>;
  if (error) return <p role="alert" className="error">Error: {error}</p>;
  return (
    <ul>
      {users.slice(0, 5).map((u) => (
        <li key={u.id}>{u.name} <span className="muted">· {u.company.name}</span></li>
      ))}
    </ul>
  );
}

// --------------------------------------------------------------------------- //
// B. A reusable hook. Re-fetches whenever `path` changes.
// --------------------------------------------------------------------------- //
type FetchState<T> = { data: T | null; loading: boolean; error: string | null };

export function useFetch<T>(path: string | null): FetchState<T> & { reload: () => void } {
  const [state, setState] = useState<FetchState<T>>({ data: null, loading: !!path, error: null });
  const [version, setVersion] = useState(0); // bump to force a reload

  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));
    getJson<T>(path, controller.signal)
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        setState({ data: null, loading: false, error: e instanceof Error ? e.message : String(e) });
      });
    return () => controller.abort();
  }, [path, version]);

  return { ...state, reload: () => setVersion((v) => v + 1) };
}

function PostsByUser() {
  const [userId, setUserId] = useState(1);
  const { data: posts, loading, error, reload } = useFetch<Post[]>(`/posts?userId=${userId}&_limit=3`);

  return (
    <div className="stack">
      <div className="row">
        <label>
          User{" "}
          <select value={userId} onChange={(e) => setUserId(Number(e.target.value))}>
            {[1, 2, 3, 4, 5].map((id) => <option key={id} value={id}>#{id}</option>)}
          </select>
        </label>
        <button className="secondary" onClick={reload}>Reload</button>
        {loading && <span className="muted">loading…</span>}
      </div>
      {error && <p role="alert" className="error">{error}</p>}
      {/* Keep showing the previous data while the next request loads: less flicker. */}
      <ol style={{ opacity: loading ? 0.5 : 1 }}>
        {posts?.map((p) => <li key={p.id}>{p.title}</li>)}
      </ol>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// C. use() + Suspense + Error Boundary.
//    use(promise) suspends the component until the promise resolves; the nearest
//    <Suspense> shows its fallback meanwhile. If it rejects, the nearest Error
//    Boundary catches it. The promise must be created OUTSIDE the component (or
//    cached), otherwise every render creates a new request.
// --------------------------------------------------------------------------- //
function UserCard({ userPromise }: { userPromise: Promise<User> }) {
  const user = use(userPromise);
  return <p><strong>{user.name}</strong> · {user.email}</p>;
}

// Error boundaries are still class components (or use the react-error-boundary library).
class ErrorBoundary extends Component<{ fallback: (error: Error, reset: () => void) => ReactNode; children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error }; // switch to the fallback UI on the next render
  }

  componentDidCatch(error: Error) {
    console.error("Caught by ErrorBoundary:", error); // send to Sentry or similar here
  }

  render() {
    if (this.state.error) return this.props.fallback(this.state.error, () => this.setState({ error: null }));
    return this.props.children;
  }
}

function SuspenseDemo() {
  const [userPromise, setUserPromise] = useState(() => getJson<User>("/users/1"));
  const load = (id: number) => setUserPromise(getJson<User>(`/users/${id}`));

  return (
    <div className="stack">
      <div className="row">
        <button onClick={() => load(Math.ceil(Math.random() * 10))}>Load random user</button>
        <button className="secondary" onClick={() => load(9999)}>Load missing user (404)</button>
      </div>
      <ErrorBoundary
        fallback={(error, reset) => (
          <p role="alert" className="error">
            {error.message} <button className="link" onClick={() => { load(1); reset(); }}>Retry</button>
          </p>
        )}
      >
        <Suspense fallback={<p className="muted">Loading user…</p>}>
          <UserCard userPromise={userPromise} />
        </Suspense>
      </ErrorBoundary>
    </div>
  );
}

export default function DataFetchingExample() {
  return (
    <>
      <h1>05 · Data fetching</h1>
      <p className="muted">Data from <code>{API_BASE_URL}</code> (set in <code>src/config.ts</code>).</p>
      <Card title="A. fetch in useEffect"><UserList /></Card>
      <Card title="B. useFetch hook with parameters"><PostsByUser /></Card>
      <Card title="C. use() + Suspense + Error Boundary"><SuspenseDemo /></Card>
    </>
  );
}
