/**
 * 02 - STATE & EVENTS: making components interactive.
 *
 * State is data a component remembers between renders. Calling the setter
 * schedules a re-render with the new value. Never mutate state directly:
 * always pass a NEW object or array to the setter.
 *
 * What's covered:
 *   - useState with primitives, objects and arrays
 *   - event handlers (click, change, submit, keyboard)
 *   - controlled inputs
 *   - functional updates (setX(prev => ...))
 *   - immutable updates: add, remove, update items in an array
 *   - derived values (compute them, don't store them in state)
 *   - lifting state up to share it between siblings
 */
import { useState, type FormEvent } from "react";

import { Card } from "./01_components_props";

// --------------------------------------------------------------------------- //
// 1. A counter: the smallest stateful component.
// --------------------------------------------------------------------------- //
export function Counter({ initial = 0, step = 1 }: { initial?: number; step?: number }) {
  const [count, setCount] = useState(initial);

  // Functional update: use it when the new value depends on the old one.
  // `setCount(count + 1)` three times in one handler would only add 1, because
  // `count` is a snapshot of this render. The function form always gets the latest.
  const addThree = () => {
    setCount((c) => c + step);
    setCount((c) => c + step);
    setCount((c) => c + step);
  };

  return (
    <div className="row">
      <button onClick={() => setCount((c) => c - step)} aria-label="decrement">−</button>
      <output aria-label="count">{count}</output>
      <button onClick={() => setCount((c) => c + step)} aria-label="increment">+</button>
      <button className="secondary" onClick={addThree}>+3 steps</button>
      <button className="secondary" onClick={() => setCount(initial)}>Reset</button>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// 2. Arrays in state: a todo list with immutable updates.
// --------------------------------------------------------------------------- //
type Todo = { id: string; text: string; done: boolean };

export function TodoList() {
  const [todos, setTodos] = useState<Todo[]>([
    { id: crypto.randomUUID(), text: "Learn useState", done: true },
  ]);
  const [draft, setDraft] = useState(""); // controlled input: state is the single source of truth
  const [filter, setFilter] = useState<"all" | "open" | "done">("all");

  // Derived values: computed during render, never stored separately in state.
  // Storing them would risk getting out of sync with `todos`.
  const visible = todos.filter((t) => filter === "all" || (filter === "done") === t.done);
  const remaining = todos.filter((t) => !t.done).length;

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault(); // stop the browser from reloading the page
    const text = draft.trim();
    if (!text) return;
    setTodos((prev) => [...prev, { id: crypto.randomUUID(), text, done: false }]); // add
    setDraft("");
  }

  const toggle = (id: string) =>
    setTodos((prev) => prev.map((t) => (t.id === id ? { ...t, done: !t.done } : t))); // update one

  const remove = (id: string) => setTodos((prev) => prev.filter((t) => t.id !== id)); // remove

  return (
    <>
      <form onSubmit={handleSubmit} className="row">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Escape" && setDraft("")}
          placeholder="What needs doing?"
          aria-label="new todo"
        />
        <button type="submit">Add</button>
      </form>

      <div className="row">
        {(["all", "open", "done"] as const).map((f) => (
          <label key={f}>
            <input type="radio" name="filter" checked={filter === f} onChange={() => setFilter(f)} /> {f}
          </label>
        ))}
      </div>

      <ul className="plain">
        {visible.map((t) => (
          <li key={t.id} className="row">
            <label>
              <input type="checkbox" checked={t.done} onChange={() => toggle(t.id)} /> {t.text}
            </label>
            <button className="link" onClick={() => remove(t.id)} aria-label={`delete ${t.text}`}>✕</button>
          </li>
        ))}
      </ul>
      <p className="muted">{remaining} remaining</p>
    </>
  );
}

// --------------------------------------------------------------------------- //
// 3. Objects in state: copy with spread and change only what's needed.
// --------------------------------------------------------------------------- //
function ProfileEditor() {
  const [profile, setProfile] = useState({ name: "Ada", address: { city: "London" } });

  return (
    <div className="stack">
      <input
        aria-label="name"
        value={profile.name}
        onChange={(e) => setProfile({ ...profile, name: e.target.value })}
      />
      <input
        aria-label="city"
        value={profile.address.city}
        // Nested objects need a copy at every level that changes.
        onChange={(e) => setProfile({ ...profile, address: { ...profile.address, city: e.target.value } })}
      />
      <p className="muted">{profile.name} lives in {profile.address.city}.</p>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// 4. Lifting state up: two siblings need the same value, so the parent owns it
//    and passes the value and a setter down.
// --------------------------------------------------------------------------- //
function CelsiusInput({ value, onChange }: { value: number; onChange: (c: number) => void }) {
  return (
    <label>
      °C <input type="number" value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function FahrenheitInput({ celsius, onChange }: { celsius: number; onChange: (c: number) => void }) {
  const fahrenheit = Math.round((celsius * 9) / 5 + 32);
  return (
    <label>
      °F <input type="number" value={fahrenheit} onChange={(e) => onChange(((Number(e.target.value) - 32) * 5) / 9)} />
    </label>
  );
}

function TemperatureConverter() {
  const [celsius, setCelsius] = useState(20); // one source of truth for both inputs
  return (
    <div className="row">
      <CelsiusInput value={Math.round(celsius)} onChange={setCelsius} />
      <FahrenheitInput celsius={celsius} onChange={setCelsius} />
    </div>
  );
}

export default function StateEventsExample() {
  return (
    <>
      <h1>02 · State & events</h1>
      <Card title="Counter"><Counter /></Card>
      <Card title="Todo list (array state)"><TodoList /></Card>
      <Card title="Profile (object state)"><ProfileEditor /></Card>
      <Card title="Lifting state up"><TemperatureConverter /></Card>
    </>
  );
}
