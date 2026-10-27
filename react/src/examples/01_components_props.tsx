/**
 * 01 - COMPONENTS & PROPS: the building blocks of every React UI.
 *
 * A component is a function that takes props (an object) and returns JSX.
 * Data flows DOWN through props; components never modify their own props.
 *
 * What's covered:
 *   - typed props, optional props with defaults
 *   - children and composition (wrapper components)
 *   - rendering lists with keys
 *   - conditional rendering (&&, ternary, early return)
 *   - passing callbacks as props
 *   - Fragments, className, inline styles
 */
import type { ReactNode } from "react";

// --------------------------------------------------------------------------- //
// 1. Typed props. Describe props with a type; optional ones get defaults in
//    the parameter list.
// --------------------------------------------------------------------------- //
type BadgeProps = {
  label: string;
  tone?: "neutral" | "success" | "danger"; // union types = a fixed set of allowed values
};

function Badge({ label, tone = "neutral" }: BadgeProps) {
  return <span className={`badge badge-${tone}`}>{label}</span>;
}

// --------------------------------------------------------------------------- //
// 2. children: whatever is placed between the opening and closing tags.
//    Wrapper components like this are how you build layouts and design systems.
// --------------------------------------------------------------------------- //
type CardProps = {
  title: string;
  footer?: ReactNode; // ReactNode = anything renderable: JSX, string, number, null, arrays
  children: ReactNode;
};

export function Card({ title, footer, children }: CardProps) {
  return (
    <section className="card">
      <h3>{title}</h3>
      <div>{children}</div>
      {footer && <footer className="muted">{footer}</footer>}
    </section>
  );
}

// --------------------------------------------------------------------------- //
// 3. Lists and keys. `key` must be stable and unique among siblings, usually a
//    database id. Using the array index breaks state when items are reordered.
// --------------------------------------------------------------------------- //
type Task = { id: number; title: string; done: boolean; priority: "low" | "high" };

const TASKS: Task[] = [
  { id: 1, title: "Read the React docs", done: true, priority: "low" },
  { id: 2, title: "Build a component", done: false, priority: "high" },
  { id: 3, title: "Write a test", done: false, priority: "low" },
];

function TaskItem({ task }: { task: Task }) {
  return (
    <li style={{ textDecoration: task.done ? "line-through" : "none" }}>
      {task.title}{" "}
      {/* && renders the right side only when the left side is truthy.
          Careful: {count && <X/>} renders "0" when count is 0. Use {count > 0 && <X/>}. */}
      {task.priority === "high" && <Badge label="high priority" tone="danger" />}
    </li>
  );
}

function TaskList({ tasks }: { tasks: Task[] }) {
  // Early return for the empty state.
  if (tasks.length === 0) return <p className="muted">Nothing to do.</p>;

  return (
    <ul>
      {tasks.map((task) => (
        <TaskItem key={task.id} task={task} />
      ))}
    </ul>
  );
}

// --------------------------------------------------------------------------- //
// 4. Callbacks as props: the child tells the parent something happened.
//    Name them on<Event> in props and handle<Event> in the parent.
// --------------------------------------------------------------------------- //
type ButtonProps = {
  onClick: () => void;
  variant?: "primary" | "secondary";
  children: ReactNode;
};

function Button({ onClick, variant = "primary", children }: ButtonProps) {
  return (
    <button type="button" className={variant} onClick={onClick}>
      {children}
    </button>
  );
}

// --------------------------------------------------------------------------- //
// 5. Putting it together. The page component composes the smaller ones.
// --------------------------------------------------------------------------- //
export default function ComponentsPropsExample() {
  const pending = TASKS.filter((t) => !t.done);
  const user = { name: "Ada", isAdmin: true };

  function handleGreet() {
    alert(`Hello, ${user.name}!`);
  }

  return (
    // <>...</> is a Fragment: groups elements without adding a DOM node.
    <>
      <h1>01 · Components & props</h1>

      <Card title="Badges" footer="Props with default values">
        <Badge label="neutral" /> <Badge label="success" tone="success" />{" "}
        <Badge label="danger" tone="danger" />
      </Card>

      <Card title={`Tasks (${pending.length} pending)`}>
        <TaskList tasks={TASKS} />
        <TaskList tasks={[]} />
      </Card>

      <Card title="Conditional rendering">
        {/* Ternary for either/or. JS expressions go inside { }. */}
        <p>{user.isAdmin ? "You are an admin." : "You are a regular user."}</p>
        <Button onClick={handleGreet}>Greet</Button>{" "}
        <Button variant="secondary" onClick={() => console.log("clicked")}>
          Log to console
        </Button>
      </Card>
    </>
  );
}
