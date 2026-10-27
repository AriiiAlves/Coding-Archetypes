/**
 * 04 - FORMS & ACTIONS: collecting and validating user input.
 *
 * Two approaches, both common:
 *   A. Controlled form: every field lives in state. Best for live validation,
 *      dependent fields and complex UIs.
 *   B. Form actions (React 19): <form action={fn}> receives FormData, and
 *      useActionState tracks the result and pending status. Less code, works
 *      with uncontrolled inputs, and is the model used by server actions.
 *
 * What's covered:
 *   - controlled inputs of every type (text, select, checkbox, textarea)
 *   - validation on change/blur and on submit, accessible error messages
 *   - <form action>, useActionState, useFormStatus, FormData
 *   - useOptimistic: show the result before the server confirms
 *
 * For large forms, the react-hook-form + zod libraries reduce boilerplate.
 */
import { useActionState, useOptimistic, useState, type ChangeEvent, type FormEvent, type ReactElement } from "react";
import { useFormStatus } from "react-dom";

import { Card } from "./01_components_props";

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

// --------------------------------------------------------------------------- //
// A. Controlled form with validation
// --------------------------------------------------------------------------- //
type SignupValues = { email: string; password: string; plan: "free" | "pro"; terms: boolean; bio: string };
type Errors = Partial<Record<keyof SignupValues, string>>;

// A pure validation function: easy to unit test and reuse on the server.
export function validateSignup(v: SignupValues): Errors {
  const errors: Errors = {};
  if (!/^\S+@\S+\.\S+$/.test(v.email)) errors.email = "Enter a valid email.";
  if (v.password.length < 8) errors.password = "At least 8 characters.";
  if (!v.terms) errors.terms = "You must accept the terms.";
  return errors;
}

export function SignupForm({ onSubmit }: { onSubmit: (values: SignupValues) => void }) {
  const [values, setValues] = useState<SignupValues>({ email: "", password: "", plan: "free", terms: false, bio: "" });
  const [touched, setTouched] = useState<Partial<Record<keyof SignupValues, boolean>>>({});
  const errors = validateSignup(values); // derived on every render
  const showError = (field: keyof SignupValues) => (touched[field] ? errors[field] : undefined);

  // One handler for every field: uses the input's `name` attribute as the key.
  function handleChange(e: ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) {
    const { name, value, type } = e.target;
    const next = type === "checkbox" ? (e.target as HTMLInputElement).checked : value;
    setValues((prev) => ({ ...prev, [name]: next }));
  }
  const handleBlur = (e: { target: { name: string } }) => setTouched((t) => ({ ...t, [e.target.name]: true }));

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setTouched({ email: true, password: true, terms: true }); // reveal all errors
    if (Object.keys(errors).length === 0) onSubmit(values);
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="stack">
      <Field label="Email" error={showError("email")}>
        <input name="email" type="email" value={values.email} onChange={handleChange} onBlur={handleBlur} />
      </Field>
      <Field label="Password" error={showError("password")}>
        <input name="password" type="password" value={values.password} onChange={handleChange} onBlur={handleBlur} />
      </Field>
      <Field label="Plan">
        <select name="plan" value={values.plan} onChange={handleChange}>
          <option value="free">Free</option>
          <option value="pro">Pro</option>
        </select>
      </Field>
      <Field label="Bio">
        <textarea name="bio" rows={2} value={values.bio} onChange={handleChange} />
      </Field>
      <Field label="Accept terms" error={showError("terms")} inline>
        <input name="terms" type="checkbox" checked={values.terms} onChange={handleChange} onBlur={handleBlur} />
      </Field>
      <button type="submit">Sign up</button>
    </form>
  );
}

// Wraps an input with its label and error. Wrapping the input in <label>
// associates them, so clicking the label focuses the input and screen readers
// announce it. aria-invalid + role="alert" make the error accessible.
function Field({ label, error, inline, children }: { label: string; error?: string; inline?: boolean; children: ReactElement }) {
  return (
    <div className="field">
      <label className={inline ? "row" : "stack"}>
        <span>{label}</span>
        {children}
      </label>
      {error && <small role="alert" className="error">{error}</small>}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// B. Form actions: useActionState(action, initialState)
//    -> [state, formAction, isPending]. The action gets (previousState, formData).
// --------------------------------------------------------------------------- //
type NewsletterState = { status: "idle" | "ok" | "error"; message: string };

async function subscribeAction(_prev: NewsletterState, formData: FormData): Promise<NewsletterState> {
  const email = String(formData.get("email") ?? "");
  await wait(800); // stands in for `await fetch("/api/subscribe", { method: "POST", body: formData })`
  if (!email.includes("@")) return { status: "error", message: "That doesn't look like an email." };
  return { status: "ok", message: `Subscribed ${email}!` };
}

// useFormStatus reads the pending state of the PARENT <form>. It must be used in
// a component rendered inside the form, which makes reusable submit buttons easy.
function SubmitButton({ children }: { children: string }) {
  const { pending } = useFormStatus();
  return <button type="submit" disabled={pending}>{pending ? "Sending…" : children}</button>;
}

function NewsletterForm() {
  const [state, formAction, isPending] = useActionState<NewsletterState, FormData>(subscribeAction, { status: "idle", message: "" });
  return (
    // Uncontrolled input: no value/onChange, the browser holds the value until submit.
    // After a successful action, React resets uncontrolled fields automatically.
    <form action={formAction} className="stack">
      <div className="row">
        <input name="email" placeholder="you@example.com" aria-label="email" disabled={isPending} />
        <SubmitButton>Subscribe</SubmitButton>
      </div>
      {state.status !== "idle" && (
        <p role="status" className={state.status === "error" ? "error" : "success"}>{state.message}</p>
      )}
    </form>
  );
}

// --------------------------------------------------------------------------- //
// C. useOptimistic: show the new message immediately, roll back automatically
//    if the action fails.
// --------------------------------------------------------------------------- //
type Message = { id: number; text: string; sending?: boolean };

function Chat() {
  const [messages, setMessages] = useState<Message[]>([{ id: 1, text: "Hi there!" }]);
  const [optimistic, addOptimistic] = useOptimistic(messages, (current, text: string) => [
    ...current,
    { id: Date.now(), text, sending: true },
  ]);

  async function sendAction(formData: FormData) {
    const text = String(formData.get("text") ?? "").trim();
    if (!text) return;
    addOptimistic(text);                  // shown instantly with "sending"
    await wait(1000);                     // the real request
    setMessages((m) => [...m, { id: Date.now(), text }]); // confirmed state replaces the optimistic one
  }

  return (
    <form action={sendAction} className="stack">
      <ul className="plain">
        {optimistic.map((m) => (
          <li key={m.id} className={m.sending ? "muted" : undefined}>{m.text}{m.sending && " (sending…)"}</li>
        ))}
      </ul>
      <div className="row">
        <input name="text" placeholder="Message" aria-label="message" />
        <SubmitButton>Send</SubmitButton>
      </div>
    </form>
  );
}

export default function FormsExample() {
  const [submitted, setSubmitted] = useState<SignupValues | null>(null);
  return (
    <>
      <h1>04 · Forms & actions</h1>
      <Card title="Controlled form with validation" footer={submitted && <pre>{JSON.stringify(submitted, null, 2)}</pre>}>
        <SignupForm onSubmit={setSubmitted} />
      </Card>
      <Card title="Form action + useActionState"><NewsletterForm /></Card>
      <Card title="useOptimistic"><Chat /></Card>
    </>
  );
}
