# React Archetype: Modern React with TypeScript

A starter kit for building web UIs with **React 19**, **TypeScript**, **Vite**, **React Router** and **Vitest**.
Each example is one commented page in a single app, ordered from basic to advanced.
Run the app, open a page, and read its file side by side. Copy the example closest to your goal.

## Setup

```bash
cd react
npm install
npm run dev        # http://localhost:5173, with hot reload
```

Requires Node 22.22 or newer. Settings (API URL, timeouts, storage prefix) live in **`src/config.ts`**. Per-environment values come from a `.env.local` file (see the comments in that file).

| Command | What it does |
|---------|--------------|
| `npm run dev` | Development server with hot module replacement |
| `npm test` | Vitest in watch mode (`npm test -- --run` runs once, for CI) |
| `npm run typecheck` | TypeScript check without emitting files |
| `npm run build` | Type-check, then production build into `dist/` |
| `npm run preview` | Serve the production build locally |

## The examples (in learning order)

| # | File (`src/examples/`) | What you learn | Key tools |
|---|------|----------------|-----------|
| 01 | `01_components_props.tsx` | Components, typed props and defaults, `children`, lists and keys, conditional rendering, callback props | JSX, `ReactNode` |
| 02 | `02_state_events.tsx` | State with primitives, objects and arrays, immutable updates, controlled inputs, derived values, lifting state up | `useState`, event handlers |
| 03 | `03_effects_refs.tsx` | Syncing with browser APIs, cleanup, timers, DOM refs, `ref` as a prop, when *not* to use an effect | `useEffect`, `useRef` |
| 04 | `04_forms.tsx` | Controlled forms with accessible validation, React 19 form actions, pending states, optimistic UI | `useActionState`, `useFormStatus`, `useOptimistic` |
| 05 | `05_data_fetching.tsx` | Typed fetch helper, abort and timeout, loading and error states, a `useFetch` hook, Suspense data loading, error boundaries | `fetch`, `AbortController`, `use`, `Suspense` |
| 06 | `06_context_reducer.tsx` | App-wide state (a shopping cart) without prop drilling: typed reducer, context provider and hook | `useReducer`, `createContext`, `use(Context)` |
| 07 | `07_custom_hooks.tsx` | Reusable hooks: `useToggle`, `useLocalStorage`, `useDebounce`, `useMediaQuery`, `usePrevious` | `useSyncExternalStore`, `useCallback` |
| 08 | `08_performance.tsx` | Skipping re-renders, caching calculations, keeping input responsive during heavy updates, code splitting | `memo`, `useMemo`, `useTransition`, `useDeferredValue`, `lazy` |
| 09 | `09_routing.tsx` | Nested routes, layouts, links, URL and search params, loaders, pending UI, route error pages | `react-router` (data mode) |
| 10 | `10_testing.test.tsx` | Testing components, forms, reducers, hooks, mocked fetch and routes the way users use them | Vitest, Testing Library, `userEvent` |

The app shell is `src/App.tsx` (sidebar and route table), and `src/main.tsx` mounts it.
To start a real app, keep this structure, replace the example pages with your own, and move reusable pieces into folders such as `src/components/`, `src/hooks/` and `src/api/`.

## Which tool should I use?

| Need | Use | Example |
|------|-----|---------|
| A value that changes and is displayed | `useState` | 02 |
| A value computed from props or state | A plain variable in render (no state, no effect) | 02, 03 |
| Several related updates with logic | `useReducer` | 06 |
| A value many distant components read | Context (or Zustand / Redux Toolkit when it changes often) | 06 |
| Sync with something outside React | `useEffect` with cleanup, or `useSyncExternalStore` | 03, 07 |
| A DOM node, or a value that must not trigger renders | `useRef` | 03 |
| Server data | Router loaders or TanStack Query, `useFetch` for small apps | 05, 09 |
| Form submission with pending and error states | Form actions + `useActionState` | 04 |
| Shared logic between components | A custom hook | 07 |
| Slow renders | Measure with the React DevTools Profiler, then use `memo` / `useMemo` / `useTransition` | 08 |

## Scaling up

- **Data fetching and caching**: [TanStack Query](https://tanstack.com/query) adds caching, background refetching, retries and mutations on top of the patterns in 05.
- **Forms**: [react-hook-form](https://react-hook-form.com) with [zod](https://zod.dev) schemas for large forms with complex validation.
- **Global state**: [Zustand](https://zustand.docs.pmnd.rs) or [Redux Toolkit](https://redux-toolkit.js.org) when context re-renders become a problem.
- **Automatic memoization**: enable the [React Compiler](https://react.dev/learn/react-compiler) and remove most manual `memo` / `useMemo` / `useCallback`.
- **Styling**: CSS Modules (built into Vite), Tailwind CSS, or a component library such as shadcn/ui or MUI.
- **Server rendering, SEO, file-based routes**: React Router's framework mode or Next.js, both built on the same concepts.
- **End-to-end tests**: Playwright drives a real browser through whole user flows. Mock APIs in unit tests with MSW.
- **Linting**: add ESLint with `eslint-plugin-react-hooks`. It catches missing effect dependencies and rule-of-hooks violations.

## Common pitfalls

- **Mutating state** (`items.push(x); setItems(items)`): React sees the same object and doesn't re-render. Always create a new array or object.
- **Stale values in handlers**: `setCount(count + 1)` twice adds 1. Use the function form `setCount(c => c + 1)`.
- **Missing effect dependencies**: the effect keeps using old values. List everything from the component that the effect reads.
- **Effects that run twice in development**: this is StrictMode testing your cleanup, not a bug. Production runs them once.
- **Index as `key`** in lists that can be reordered or filtered: rows swap their state. Use a stable id.
- **`{count && <X />}` renders `0`** when count is 0. Use `{count > 0 && <X />}`.
- **Fetching without aborting**: a slow old response can overwrite a newer one. Abort in the effect cleanup (05).
- **Creating a promise inside a component for `use()`**: it creates a new request on every render. Create it in an event handler, a loader or a cache.
- **Secrets in `VITE_*` variables**: they are bundled into public JavaScript.
- **Library versions**: this archetype was tested with React 19.3, React Router 8.5, Vite 8, Vitest 5 and TypeScript 7. Older tutorials use APIs that changed: `forwardRef` (now `ref` is a normal prop), `<Context.Provider>` (now `<Context>`), `useFormState` (now `useActionState`), `ReactDOM.render` (now `createRoot`), and `react-router-dom` (now `react-router`, with `RouterProvider` from `react-router/dom`).
