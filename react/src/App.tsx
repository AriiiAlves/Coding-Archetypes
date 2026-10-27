/**
 * App shell: a sidebar listing every example, plus the route table.
 *
 * Each example is a self-contained page in src/examples/. To start your own
 * app, keep this file's structure and replace the examples with your pages.
 */
import { Link, NavLink, Outlet, useRouteError, type RouteObject } from "react-router";

import { APP_NAME } from "./config";
import ComponentsProps from "./examples/01_components_props";
import StateEvents from "./examples/02_state_events";
import EffectsRefs from "./examples/03_effects_refs";
import Forms from "./examples/04_forms";
import DataFetching from "./examples/05_data_fetching";
import ContextReducer from "./examples/06_context_reducer";
import CustomHooks from "./examples/07_custom_hooks";
import Performance from "./examples/08_performance";
import { routingRoutes } from "./examples/09_routing";

const examples = [
  { path: "01", title: "Components & props", element: <ComponentsProps /> },
  { path: "02", title: "State & events", element: <StateEvents /> },
  { path: "03", title: "Effects & refs", element: <EffectsRefs /> },
  { path: "04", title: "Forms & actions", element: <Forms /> },
  { path: "05", title: "Data fetching", element: <DataFetching /> },
  { path: "06", title: "Context & reducer", element: <ContextReducer /> },
  { path: "07", title: "Custom hooks", element: <CustomHooks /> },
  { path: "08", title: "Performance", element: <Performance /> },
  { path: "09", title: "Routing", children: routingRoutes },
] satisfies (RouteObject & { title: string })[];

function Layout() {
  return (
    <div className="layout">
      <nav className="sidebar">
        <Link to="/" className="brand">{APP_NAME}</Link>
        <ol>
          {examples.map((ex) => (
            <li key={ex.path}>
              {/* NavLink adds class="active" when its route matches */}
              <NavLink to={`/${ex.path}`}>
                {ex.path} · {ex.title}
              </NavLink>
            </li>
          ))}
        </ol>
      </nav>
      <main className="content">
        <Outlet /> {/* the matched child route renders here */}
      </main>
    </div>
  );
}

function Home() {
  return (
    <>
      <h1>{APP_NAME}</h1>
      <p>
        Each page is one example from <code>src/examples/</code>, ordered from basic to advanced.
        Open the file next to the page to read the commented code.
      </p>
      <p>
        Tests live in <code>src/examples/10_testing.test.tsx</code>. Run them with <code>npm test</code>.
      </p>
    </>
  );
}

// Rendered when a loader or component throws, or no route matches.
function RouteError() {
  const error = useRouteError();
  const message = error instanceof Error ? error.message : JSON.stringify(error);
  return (
    <div className="content">
      <h1>Something went wrong</h1>
      <pre>{message}</pre>
      <Link to="/">Back home</Link>
    </div>
  );
}

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <Layout />,
    errorElement: <RouteError />,
    // Shown while the first page's loaders run, when the app opens directly on a route with a loader.
    hydrateFallbackElement: <p className="content muted">Loading…</p>,
    children: [
      { index: true, element: <Home /> },
      ...examples.map(({ title: _title, ...route }) => route),
      { path: "*", element: <h1>Page not found</h1> },
    ],
  },
];
