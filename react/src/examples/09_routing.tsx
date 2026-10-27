/**
 * 09 - ROUTING with React Router: multiple pages, URLs and per-route data.
 *
 * The URL is state too: it can be bookmarked, shared, and survives reloads.
 * These routes are mounted under /09 in App.tsx (the whole app uses the same router).
 *
 * What's covered:
 *   - nested routes with a layout and <Outlet />
 *   - <Link> / <NavLink> and programmatic navigation with useNavigate
 *   - URL params (/09/users/:userId) with useParams
 *   - search params (?q=...) with useSearchParams
 *   - loaders: fetch data BEFORE the page renders, read it with useLoaderData
 *   - pending UI with useNavigation, per-route errorElement, throwing 404 responses
 *
 * Loaders replace the useEffect fetching from 05: data starts loading as soon as
 * the user clicks, and the page never renders in a half-loaded state.
 */
import {
  Link,
  NavLink,
  Outlet,
  isRouteErrorResponse,
  useLoaderData,
  useNavigate,
  useNavigation,
  useParams,
  useRouteError,
  useSearchParams,
  type LoaderFunctionArgs,
  type RouteObject,
} from "react-router";

import { Card } from "./01_components_props";
import { getJson, type User } from "./05_data_fetching";

// --------------------------------------------------------------------------- //
// 1. Loaders run before their route renders. They receive the URL params and
//    the Request (with an AbortSignal that fires if the user navigates away).
// --------------------------------------------------------------------------- //
async function usersLoader({ request }: LoaderFunctionArgs) {
  return getJson<User[]>("/users", request.signal);
}

async function userLoader({ params, request }: LoaderFunctionArgs) {
  const id = Number(params.userId);
  if (!Number.isInteger(id)) throw new Response("Invalid user id", { status: 400 });
  try {
    return await getJson<User>(`/users/${id}`, request.signal);
  } catch {
    // Throwing a Response renders the nearest errorElement with that status.
    throw new Response("User not found", { status: 404 });
  }
}

// --------------------------------------------------------------------------- //
// 2. Layout route: shared UI around the child routes.
// --------------------------------------------------------------------------- //
function RoutingLayout() {
  const navigation = useNavigation(); // "idle" | "loading" | "submitting"
  return (
    <>
      <h1>09 · Routing</h1>
      <nav className="row tabs">
        {/* `end` makes /09 active only on exactly /09, not on /09/users */}
        <NavLink to="/09" end>Overview</NavLink>
        <NavLink to="/09/users">Users</NavLink>
        <NavLink to="/09/users/999">Missing user</NavLink>
      </nav>
      {navigation.state === "loading" && <p className="muted">Loading…</p>}
      <Outlet />
    </>
  );
}

function Overview() {
  const navigate = useNavigate();
  return (
    <Card title="Overview">
      <p>Every link here changes the URL without reloading the page. Try the browser's back button.</p>
      <div className="row">
        <Link to="users?q=le">Users matching “le”</Link>
        {/* Programmatic navigation: after a form submit, a login, a timer... */}
        <button onClick={() => navigate("/09/users/3")}>Open user #3</button>
      </div>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
// 3. Loader data + search params. Search params are the right place for filters,
//    sorting, pagination and tabs: shareable and kept on reload.
// --------------------------------------------------------------------------- //
function UsersPage() {
  const users = useLoaderData<typeof usersLoader>();       // typed from the loader's return type
  const [searchParams, setSearchParams] = useSearchParams();
  const q = searchParams.get("q") ?? "";
  const visible = users.filter((u) => u.name.toLowerCase().includes(q.toLowerCase()));

  return (
    <Card title={`Users (${visible.length})`}>
      <input
        value={q}
        // replace: don't add a history entry for every keystroke
        onChange={(e) => setSearchParams(e.target.value ? { q: e.target.value } : {}, { replace: true })}
        placeholder="Filter by name (stored in the URL)"
        aria-label="filter users"
      />
      <ul>
        {visible.map((u) => (
          <li key={u.id}>
            {/* Relative link: resolves to /09/users/<id> */}
            <Link to={String(u.id)}>{u.name}</Link>
          </li>
        ))}
      </ul>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
// 4. URL params
// --------------------------------------------------------------------------- //
function UserPage() {
  const user = useLoaderData<typeof userLoader>();
  const { userId } = useParams(); // always strings (or undefined)
  const navigate = useNavigate();
  return (
    <Card title={user.name} footer={`Route param userId = "${userId}"`}>
      <p>{user.email} · {user.company.name}</p>
      <button className="secondary" onClick={() => navigate(-1)}>← Back</button>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
// 5. Route-level error UI. Errors in a child route render here, inside the
//    layout, so the rest of the app keeps working.
// --------------------------------------------------------------------------- //
function RoutingError() {
  const error = useRouteError();
  const message = isRouteErrorResponse(error)
    ? `${error.status} · ${error.data}`
    : error instanceof Error ? error.message : "Unknown error";
  return (
    <Card title="Couldn't load this page">
      <p role="alert" className="error">{message}</p>
      <Link to="/09/users">Back to users</Link>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
// 6. The route table. Paths without a leading "/" are relative to the parent.
//    For big apps, `lazy: () => import("./page")` loads a route's code on demand.
// --------------------------------------------------------------------------- //
export const routingRoutes: RouteObject[] = [
  {
    element: <RoutingLayout />, // pathless layout route: wraps children, adds no URL segment
    children: [
      {
        errorElement: <RoutingError />,
        children: [
          { index: true, element: <Overview /> },
          { path: "users", loader: usersLoader, element: <UsersPage /> },
          { path: "users/:userId", loader: userLoader, element: <UserPage /> },
        ],
      },
    ],
  },
];
