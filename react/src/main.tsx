/**
 * Entry point: mounts React into <div id="root"> from index.html.
 *
 * createBrowserRouter + RouterProvider is React Router's "data mode": routes are
 * plain objects (see App.tsx) and can have loaders, actions and error boundaries.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter } from "react-router";
import { RouterProvider } from "react-router/dom";

import { routes } from "./App";
import "./styles.css";

const router = createBrowserRouter(routes);

createRoot(document.getElementById("root")!).render(
  // StrictMode (development only) renders components twice and runs effects
  // mount -> unmount -> mount, to surface impure renders and missing cleanups.
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
