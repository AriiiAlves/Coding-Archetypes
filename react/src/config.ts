/**
 * Central configuration shared by every example in this archetype.
 *
 * Change settings here once instead of hunting through each file.
 * Values that differ per environment (dev / staging / prod) come from
 * Vite env variables: create a `.env.local` file with lines like
 *   VITE_API_BASE_URL=https://api.example.com
 * Only variables prefixed with VITE_ are exposed to browser code, and
 * they are PUBLIC: anyone can read them in the bundle. Never put secrets here.
 */

export const APP_NAME = "React Archetype";

// REST API used by the data-fetching examples. The default is a free fake API.
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "https://jsonplaceholder.typicode.com";

// Abort requests that take longer than this.
export const REQUEST_TIMEOUT_MS = 8000;

// Delay before a search box triggers filtering or a request (07, 08).
export const SEARCH_DEBOUNCE_MS = 300;

// Prefix for localStorage keys, so several apps on one domain don't collide.
export const STORAGE_PREFIX = "react-archetype:";

// import.meta.env.DEV is true under `npm run dev`, false in the production build.
export const IS_DEV = import.meta.env.DEV;
