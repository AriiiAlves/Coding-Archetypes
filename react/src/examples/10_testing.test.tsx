/**
 * 10 - TESTING with Vitest + React Testing Library.
 *
 * Test what the USER sees and does, not implementation details: find elements by
 * role, label or text (like a screen reader would), interact with userEvent, and
 * assert on the visible result. Tests written this way survive refactors.
 *
 * What's covered:
 *   - rendering a component and querying it (getBy / queryBy / findBy)
 *   - simulating typing and clicking with userEvent
 *   - testing a pure reducer and a validation function (fastest, no DOM)
 *   - testing a custom hook with renderHook and fake timers
 *   - testing a context consumer by wrapping it in its provider
 *   - mocking fetch for components that load data
 *   - testing routes with createMemoryRouter
 *
 * Run:  npm test             (watch mode: re-runs on save)
 *       npm test -- --run    (once, for CI)
 */
import { act, render, renderHook, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { RouterProvider } from "react-router/dom";

import { Counter, TodoList } from "./02_state_events";
import { SignupForm, validateSignup } from "./04_forms";
import { UserList } from "./05_data_fetching";
import { CartProvider, cartReducer, cartTotal, useCart } from "./06_context_reducer";
import { useDebounce, useToggle } from "./07_custom_hooks";
import { routingRoutes } from "./09_routing";

// --------------------------------------------------------------------------- //
// 1. Components: render, interact, assert
// --------------------------------------------------------------------------- //
describe("Counter", () => {
  it("increments, decrements and resets", async () => {
    const user = userEvent.setup();
    render(<Counter initial={5} />);

    const count = screen.getByLabelText("count"); // getBy* throws if not found: a built-in assertion
    expect(count).toHaveTextContent("5");

    await user.click(screen.getByRole("button", { name: "increment" }));
    await user.click(screen.getByRole("button", { name: "increment" }));
    expect(count).toHaveTextContent("7");

    await user.click(screen.getByRole("button", { name: "decrement" }));
    expect(count).toHaveTextContent("6");

    await user.click(screen.getByRole("button", { name: "Reset" }));
    expect(count).toHaveTextContent("5");
  });
});

describe("TodoList", () => {
  it("adds, completes and deletes todos", async () => {
    const user = userEvent.setup();
    render(<TodoList />);

    await user.type(screen.getByLabelText("new todo"), "Write tests{Enter}");
    expect(screen.getByText("Write tests")).toBeInTheDocument();
    expect(screen.getByText("1 remaining")).toBeInTheDocument();

    await user.click(screen.getByRole("checkbox", { name: /write tests/i }));
    expect(screen.getByText("0 remaining")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "delete Write tests" }));
    // queryBy* returns null instead of throwing: use it to assert absence.
    expect(screen.queryByText("Write tests")).not.toBeInTheDocument();
  });
});

describe("SignupForm", () => {
  it("shows errors and does not submit invalid data", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn(); // a mock function records its calls
    render(<SignupForm onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "Sign up" }));
    expect(screen.getAllByRole("alert")).toHaveLength(3);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits valid data", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SignupForm onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText("Email"), "ada@example.com");
    await user.type(screen.getByLabelText("Password"), "correct-horse");
    await user.selectOptions(screen.getByLabelText("Plan"), "pro");
    await user.click(screen.getByLabelText("Accept terms"));
    await user.click(screen.getByRole("button", { name: "Sign up" }));

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ email: "ada@example.com", plan: "pro", terms: true }),
    );
  });
});

// --------------------------------------------------------------------------- //
// 2. Pure functions: no rendering needed. Put as much logic here as you can.
// --------------------------------------------------------------------------- //
describe("cartReducer", () => {
  const keyboard = { id: "kb", name: "Keyboard", price: 80 };

  it("adds a product, then increments its quantity", () => {
    let state = cartReducer({ items: [] }, { type: "add", product: keyboard });
    state = cartReducer(state, { type: "add", product: keyboard });
    expect(state.items).toEqual([{ ...keyboard, quantity: 2 }]);
    expect(cartTotal(state)).toBe(160);
  });

  it("removes an item when its quantity drops to 0", () => {
    const state = cartReducer({ items: [{ ...keyboard, quantity: 1 }] }, { type: "setQuantity", id: "kb", quantity: 0 });
    expect(state.items).toHaveLength(0);
  });
});

it("validateSignup accepts valid input", () => {
  expect(validateSignup({ email: "a@b.co", password: "12345678", plan: "free", terms: true, bio: "" })).toEqual({});
});

// --------------------------------------------------------------------------- //
// 3. Hooks: renderHook runs a hook inside a throwaway test component.
// --------------------------------------------------------------------------- //
describe("custom hooks", () => {
  it("useToggle flips its value", () => {
    const { result } = renderHook(() => useToggle());
    expect(result.current[0]).toBe(false);
    act(() => result.current[1]()); // state updates outside events must be wrapped in act()
    expect(result.current[0]).toBe(true);
  });

  it("useDebounce only updates after the delay", () => {
    vi.useFakeTimers(); // control setTimeout manually instead of really waiting
    const { result, rerender } = renderHook(({ value }) => useDebounce(value, 300), {
      initialProps: { value: "a" },
    });
    rerender({ value: "ab" });
    expect(result.current).toBe("a");
    act(() => vi.advanceTimersByTime(300));
    expect(result.current).toBe("ab");
    vi.useRealTimers();
  });

  it("useCart works inside its provider and fails outside it", () => {
    const { result } = renderHook(() => useCart(), { wrapper: CartProvider });
    act(() => result.current.add({ id: "x", name: "X", price: 10 }));
    expect(result.current.count).toBe(1);

    vi.spyOn(console, "error").mockImplementation(() => {}); // silence React's error log
    expect(() => renderHook(() => useCart())).toThrow("inside <CartProvider>");
  });
});

// --------------------------------------------------------------------------- //
// 4. Network: mock fetch so tests are fast, deterministic and offline.
//    For larger apps, MSW (Mock Service Worker) mocks at the network level.
// --------------------------------------------------------------------------- //
describe("UserList", () => {
  afterEach(() => vi.restoreAllMocks());

  it("shows users after loading", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json([{ id: 1, name: "Ada Lovelace", email: "ada@x.io", company: { name: "Engines" } }]),
    );
    render(<UserList />);
    expect(screen.getByText(/loading users/i)).toBeInTheDocument();
    // findBy* waits (polls) until the element appears: use it for async UI.
    expect(await screen.findByText("Ada Lovelace")).toBeInTheDocument();
  });

  it("shows an error when the request fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("nope", { status: 500 }));
    render(<UserList />);
    expect(await screen.findByRole("alert")).toHaveTextContent("HTTP 500");
  });
});

// --------------------------------------------------------------------------- //
// 5. Routes: createMemoryRouter keeps the URL in memory, starting where you choose.
// --------------------------------------------------------------------------- //
describe("routing", () => {
  afterEach(() => vi.restoreAllMocks());

  it("loads and shows a user page from its URL", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({ id: 3, name: "Grace Hopper", email: "grace@x.io", company: { name: "Navy" } }),
    );
    const router = createMemoryRouter([{ path: "/09", children: routingRoutes }], {
      initialEntries: ["/09/users/3"],
    });
    render(<RouterProvider router={router} />);

    const heading = await screen.findByRole("heading", { name: "Grace Hopper" });
    const card = heading.closest("section")!;
    expect(within(card).getByText(/userId = "3"/)).toBeInTheDocument();
  });
});
