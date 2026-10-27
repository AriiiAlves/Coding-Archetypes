/**
 * 06 - CONTEXT & REDUCER: app-wide state without passing props through every level.
 *
 * useReducer moves state-update logic into one pure function:
 *   (state, action) => newState
 * Context makes a value available to every component below a provider.
 * Together they are a small, dependency-free "store" for things like a cart,
 * the current user, a theme or feature flags.
 *
 * What's covered:
 *   - a typed reducer with a discriminated union of actions
 *   - a context + provider + custom hook (the standard trio)
 *   - consuming the store from deeply nested components
 *   - a second, simple context (theme)
 *
 * When state is updated very often or by many components, a store library
 * (Zustand, Redux Toolkit, Jotai) avoids re-rendering every consumer.
 */
import { createContext, use, useMemo, useReducer, useState, type ReactNode } from "react";

import { Card } from "./01_components_props";

// --------------------------------------------------------------------------- //
// 1. State, actions and the reducer. Exported so tests can call it directly:
//    a pure function is the easiest thing in the app to test.
// --------------------------------------------------------------------------- //
export type Product = { id: string; name: string; price: number };
export type CartItem = Product & { quantity: number };
export type CartState = { items: CartItem[] };

// Discriminated union: TypeScript knows which fields exist for each `type`.
export type CartAction =
  | { type: "add"; product: Product }
  | { type: "remove"; id: string }
  | { type: "setQuantity"; id: string; quantity: number }
  | { type: "clear" };

export function cartReducer(state: CartState, action: CartAction): CartState {
  switch (action.type) {
    case "add": {
      const existing = state.items.find((i) => i.id === action.product.id);
      if (existing) {
        return {
          items: state.items.map((i) => (i.id === action.product.id ? { ...i, quantity: i.quantity + 1 } : i)),
        };
      }
      return { items: [...state.items, { ...action.product, quantity: 1 }] };
    }
    case "remove":
      return { items: state.items.filter((i) => i.id !== action.id) };
    case "setQuantity":
      if (action.quantity <= 0) return cartReducer(state, { type: "remove", id: action.id });
      return { items: state.items.map((i) => (i.id === action.id ? { ...i, quantity: action.quantity } : i)) };
    case "clear":
      return { items: [] };
    default: {
      // Compile-time check that every action type is handled.
      const unhandled: never = action;
      return unhandled;
    }
  }
}

export const cartTotal = (state: CartState) => state.items.reduce((sum, i) => sum + i.price * i.quantity, 0);

// --------------------------------------------------------------------------- //
// 2. Context + Provider + hook
// --------------------------------------------------------------------------- //
type CartContextValue = {
  state: CartState;
  total: number;
  count: number;
  add: (p: Product) => void;
  remove: (id: string) => void;
  setQuantity: (id: string, quantity: number) => void;
  clear: () => void;
};

// null default + a guard in the hook: using the hook outside the provider fails loudly.
const CartContext = createContext<CartContextValue | null>(null);

export function CartProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(cartReducer, { items: [] });

  // Memoize the value so consumers only re-render when the cart actually changes.
  // (dispatch is stable, so the helper functions never change.)
  const value = useMemo<CartContextValue>(
    () => ({
      state,
      total: cartTotal(state),
      count: state.items.reduce((n, i) => n + i.quantity, 0),
      add: (product) => dispatch({ type: "add", product }),
      remove: (id) => dispatch({ type: "remove", id }),
      setQuantity: (id, quantity) => dispatch({ type: "setQuantity", id, quantity }),
      clear: () => dispatch({ type: "clear" }),
    }),
    [state],
  );

  // React 19: render the context itself as the provider (<Ctx.Provider> still works).
  return <CartContext value={value}>{children}</CartContext>;
}

export function useCart() {
  // use(Context) is the React 19 form of useContext(Context); it can also be called conditionally.
  const ctx = use(CartContext);
  if (!ctx) throw new Error("useCart must be used inside <CartProvider>");
  return ctx;
}

// --------------------------------------------------------------------------- //
// 3. Consumers anywhere in the tree. No props passed between them.
// --------------------------------------------------------------------------- //
const PRODUCTS: Product[] = [
  { id: "kb", name: "Keyboard", price: 79 },
  { id: "ms", name: "Mouse", price: 29 },
  { id: "mn", name: "Monitor", price: 249 },
];

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

function ProductCatalog() {
  const { add } = useCart();
  return (
    <ul className="plain">
      {PRODUCTS.map((p) => (
        <li key={p.id} className="row">
          <span>{p.name} · {money.format(p.price)}</span>
          <button onClick={() => add(p)}>Add to cart</button>
        </li>
      ))}
    </ul>
  );
}

function CartBadge() {
  const { count } = useCart();
  return <span className="badge badge-success">🛒 {count}</span>;
}

function CartSummary() {
  const { state, total, setQuantity, remove, clear } = useCart();
  if (state.items.length === 0) return <p className="muted">Your cart is empty.</p>;
  return (
    <div className="stack">
      <ul className="plain">
        {state.items.map((i) => (
          <li key={i.id} className="row">
            <span>{i.name}</span>
            <input
              type="number"
              min={0}
              value={i.quantity}
              onChange={(e) => setQuantity(i.id, Number(e.target.value))}
              aria-label={`${i.name} quantity`}
              style={{ width: "4rem" }}
            />
            <span>{money.format(i.price * i.quantity)}</span>
            <button className="link" onClick={() => remove(i.id)}>remove</button>
          </li>
        ))}
      </ul>
      <p><strong>Total: {money.format(total)}</strong></p>
      <button className="secondary" onClick={clear}>Clear cart</button>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// 4. A simple context for a value that rarely changes (theme, locale, user).
// --------------------------------------------------------------------------- //
type Theme = "light" | "dark";
const ThemeContext = createContext<Theme>("light"); // a sensible default instead of null

function ThemedBox({ children }: { children: ReactNode }) {
  const theme = use(ThemeContext);
  return <div className={`themed themed-${theme}`}>{children}</div>;
}

export default function ContextReducerExample() {
  const [theme, setTheme] = useState<Theme>("light");
  return (
    <CartProvider>
      <h1>06 · Context & reducer <CartBadge /></h1>
      <Card title="Catalog"><ProductCatalog /></Card>
      <Card title="Cart"><CartSummary /></Card>
      <Card title="Theme context">
        <ThemeContext value={theme}>
          <button onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}>Toggle theme</button>
          <ThemedBox>This box reads the theme from context: {theme}</ThemedBox>
        </ThemeContext>
      </Card>
    </CartProvider>
  );
}
