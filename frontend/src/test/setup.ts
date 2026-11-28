// Loaded by Vitest before every test file (see `test.setupFiles` in vite.config.ts)
import "@testing-library/jest-dom/vitest"; // adds toBeInTheDocument(), toHaveAttribute()...
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup(); // remove what the previous test rendered
  vi.unstubAllGlobals(); // undo the fake fetch of the page tests
});
