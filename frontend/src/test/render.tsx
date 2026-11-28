import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { vi } from "vitest";
import type { AnalysisSummary } from "../api/client";

/**
 * Render a component with the same providers as main.tsx: a fresh query cache for each
 * test, and an in-memory router starting at `route` instead of the browser address.
 */
export function renderWithProviders(ui: ReactNode, route = "/") {
  const queryClient = new QueryClient({
    // No retry: an error state must show at once, not after 3 attempts
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

/** Replace `fetch` with a fake one answering `body` with the HTTP `status`. */
export function mockFetch(body: unknown, status = 200) {
  const fetchMock = vi.fn(async () => Response.json(body, { status }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

export const ANALYSIS: AnalysisSummary = {
  id: "fonseca",
  video_name: "fonseca_vs_djokovic.mp4",
  fps: 50,
  frame_count: 493,
  duration_s: 9.86,
  overlay_url: "/files/fonseca/overlay.mp4",
  players: [
    {
      id: 1,
      name: "player A",
      distance_m: 11.06,
      mean_speed_kmh: 4.05,
      max_speed_kmh: 7.57,
      heatmap_url: "/files/fonseca/heatmaps/heatmap_player_1.png",
    },
    {
      id: 2,
      name: "player B",
      distance_m: 16.98,
      mean_speed_kmh: 6.21,
      max_speed_kmh: 14.78,
      heatmap_url: "/files/fonseca/heatmaps/heatmap_player_2.png",
    },
  ],
};
