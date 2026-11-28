import { screen } from "@testing-library/react";
import { Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import { ANALYSIS, mockFetch, renderWithProviders } from "../test/render";
import AnalysisPage from "./AnalysisPage";

// The page reads the id from the address: it needs its real route around it
function renderAt(route: string) {
  return renderWithProviders(
    <Routes>
      <Route path="/analyses/:id" element={<AnalysisPage />} />
    </Routes>,
    route,
  );
}

describe("AnalysisPage", () => {
  it("loads the analysis of the address and shows its content", async () => {
    const fetchMock = mockFetch(ANALYSIS);
    const { container } = renderAt("/analyses/fonseca");

    expect(
      await screen.findByRole("heading", { level: 1, name: "fonseca_vs_djokovic" }),
    ).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/analyses/fonseca");
    expect(container.querySelector("video")).toHaveAttribute(
      "src",
      "/files/fonseca/overlay.mp4",
    );
    expect(screen.getByRole("heading", { name: "player A" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "player B" })).toBeInTheDocument();
  });

  it("shows a message instead of the video when there is none", async () => {
    mockFetch({ ...ANALYSIS, overlay_url: null });
    const { container } = renderAt("/analyses/fonseca");

    expect(
      await screen.findByText("No overlay video for this analysis."),
    ).toBeInTheDocument();
    expect(container.querySelector("video")).toBeNull();
  });

  it("shows the API error for an unknown analysis", async () => {
    mockFetch({ detail: "Analysis not found" }, 404);
    renderAt("/analyses/unknown");

    expect(await screen.findByRole("alert")).toHaveTextContent("404");
  });
});
