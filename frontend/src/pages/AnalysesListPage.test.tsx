import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ANALYSIS, mockFetch, renderWithProviders } from "../test/render";
import AnalysesListPage from "./AnalysesListPage";

describe("AnalysesListPage", () => {
  it("shows one card per analysis, linking to its page", async () => {
    const fetchMock = mockFetch([ANALYSIS]);
    renderWithProviders(<AnalysesListPage />);

    // findBy... waits for the (fake) API answer, getBy... would fail during the loading
    const card = await screen.findByRole("link", { name: /fonseca_vs_djokovic/ });
    expect(card).toHaveAttribute("href", "/analyses/fonseca");
    expect(screen.getByText("9.9 s")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/analyses");
  });

  it("shows a message when there is no analysis", async () => {
    mockFetch([]);
    renderWithProviders(<AnalysesListPage />);

    expect(await screen.findByText(/No analysis yet/)).toBeInTheDocument();
  });

  it("shows the error when the API fails", async () => {
    mockFetch({ detail: "boom" }, 500);
    renderWithProviders(<AnalysesListPage />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not load the analyses: 500",
    );
  });
});
