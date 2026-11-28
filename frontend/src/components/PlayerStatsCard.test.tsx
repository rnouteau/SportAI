import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PlayerStatsCard from "./PlayerStatsCard";

describe("PlayerStatsCard", () => {
  it("shows the name, the stats and the heatmap", () => {
    render(
      <PlayerStatsCard
        player={{
          id: 1,
          name: "player A",
          distance_m: 11.06,
          mean_speed_kmh: 4.05,
          max_speed_kmh: 7.57,
          heatmap_url: "/files/a/heatmaps/heatmap_player_1.png",
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: "player A" })).toBeInTheDocument();
    expect(screen.getByText("11.1 m")).toBeInTheDocument();
    expect(screen.getByText("4.0 km/h")).toBeInTheDocument();
    expect(screen.getByText("7.6 km/h")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "Court position heatmap of player A" }),
    ).toHaveAttribute("src", "/files/a/heatmaps/heatmap_player_1.png");
  });

  it("falls back on the id, dashes and a message when data is missing", () => {
    render(<PlayerStatsCard player={{ id: 3 }} />);

    expect(screen.getByRole("heading", { name: "Player 3" })).toBeInTheDocument();
    expect(screen.getAllByText("-")).toHaveLength(3);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("No heatmap for this player.")).toBeInTheDocument();
  });
});
