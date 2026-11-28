import type { PlayerSummary } from "../api/client";
import { formatValue } from "../utils/format";
import StatItem from "./StatItem";

type PlayerStatsCardProps = {
  player: PlayerSummary;
};

export default function PlayerStatsCard({ player }: PlayerStatsCardProps) {
  const name = player.name ?? `Player ${player.id}`;

  return (
    <article className="flex h-full flex-col rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="font-semibold capitalize">{name}</h3>
      <dl className="mt-4 grid grid-cols-3 gap-4">
        <StatItem
          label="Distance"
          value={formatValue(player.distance_m, "m")}
          size="lg"
        />
        <StatItem
          label="Mean speed"
          value={formatValue(player.mean_speed_kmh, "km/h")}
          size="lg"
        />
        <StatItem
          label="Max speed"
          value={formatValue(player.max_speed_kmh, "km/h")}
          size="lg"
        />
      </dl>
      {player.heatmap_url ? (
        <figure className="mt-6">
          <img
            src={player.heatmap_url}
            alt={`Court position heatmap of ${name}`}
            loading="lazy"
            className="mx-auto max-h-96 w-auto rounded-lg"
          />
          <figcaption className="mt-2 text-center text-xs text-slate-500">
            Court positions over the clip
          </figcaption>
        </figure>
      ) : (
        <p className="mt-6 text-sm text-slate-500">No heatmap for this player.</p>
      )}
    </article>
  );
}
