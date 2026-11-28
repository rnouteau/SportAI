import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router";
import { fetchAnalysis } from "../api/client";
import OverlayPlayer from "../components/OverlayPlayer";
import PlayerStatsCard from "../components/PlayerStatsCard";
import StatItem from "../components/StatItem";
import StatusMessage from "../components/StatusMessage";
import { formatDuration, videoTitle } from "../utils/format";

type AnalysisDetailsProps = {
  id: string;
};

function AnalysisDetails({ id }: AnalysisDetailsProps) {
  const { data, isPending, isError, error } = useQuery({
    queryKey: ["analysis", id],
    queryFn: () => fetchAnalysis(id),
  });

  if (isPending) {
    return <StatusMessage>Loading the analysis...</StatusMessage>;
  }

  if (isError) {
    return (
      <StatusMessage tone="error">
        Could not load this analysis: {error.message}
      </StatusMessage>
    );
  }

  return (
    <article className="space-y-10">
      <header>
        <h1
          className="text-2xl font-bold tracking-tight break-words"
          title={data.video_name}
        >
          {videoTitle(data.video_name)}
        </h1>
        <dl className="mt-4 flex flex-wrap gap-x-10 gap-y-4">
          <StatItem label="Duration" value={formatDuration(data.duration_s)} />
          <StatItem label="Frames" value={String(data.frame_count)} />
          <StatItem label="Frame rate" value={`${data.fps.toFixed(0)} fps`} />
          <StatItem label="Players" value={String(data.players.length)} />
        </dl>
      </header>

      <section>
        <h2 className="mb-4 text-lg font-semibold">Overlay video</h2>
        {data.overlay_url ? (
          <OverlayPlayer url={data.overlay_url} />
        ) : (
          <StatusMessage>No overlay video for this analysis.</StatusMessage>
        )}
      </section>

      <section>
        <h2 className="mb-4 text-lg font-semibold">Players</h2>
        {data.players.length === 0 ? (
          <StatusMessage>No player stats for this analysis.</StatusMessage>
        ) : (
          <ul className="grid gap-6 md:grid-cols-2">
            {data.players.map((player) => (
              <li key={player.id}>
                <PlayerStatsCard player={player} />
              </li>
            ))}
          </ul>
        )}
      </section>
    </article>
  );
}

export default function AnalysisPage() {
  const { id } = useParams();

  return (
    <div>
      <Link
        to="/"
        className="text-sm font-medium text-slate-600 hover:text-blue-700"
      >
        ← All analyses
      </Link>
      <div className="mt-4">
        {id ? (
          <AnalysisDetails id={id} />
        ) : (
          <StatusMessage tone="error">Missing analysis id.</StatusMessage>
        )}
      </div>
    </div>
  );
}
