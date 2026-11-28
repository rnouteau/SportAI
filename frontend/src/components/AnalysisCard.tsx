import { Link } from "react-router";
import type { AnalysisSummary } from "../api/client";
import { formatDuration, videoTitle } from "../utils/format";
import StatItem from "./StatItem";

type AnalysisCardProps = {
  analysis: AnalysisSummary;
};

/** Clickable card of the analyses list. */
export default function AnalysisCard({ analysis }: AnalysisCardProps) {
  return (
    <Link
      to={`/analyses/${analysis.id}`}
      className="group flex h-full flex-col rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:border-blue-300 hover:shadow-md focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
    >
      <h2
        className="line-clamp-2 font-semibold break-words group-hover:text-blue-700"
        title={analysis.video_name}
      >
        {videoTitle(analysis.video_name)}
      </h2>
      <dl className="mt-4 grid grid-cols-3 gap-2">
        <StatItem label="Duration" value={formatDuration(analysis.duration_s)} />
        <StatItem label="Players" value={String(analysis.players.length)} />
        <StatItem label="FPS" value={analysis.fps.toFixed(0)} />
      </dl>
      <span className="mt-auto pt-4 text-sm font-medium text-blue-600">
        View analysis →
      </span>
    </Link>
  );
}
