import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { fetchAnalyses } from "../api/client";

export default function AnalysesListPage() {
  const { data, isPending, isError, error } = useQuery({
    queryKey: ["analyses"],
    queryFn: fetchAnalyses,
  });

  if (isPending) {
    return <p>Loading...</p>;
  }

  if (isError) {
    return <p>Error: {error.message}</p>;
  }

  if (data.length === 0) {
    return <p>No analysis yet.</p>;
  }

  return (
    <ul>
      {data.map((analysis) => (
        <li key={analysis.id}>
          <Link to={`/analyses/${analysis.id}`}>
            {analysis.video_name}: {analysis.players.length} player(s),{" "}
            {analysis.duration_s.toFixed(1)} s
          </Link>
        </li>
      ))}
    </ul>
  );
}
