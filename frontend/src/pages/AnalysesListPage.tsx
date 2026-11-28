import { useQuery } from "@tanstack/react-query";
import { fetchAnalyses } from "../api/client";
import AnalysisCard from "../components/AnalysisCard";
import StatusMessage from "../components/StatusMessage";

function AnalysesList() {
  const { data, isPending, isError, error } = useQuery({
    queryKey: ["analyses"],
    queryFn: fetchAnalyses,
  });

  if (isPending) {
    return <StatusMessage>Loading analyses...</StatusMessage>;
  }

  if (isError) {
    return (
      <StatusMessage tone="error">
        Could not load the analyses: {error.message}
      </StatusMessage>
    );
  }

  if (data.length === 0) {
    return (
      <StatusMessage>
        No analysis yet. Run the pipeline on a video to see it here.
      </StatusMessage>
    );
  }

  return (
    <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {data.map((analysis) => (
        <li key={analysis.id}>
          <AnalysisCard analysis={analysis} />
        </li>
      ))}
    </ul>
  );
}

export default function AnalysesListPage() {
  return (
    <section>
      <h1 className="mb-6 text-2xl font-bold tracking-tight">Analyses</h1>
      <AnalysesList />
    </section>
  );
}
