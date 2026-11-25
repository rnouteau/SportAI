import { Link, useParams } from "react-router";

export default function AnalysisPage() {
  const { id } = useParams();

  return (
    <div>
      <Link to="/">← Back to the list</Link>
      <h1>Analysis {id}</h1>
    </div>
  );
}
