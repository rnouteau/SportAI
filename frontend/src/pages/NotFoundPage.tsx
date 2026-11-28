import { Link } from "react-router";
import StatusMessage from "../components/StatusMessage";

export default function NotFoundPage() {
  return (
    <StatusMessage>
      <h1 className="text-lg font-semibold text-slate-900">Page not found</h1>
      <Link
        to="/"
        className="mt-2 inline-block text-sm font-medium text-blue-600 hover:text-blue-700"
      >
        Back to the analyses
      </Link>
    </StatusMessage>
  );
}
