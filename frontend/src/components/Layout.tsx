import { Link, Outlet } from "react-router";

export default function Layout() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-baseline gap-3 px-4 py-4">
          <Link to="/" className="text-xl font-bold tracking-tight">
            Sport<span className="text-blue-600">AI</span>
          </Link>
          <span className="text-sm text-slate-500">Tennis video analysis</span>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-8">
        <Outlet />
      </main>
    </div>
  );
}
