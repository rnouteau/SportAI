import type { components } from "./types";

export type AnalysisSummary = components["schemas"]["AnalysisSummary"];
export type PlayerSummary = components["schemas"]["PlayerSummary"];

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} (${url})`);
  }
  return (await response.json()) as T;
}

export function fetchAnalyses(): Promise<AnalysisSummary[]> {
  return getJson<AnalysisSummary[]>("/api/analyses");
}

export function fetchAnalysis(id: string): Promise<AnalysisSummary> {
  return getJson<AnalysisSummary>(`/api/analyses/${encodeURIComponent(id)}`);
}
