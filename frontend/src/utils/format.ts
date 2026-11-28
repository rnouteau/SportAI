export function formatValue(
  value: number | null | undefined,
  unit: string,
  digits = 1,
): string {
  if (value === null || value === undefined) {
    return "-";
  }
  return `${value.toFixed(digits)} ${unit}`;
}

/** "9.9 s" under a minute, then "4:05" or "1:02:09" (rounded to the second). */
export function formatDuration(seconds: number): string {
  const tenths = Math.round(seconds * 10) / 10;
  if (tenths < 60) {
    // Rounded first, so that 59.97 s does not show as "60.0 s"
    return `${tenths.toFixed(1)} s`;
  }
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, "0");
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${secs}`;
  }
  return `${minutes}:${secs}`;
}

/** Video file name without its extension: "match.mp4" -> "match". */
export function videoTitle(videoName: string): string {
  return videoName.replace(/\.[^./\\]+$/, "");
}
