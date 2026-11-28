import { describe, expect, it } from "vitest";
import { formatDuration, formatValue, videoTitle } from "./format";

describe("formatValue", () => {
  it("rounds to one decimal and adds the unit", () => {
    expect(formatValue(17.04, "m")).toBe("17.0 m");
  });

  it("uses the requested number of digits", () => {
    expect(formatValue(14.78, "km/h", 0)).toBe("15 km/h");
  });

  it("shows a dash when the value is missing", () => {
    expect(formatValue(null, "m")).toBe("-");
    expect(formatValue(undefined, "m")).toBe("-");
  });

  it("keeps zero (a player who did not move)", () => {
    expect(formatValue(0, "m")).toBe("0.0 m");
  });
});

describe("formatDuration", () => {
  it("shows seconds under a minute", () => {
    expect(formatDuration(9.86)).toBe("9.9 s");
  });

  it("does not show 60.0 s just under a minute", () => {
    expect(formatDuration(59.97)).toBe("1:00");
  });

  it("shows minutes and seconds", () => {
    expect(formatDuration(360)).toBe("6:00");
    expect(formatDuration(245.4)).toBe("4:05");
  });

  it("shows hours for a full match", () => {
    expect(formatDuration(3729)).toBe("1:02:09");
  });
});

describe("videoTitle", () => {
  it("removes the extension", () => {
    expect(videoTitle("match.mp4")).toBe("match");
  });

  it("only removes the last extension", () => {
    expect(videoTitle("match.v2.mp4")).toBe("match.v2");
  });

  it("keeps a name without extension", () => {
    expect(videoTitle("match")).toBe("match");
  });
});
