import type { ReactNode } from "react";

type StatusMessageProps = {
  children: ReactNode;
  tone?: "neutral" | "error";
};

// Full class names written out: Tailwind only generates the classes it finds in the code
const TONE_CLASSES = {
  neutral: "border-slate-200 bg-white text-slate-600",
  error: "border-red-200 bg-red-50 text-red-700",
};

/** Boxed message for the loading, error and empty states of a page. */
export default function StatusMessage({
  children,
  tone = "neutral",
}: StatusMessageProps) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={`rounded-xl border px-6 py-10 text-center ${TONE_CLASSES[tone]}`}
    >
      {children}
    </div>
  );
}
