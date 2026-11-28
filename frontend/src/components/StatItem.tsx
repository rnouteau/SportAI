type StatItemProps = {
  label: string;
  value: string;
  size?: "md" | "lg";
};

/** One label/value pair, to put inside a <dl> (description list). */
export default function StatItem({ label, value, size = "md" }: StatItemProps) {
  return (
    <div>
      <dt className="text-xs font-medium tracking-wide text-slate-500 uppercase">
        {label}
      </dt>
      <dd
        className={`mt-1 font-semibold whitespace-nowrap tabular-nums ${size === "lg" ? "text-lg sm:text-xl" : "text-base"}`}
      >
        {value}
      </dd>
    </div>
  );
}
