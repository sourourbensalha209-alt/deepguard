import { useEffect, useState } from "react";

function Bar({ label, value, tone, shown }) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between text-sm">
        <span>{label}</span>
        <span className="font-semibold tabular-nums">{pct} %</span>
      </div>
      <div
        className="mt-1 h-3 w-full border border-ink bg-paper"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
      >
        <div
          className={`h-full ${tone} transition-[width] duration-700 ease-out`}
          style={{ width: shown ? `${pct}%` : "0%" }}
        />
      </div>
    </div>
  );
}

/** Scores de probabilité en barres de progression animées. */
export default function ProbabilityChart({ probFake }) {
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  return (
    <div className="mt-6 space-y-4">
      <Bar label="Réel" value={1 - probFake} tone="bg-real" shown={shown} />
      <Bar label="Généré par IA" value={probFake} tone="bg-fake" shown={shown} />
    </div>
  );
}
