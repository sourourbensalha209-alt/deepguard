import { useEffect, useState } from "react";

/** Calque de chaleur + curseur de comparaison. Monté seulement quand le résultat existe. */
function HeatCompare({ result }) {
  const [reveal, setReveal] = useState(0);
  const [sweeping, setSweeping] = useState(true);

  // Le calque « balaie » la photo une fois, à l'arrivée du résultat.
  useEffect(() => {
    const start = requestAnimationFrame(() => setReveal(100));
    const end = setTimeout(() => setSweeping(false), 900);
    return () => {
      cancelAnimationFrame(start);
      clearTimeout(end);
    };
  }, []);

  const [bx, by, bw, bh] = result.face_box;
  const pct = (v) => `${(v * 100).toFixed(2)}%`;

  return (
    <>
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 ease-out"
        style={{
          clipPath: `inset(0 ${100 - reveal}% 0 0)`,
          transition: sweeping ? "clip-path 800ms cubic-bezier(0.22, 1, 0.36, 1)" : "none",
        }}
      >
        <img
          src={result.heatmap}
          alt=""
          className="absolute"
          style={{ left: pct(bx), top: pct(by), width: pct(bw), height: pct(bh) }}
        />
      </div>

      {result.face_detected && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute border border-dashed border-white mix-blend-difference"
          style={{ left: pct(bx), top: pct(by), width: pct(bw), height: pct(bh) }}
        />
      )}

      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgb(18_38_58_/_0.5)]"
        style={{ left: `${reveal}%` }}
      >
        <span className="absolute top-1/2 left-1/2 size-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink bg-white" />
      </div>

      <input
        type="range"
        min="0"
        max="100"
        value={reveal}
        onChange={(e) => setReveal(Number(e.target.value))}
        aria-label="Comparer la photo et les zones qui ont pesé dans la décision"
        className="absolute inset-0 size-full cursor-ew-resize opacity-0"
      />
    </>
  );
}

/** Photo analysée + calque de chaleur (CAM ou Attention Rollout) + curseur pour comparer. */
export default function HeatmapCanvas({ previewUrl, result, busy }) {
  return (
    <div className="lightbox flex min-h-[26rem] flex-col items-center justify-center gap-4 border border-rule p-6">
      <div className="print-shadow bg-white p-2">
        <div className="relative leading-none">
          <img src={previewUrl} alt="Photo à analyser" className="block max-h-[56vh] max-w-full" />

          {result && <HeatCompare result={result} />}

          {busy && (
            <div
              role="status"
              className="absolute inset-0 grid place-items-center bg-paper/70 text-lg font-medium"
            >
              Analyse en cours…
            </div>
          )}
        </div>
      </div>
      {result && (
        <p className="text-sm text-muted">Fais glisser sur la photo : à gauche, les zones qui ont pesé.</p>
      )}
    </div>
  );
}
