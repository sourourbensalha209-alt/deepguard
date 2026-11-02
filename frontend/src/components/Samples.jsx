import { useState } from "react";

const SAMPLES = [1, 2, 3, 4].map((n) => ({ n, url: `/samples/sample_${n}.jpg` }));

function Sample({ n, url, onPick, onLoaded }) {
  const [ok, setOk] = useState(false);
  const pick = async () => {
    const blob = await (await fetch(url)).blob();
    onPick(new File([blob], `exemple_${n}.jpg`, { type: blob.type || "image/jpeg" }));
  };
  return (
    <button
      type="button"
      onClick={pick}
      hidden={!ok}
      aria-label={`Analyser l'exemple ${n}`}
      className="print-shadow bg-white p-1 hover:-translate-y-0.5"
    >
      <img
        src={url}
        alt=""
        width="64"
        height="64"
        className="block size-16 object-cover"
        onLoad={() => {
          setOk(true);
          onLoaded();
        }}
        onError={() => setOk(false)}
      />
    </button>
  );
}

export default function Samples({ onPick }) {
  const [loaded, setLoaded] = useState(0);
  return (
    <div className={loaded ? "mt-5" : "hidden"}>
      <p className="mb-2 text-sm text-muted">Pas d'image sous la main ? Essaie un exemple.</p>
      <div className="flex flex-wrap gap-3">
        {SAMPLES.map((s) => (
          <Sample key={s.n} {...s} onPick={onPick} onLoaded={() => setLoaded((c) => c + 1)} />
        ))}
      </div>
    </div>
  );
}
