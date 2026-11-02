import { useCallback, useEffect, useRef, useState } from "react";
import { ALLOWED_TYPES, analyzeImage, API, getMetadata, MAX_BYTES } from "./services/api";
import Dropzone from "./components/Dropzone";
import HeatmapCanvas from "./components/HeatmapCanvas";
import Report from "./components/Report";
import Samples from "./components/Samples";
import Webcam from "./components/Webcam";

export default function App() {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [error, setError] = useState(null);
  const [meta, setMeta] = useState(null);
  const [apiDown, setApiDown] = useState(false);
  const [camOn, setCamOn] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    getMetadata(ctrl.signal)
      .then(setMeta)
      .catch((e) => e.name !== "AbortError" && setApiDown(true));
    return () => ctrl.abort();
  }, []);

  const jobRef = useRef({ ctrl: null, url: null });

  const stopJob = useCallback(() => {
    jobRef.current.ctrl?.abort();
    if (jobRef.current.url) URL.revokeObjectURL(jobRef.current.url);
    jobRef.current = { ctrl: null, url: null };
  }, []);

  useEffect(() => stopJob, [stopJob]);

  const fail = useCallback(
    (message) => {
      stopJob();
      setPreviewUrl(null);
      setResult(null);
      setError(message);
      setStatus("error");
    },
    [stopJob],
  );

  // Une nouvelle image lance l'analyse ; une réponse périmée est ignorée.
  const choose = useCallback(
    (f) => {
      setCamOn(false);
      if (!ALLOWED_TYPES.includes(f.type))
        return fail("Format non supporté : choisis une image JPEG, PNG ou WebP.");
      if (f.size > MAX_BYTES) return fail("Image trop lourde : 5 Mo maximum.");

      stopJob();
      const ctrl = new AbortController();
      const url = URL.createObjectURL(f);
      jobRef.current = { ctrl, url };
      setPreviewUrl(url);
      setResult(null);
      setError(null);
      setStatus("loading");

      analyzeImage(f, ctrl.signal)
        .then((data) => {
          setResult(data);
          setStatus("done");
          setApiDown(false);
        })
        .catch((e) => {
          if (e.name === "AbortError") return;
          setError(e.message);
          setStatus("error");
        });
      return undefined;
    },
    [fail, stopJob],
  );

  const reset = () => {
    stopJob();
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setStatus("idle");
  };

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-5 sm:px-8">
      <header className="flex flex-wrap items-baseline justify-between gap-2 border-b border-ink py-5">
        <h1 className="font-display text-3xl font-semibold tracking-tight text-navy">DeepGuard</h1>
        <p className="text-sm text-muted">
          {apiDown
            ? "API injoignable : lance le serveur (voir le README)"
            : meta
              ? meta.demo
                ? "Modèle de démonstration"
                : `Modèle : ${meta.architecture}`
              : "Connexion à l'API…"}
        </p>
      </header>

      {meta?.demo && (
        <p role="note" className="border-b border-rule bg-paper px-3 py-3 text-sm text-caution">
          Modèle de démonstration : ses poids sont aléatoires, donc ses résultats n'ont aucune valeur.
          Entraîne et exporte un vrai modèle pour des prédictions réelles (voir le README).
        </p>
      )}

      <main className="grid flex-1 gap-10 py-8 lg:grid-cols-[1.35fr_1fr] lg:gap-14">
        <section aria-label="Image à analyser">
          {camOn ? (
            <Webcam onCapture={choose} onCancel={() => setCamOn(false)} />
          ) : previewUrl ? (
            <HeatmapCanvas previewUrl={previewUrl} result={result} busy={status === "loading"} />
          ) : (
            <Dropzone onFile={choose} onCamera={() => setCamOn(true)} />
          )}
          {!previewUrl && !camOn && <Samples onPick={choose} />}
        </section>

        <section aria-label="Résultat" className="lg:border-l lg:border-rule lg:pl-12">
          <Report status={status} result={result} error={error} meta={meta} onReset={reset} />
        </section>
      </main>

      <footer className="flex flex-wrap justify-between gap-2 border-t border-rule py-4 text-sm text-muted">
        <span>Les images sont analysées en mémoire et ne sont pas conservées.</span>
        <a href={`${API}/docs`} className="underline underline-offset-2 hover:text-ink">
          Documentation de l'API
        </a>
      </footer>
    </div>
  );
}
