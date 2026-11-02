import ProbabilityChart from "./ProbabilityChart";

const UNSURE_BELOW = 0.65;

function verdictOf(result) {
  if (result.confidence < UNSURE_BELOW) {
    return {
      title: "Résultat peu sûr",
      text: "Le modèle hésite entre les deux : ne conclus rien à partir de cette seule image.",
      tone: "text-caution",
    };
  }
  if (result.label === "FAKE") {
    return {
      title: "Probablement générée par IA",
      text: "Le modèle a repéré des motifs typiques d'images synthétiques.",
      tone: "text-fake",
    };
  }
  return {
    title: "Probablement réelle",
    text: "Le modèle n'a pas trouvé de trace de génération par IA.",
    tone: "text-real",
  };
}

function Row({ term, children }) {
  return (
    <div className="grid grid-cols-[9rem_1fr] gap-3 border-t border-rule py-2.5 text-sm">
      <dt className="text-muted">{term}</dt>
      <dd>{children}</dd>
    </div>
  );
}

export default function Report({ status, result, error, meta, onReset }) {
  if (status === "error") {
    return (
      <div role="alert">
        <h2 className="font-display text-3xl font-medium text-fake">L'analyse a échoué</h2>
        <p className="mt-3">{error}</p>
        <button
          type="button"
          onClick={onReset}
          className="mt-6 border border-ink px-5 py-2.5 font-medium hover:bg-paper"
        >
          Essayer une autre image
        </button>
      </div>
    );
  }

  if (!result) {
    return (
      <div>
        <h2 className="font-display text-3xl font-medium">Vérifier un visage</h2>
        <p className="mt-3 max-w-prose">
          DeepGuard estime si un visage est réel ou généré par IA, et montre les zones de la photo qui ont
          pesé dans sa décision.
        </p>
        <h3 className="mt-8 font-semibold">Pour un résultat fiable</h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-muted">
          <li>Un seul visage, de face, bien éclairé.</li>
          <li>Une photo nette, sans filtre ni recadrage serré.</li>
          <li>Une résolution d'au moins 256 pixels de côté.</li>
        </ul>
        <p className="mt-8 text-sm text-muted" aria-live="polite">
          {status === "loading" ? "Analyse en cours…" : "L'analyse démarre dès que tu choisis une image."}
        </p>
      </div>
    );
  }

  const v = verdictOf(result);
  return (
    <div aria-live="polite">
      <h2 className={`font-display text-4xl leading-tight font-medium ${v.tone}`}>{v.title}</h2>
      <p className="mt-3 max-w-prose">{v.text}</p>

      <ProbabilityChart probFake={result.prob_fake} />

      <dl className="mt-8">
        <Row term="Visage détecté">
          {result.face_detected
            ? "Oui, la photo a été recadrée sur le visage."
            : "Non, l'image entière a été analysée."}
        </Row>
        <Row term="Calcul serveur">{result.latency_ms.toFixed(0)} ms</Row>
        <Row term="Temps total">{result.clientMs} ms, envoi et affichage compris</Row>
        <Row term="Modèle">
          {meta ? (meta.demo ? "démonstration (poids aléatoires)" : meta.architecture) : "inconnu"}
        </Row>
        <div className="border-t border-rule" />
      </dl>

      {!result.face_detected && (
        <p className="mt-4 text-sm text-caution">
          Aucun visage détecté : cadre un visage de face pour un résultat plus fiable.
        </p>
      )}
      <p className="mt-4 text-sm text-muted">
        Un modèle peut se tromper, surtout sur des images très différentes de celles de son entraînement.
        N'utilise pas ce résultat seul comme preuve.
      </p>

      <button
        type="button"
        onClick={onReset}
        className="mt-6 bg-navy px-5 py-2.5 font-medium text-paper hover:bg-ink"
      >
        Analyser une autre image
      </button>
    </div>
  );
}
