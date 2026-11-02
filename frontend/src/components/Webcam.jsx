import { useEffect, useRef, useState } from "react";

export default function Webcam({ onCapture, onCancel }) {
  const videoRef = useRef(null);
  const [cameraError, setCameraError] = useState(null);
  const [ready, setReady] = useState(false);
  const unsupported = !navigator.mediaDevices?.getUserMedia;
  const error = unsupported
    ? "La caméra n'est disponible que sur localhost ou en HTTPS. Envoie plutôt une image."
    : cameraError;

  useEffect(() => {
    let stream = null;
    let cancelled = false;
    if (unsupported) return undefined;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "user" }, audio: false })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        videoRef.current.srcObject = s;
      })
      .catch(() =>
        setCameraError("Caméra inaccessible. Autorise-la dans le navigateur, ou envoie une image."),
      );
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [unsupported]);

  const capture = () => {
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => blob && onCapture(new File([blob], "webcam.jpg", { type: "image/jpeg" })),
      "image/jpeg",
      0.92,
    );
  };

  return (
    <div className="lightbox flex min-h-[26rem] flex-col items-center justify-center gap-5 border border-rule p-6">
      {error ? (
        <p role="alert" className="max-w-sm text-center text-fake">
          {error}
        </p>
      ) : (
        <div className="print-shadow bg-white p-2">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            onLoadedMetadata={() => setReady(true)}
            className="block max-h-[56vh] max-w-full -scale-x-100"
          />
        </div>
      )}
      <div className="flex gap-3">
        {!error && (
          <button
            type="button"
            onClick={capture}
            disabled={!ready}
            className="bg-navy px-5 py-2.5 font-medium text-paper hover:bg-ink disabled:opacity-50"
          >
            Prendre la photo
          </button>
        )}
        <button
          type="button"
          onClick={onCancel}
          className="border border-ink px-5 py-2.5 font-medium hover:bg-paper"
        >
          Annuler
        </button>
      </div>
    </div>
  );
}
