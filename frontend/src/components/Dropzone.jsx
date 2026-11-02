import { useRef, useState } from "react";

/** Zone de dépôt (glisser-déposer) + choix de fichier + accès webcam. */
export default function Dropzone({ onFile, onCamera }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`lightbox flex min-h-[26rem] flex-col items-center justify-center gap-5 border p-6 text-center ${
        dragging ? "border-navy" : "border-rule"
      }`}
    >
      <div
        className={`flex w-full max-w-sm flex-col items-center gap-2 border-2 border-dashed bg-paper/70 px-6 py-10 ${
          dragging ? "border-navy" : "border-rule"
        }`}
      >
        <p className="font-display text-2xl font-medium">Dépose une photo de visage</p>
        <p className="text-sm text-muted">JPEG, PNG ou WebP, 5 Mo maximum.</p>
      </div>
      <div className="flex flex-wrap justify-center gap-3">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="bg-navy px-5 py-2.5 font-medium text-paper hover:bg-ink"
        >
          Choisir une image
        </button>
        <button
          type="button"
          onClick={onCamera}
          className="border border-ink px-5 py-2.5 font-medium hover:bg-paper"
        >
          Utiliser la webcam
        </button>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) onFile(f);
        }}
      />
    </div>
  );
}
