export const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const MAX_BYTES = 5 * 1024 * 1024;
export const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

export async function getMetadata(signal) {
  const res = await fetch(`${API}/model/metadata`, { signal });
  if (!res.ok) throw new Error("metadata");
  return res.json();
}

export async function analyzeImage(file, signal) {
  const form = new FormData();
  form.append("file", file);
  const t0 = performance.now();
  let res;
  try {
    res = await fetch(`${API}/predict/visualize`, { method: "POST", body: form, signal });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error(`Le serveur ne répond pas. Vérifie que l'API tourne sur ${API}.`);
  }
  if (!res.ok) {
    let detail = null;
    try {
      detail = (await res.json()).detail;
    } catch {
      /* corps non JSON */
    }
    throw new Error(typeof detail === "string" ? detail : `Erreur du serveur (code ${res.status}).`);
  }
  const data = await res.json();
  return { ...data, clientMs: Math.round(performance.now() - t0) };
}
