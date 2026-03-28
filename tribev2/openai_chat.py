from __future__ import annotations

import json
import typing as tp

import pandas as pd

from tribev2.easy import (
    ImageComparisonRun,
    PredictionRun,
    collect_timestep_metadata,
    render_brain_panel_bytes,
    summarize_predictions,
)

DEFAULT_OPENAI_CHAT_MODEL = "gpt-5.4"

CHAT_SYSTEM_PROMPT = """
Tu es l'assistant d'analyse du dashboard TRIBE v2 Easy.
Ta tache est d'expliquer les sorties du modele a partir des donnees numeriques et des images des timesteps qui te sont fournies.

Regles:
- Appuie-toi d'abord sur les observations visibles dans les cartes et sur les stats jointes.
- Distingue clairement ce qui est observe, ce qui est une hypothese, et ce qui reste incertain.
- Ne presente pas la sortie comme un diagnostic medical, ni comme une lecture directe des intentions ou de l'etat mental.
- Si l'utilisateur demande une interpretation neuroscientifique, reste prudent et parle en termes de patterns plausibles, pas de certitude.
- Si la demande porte sur une comparaison, compare explicitement run 1 vs run 2 ou image 1 vs image 2.
- Reponds en francais sauf si l'utilisateur demande une autre langue.
""".strip()


def build_raw_timestep_frame(run: PredictionRun) -> pd.DataFrame:
    """Build a raw per-timestep table without hardcoded interpretive labels."""
    summary = summarize_predictions(run.preds)
    metadata = collect_timestep_metadata(run)
    rows: list[dict[str, tp.Any]] = []
    for idx, row in summary.iterrows():
        meta = metadata[idx] if idx < len(metadata) else {}
        rows.append(
            {
                "timestep": int(row["timestep"]),
                "start_s": round(float(meta.get("start", idx)), 3),
                "duration_s": round(float(meta.get("duration", 1.0)), 3),
                "text": str(meta.get("text", "") or ""),
                "mean": round(float(row["mean"]), 6),
                "std": round(float(row["std"]), 6),
                "mean_abs": round(float(row["mean_abs"]), 6),
                "max_abs": round(float(row["max_abs"]), 6),
            }
        )
    return pd.DataFrame(rows)


def _truncate_text(value: str | None, limit: int = 1800) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _select_key_timestep_indices(frame: pd.DataFrame, max_images: int = 4) -> list[int]:
    if frame.empty:
        return []
    candidates = [
        int(frame.iloc[0]["timestep"]),
        int(frame.iloc[len(frame) // 2]["timestep"]),
        int(frame.iloc[-1]["timestep"]),
        int(frame.sort_values("mean_abs", ascending=False).iloc[0]["timestep"]),
    ]
    top_dynamic = frame.sort_values("mean_abs", ascending=False)["timestep"].astype(int).tolist()
    candidates.extend(top_dynamic[: max(0, max_images * 2)])
    selected: list[int] = []
    for idx in candidates:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_images:
            break
    return selected


def _to_base64_data_url(raw_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    import base64

    return f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('ascii')}"


def _prediction_run_context_images(
    run: PredictionRun,
    *,
    run_label: str,
    image_detail: str,
    max_images: int,
) -> tuple[dict[str, tp.Any], list[dict[str, str]], list[str]]:
    frame = build_raw_timestep_frame(run)
    selected = _select_key_timestep_indices(frame, max_images=max_images)
    rows_for_prompt = frame.sort_values("mean_abs", ascending=False).head(8)
    payload = {
        "run_label": run_label,
        "input_kind": run.input_kind,
        "n_timesteps": int(len(run.preds)),
        "n_vertices": int(run.preds.shape[1]),
        "source_path": str(run.source_path) if run.source_path is not None else None,
        "raw_text_excerpt": _truncate_text(run.raw_text),
        "timestep_rows": rows_for_prompt.to_dict(orient="records"),
    }
    image_parts: list[dict[str, str]] = []
    labels: list[str] = []
    for idx in selected:
        row = frame.loc[frame["timestep"] == idx].iloc[0]
        labels.append(
            f"{run_label}: timestep {idx} ({float(row['start_s']):.2f}s, mean_abs={float(row['mean_abs']):.4f})"
        )
        image_parts.append(
            {
                "type": "input_image",
                "detail": image_detail,
                "image_url": _to_base64_data_url(
                    render_brain_panel_bytes(
                        run.preds,
                        timestep=idx,
                        image_format="JPEG",
                        quality=84,
                    )
                ),
            }
        )
    return payload, image_parts, labels


def build_openai_context_bundle(
    run: PredictionRun | ImageComparisonRun,
    *,
    image_detail: str = "low",
    max_images: int = 4,
) -> tuple[str, list[dict[str, str]], list[str]]:
    """Prepare the multimodal context sent to OpenAI."""
    if isinstance(run, ImageComparisonRun):
        per_run_limit = max(1, max_images // max(len(run.runs), 1))
        payload_runs = []
        image_parts: list[dict[str, str]] = []
        labels: list[str] = []
        for idx, item in enumerate(run.runs, start=1):
            payload, parts, run_labels = _prediction_run_context_images(
                item,
                run_label=f"image_{idx}",
                image_detail=image_detail,
                max_images=per_run_limit,
            )
            payload_runs.append(payload)
            image_parts.extend(parts)
            labels.extend(run_labels)
        context = {
            "kind": "tribev2_image_comparison",
            "n_images": len(run.runs),
            "runs": payload_runs,
        }
        return json.dumps(context, ensure_ascii=False, indent=2), image_parts[:max_images], labels[:max_images]

    payload, image_parts, labels = _prediction_run_context_images(
        run,
        run_label="run",
        image_detail=image_detail,
        max_images=max_images,
    )
    context = {
        "kind": "tribev2_prediction_run",
        "run": payload,
    }
    return json.dumps(context, ensure_ascii=False, indent=2), image_parts, labels


def extract_response_text(response: tp.Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            candidate = getattr(content, "text", None)
            if candidate:
                chunks.append(str(candidate))
    joined = "\n\n".join(part.strip() for part in chunks if str(part).strip())
    return joined or "Aucune reponse textuelle renvoyee par l'API."


def request_openai_run_explanation(
    *,
    api_key: str,
    model: str,
    reasoning_effort: str,
    user_prompt: str,
    run: PredictionRun | ImageComparisonRun,
    previous_response_id: str | None = None,
    include_context: bool = True,
    image_detail: str = "low",
    max_images: int = 4,
    context_bundle: tuple[str, list[dict[str, str]], list[str]] | None = None,
) -> tuple[str, str | None, list[str]]:
    """Send the current run plus the user prompt to OpenAI Responses API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    labels: list[str] = []
    content: list[dict[str, str]] = []
    if include_context:
        if context_bundle is None:
            context_bundle = build_openai_context_bundle(
                run,
                image_detail=image_detail,
                max_images=max_images,
            )
        context_text, image_parts, labels = context_bundle
        content.append(
            {
                "type": "input_text",
                "text": (
                    CHAT_SYSTEM_PROMPT
                    + "\n\nContexte TRIBE v2 du run courant:\n"
                    + context_text
                    + "\n\nUtilise ce contexte et les images jointes pour repondre a la question suivante."
                ),
            }
        )
        content.extend(image_parts)
    content.append({"type": "input_text", "text": user_prompt})
    response = client.responses.create(
        model=model,
        previous_response_id=previous_response_id,
        reasoning={"effort": reasoning_effort},
        input=[{"role": "user", "content": content}],
    )
    return extract_response_text(response), getattr(response, "id", None), labels
