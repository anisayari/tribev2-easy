from __future__ import annotations

import json
import logging
import typing as tp

import pandas as pd

from tribev2.easy import (
    ImageComparisonRun,
    MultiModalRun,
    PredictionRun,
    collect_timestep_metadata,
    render_run_panel_bytes,
    summarize_predictions,
)

render_brain_panel_bytes = render_run_panel_bytes

DEFAULT_OPENAI_CHAT_MODEL = "gpt-5.4"
LOGGER = logging.getLogger("tribev2.openai_chat")

COMMON_UNCERTAINTIES = [
    "l'ordre exact des images si l'affichage ne suit pas strictement t0→tN ou si seules des frames cles ont ete jointes",
    "la comparabilite stricte des intensites visuelles sans colorbar commune ou sans confirmation d'une normalisation identique",
    "l'attribution anatomique fine a une region precise plutot qu'a un grand systeme cortical",
    "toute interpretation neuroscientifique forte, car il s'agit de predictions TRIBE v2 et non de mesures fMRI reelles",
]


def _build_pipeline_summary() -> list[str]:
    return [
        "Le dashboard convertit la source brute en evenements ou segments temporels.",
        "Des encodeurs fondation extraient des representations video, audio et texte selon la modalite disponible.",
        "TRIBE v2 projette ces representations vers une activite corticale predite sur la surface cerebrale.",
        "Chaque timestep correspond a un segment predit du stimulus, pas a une mesure cerebrale enregistree.",
    ]


def _build_modality_notes(run: PredictionRun | ImageComparisonRun) -> list[str]:
    if isinstance(run, ImageComparisonRun):
        return [
            "Le run compare deux images traitees independamment puis affichees cote a cote.",
            "Chaque image fixe est convertie en court clip video silencieux statique pour passer dans TRIBE v2.",
            "Les timesteps pour une image ne racontent donc pas une evolution temporelle reelle; ils repetent le meme contenu visuel.",
            "La comparaison doit surtout porter sur la distribution spatiale, la lateralite, la saillance et le ton affectif plausible du contenu visuel.",
        ]
    if isinstance(run, MultiModalRun) or run.input_kind == "multimodal":
        return [
            "La source combine plusieurs modalites dans un meme run.",
            "Le dashboard calcule un run fusionne pour la prediction principale, puis des runs separes par modalite pour colorer le cerveau par contribution.",
            "Le code couleur de l'overlay est: rouge pour le visuel, vert pour l'audio, bleu pour le texte.",
            "Les couleurs representent une contribution relative par famille de modalite au meme timestep, pas des regions anatomiques differentes.",
        ]
    if run.input_kind == "video":
        return [
            "La source est une video. Chaque timestep correspond a un segment temporel du clip.",
            "L'interpretation peut relier les cartes au contenu visuel, au rythme, a la parole et au son s'ils sont presents dans le segment.",
            "Les emotions ou ressentis doivent etre formules comme hypotheses plausibles sur le stimulus, pas comme lecture directe d'un etat mental.",
        ]
    if run.input_kind == "audio":
        return [
            "La source est un audio. Chaque timestep correspond a un segment sonore du fichier.",
            "Si un texte aligne existe, il sert d'indice complementaire, mais la source principale reste le signal acoustique.",
            "L'interpretation peut commenter prosodie, tension, calme, menace, joie ou tristesse plausibles, en les liant au stimulus sonore.",
        ]
    if run.input_kind == "text":
        return [
            "La source est un texte. Dans ce fork, le mode texte direct construit des timings synthetiques mot par mot ou segment par segment.",
            "Les timesteps refletent donc un decoupage artificiel du texte, pas un enregistrement audio ou video reel.",
            "Les emotions ou ressentis peuvent etre deduits du champ lexical et du ton du texte, puis confrontes a la carte predite avec prudence.",
        ]
    if run.input_kind == "image":
        return [
            "La source est une image fixe.",
            "Dans ce fork, l'image est convertie en court clip video silencieux statique pour passer dans TRIBE v2.",
            "Les timesteps representent des passes repetees sur le meme contenu visuel, pas une narration temporelle reelle.",
            "L'interpretation doit surtout porter sur la structure visuelle, la saillance spatiale et le ton affectif plausible du contenu de l'image.",
        ]
    return [
        "La modalite doit etre lue a partir du contexte joint.",
        "Les timesteps correspondent a des segments predits par TRIBE v2.",
    ]


def _build_interpretation_contract(run: PredictionRun | ImageComparisonRun) -> dict[str, tp.Any]:
    comparison_hint = (
        "Si deux images ou deux runs sont presents, compare explicitement image 1 vs image 2 ou run 1 vs run 2."
        if isinstance(run, ImageComparisonRun)
        else "Si l'utilisateur compare plusieurs moments, signale clairement les differences de pattern d'un timestep a l'autre."
    )
    return {
        "mission": [
            "Expliquer ce que montre la carte d'activite corticale predite.",
            "Dire a quoi le pattern est typiquement associe: vision, audition, langage, attention, saillance, memoire de travail, cognition sociale ou charge affective plausible.",
            "Si l'utilisateur le demande, proposer une lecture prudente de valence ou d'emotions plausibles: positive, negative, mixte, peur, desir, joie, tristesse, colere, calme, tension, menace, surprise.",
            "Toujours distinguer observation, hypothese et incertitude.",
        ],
        "format_attendu": [
            "1. Ce qu'on voit",
            "2. A quoi c'est typique",
            "3. Emotion ou ressenti plausible",
            "4. Indices du stimulus et des donnees qui soutiennent cette lecture",
            "5. Ce qui reste incertain",
        ],
        "regles": [
            "Commence par la modalite et rappelle ce que represente un timestep dans ce run.",
            "Ne parle jamais comme si TRIBE v2 mesurait directement l'activite cerebrale: ce sont des predictions du modele.",
            "Ne transforme pas une emotion plausible en certitude.",
            comparison_hint,
            "Reponds en francais sauf si l'utilisateur demande une autre langue.",
        ],
        "incertitudes_a_mentionner": COMMON_UNCERTAINTIES,
    }


def build_chat_system_prompt(run: PredictionRun | ImageComparisonRun) -> str:
    sections = [
        "Tu es l'assistant d'analyse du dashboard TRIBE v2 Easy.",
        "Ta tache est d'expliquer les sorties du modele a partir des donnees numeriques et des images de timesteps qui te sont fournies.",
        "",
        "Comment fonctionne l'experience TRIBE v2 dans ce dashboard:",
        *[f"- {item}" for item in _build_pipeline_summary()],
        "",
        "Comment lire la modalite courante:",
        *[f"- {item}" for item in _build_modality_notes(run)],
        "",
        "Cadre d'interpretation:",
        *[f"- {item}" for item in _build_interpretation_contract(run)["mission"]],
        "",
        "Format de reponse attendu:",
        *[f"- {item}" for item in _build_interpretation_contract(run)["format_attendu"]],
        "",
        "Regles:",
        *[f"- {item}" for item in _build_interpretation_contract(run)["regles"]],
        "",
        "Ce qui reste incertain:",
        *[f"- {item}" for item in COMMON_UNCERTAINTIES],
        "",
        "Interdits:",
        "- pas de diagnostic medical",
        "- pas de lecture directe des intentions ou de l'etat mental",
        "- pas d'affirmation anatomique trop fine sans preuve",
        "- pas d'interpretation neuroscientifique forte presentee comme certaine",
    ]
    return "\n".join(sections).strip()


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
    modality_notes = _build_modality_notes(run)
    payload = {
        "run_label": run_label,
        "input_kind": run.input_kind,
        "n_timesteps": int(len(run.preds)),
        "n_vertices": int(run.preds.shape[1]),
        "source_path": str(run.source_path) if run.source_path is not None else None,
        "raw_text_excerpt": _truncate_text(run.raw_text),
        "tribev2_pipeline_summary": _build_pipeline_summary(),
        "modality_notes": modality_notes,
        "interpretation_contract": _build_interpretation_contract(run),
        "selected_timestep_image_policy": (
            "Les images jointes sont des timesteps cles choisis automatiquement a partir du debut, du milieu, de la fin et des pics de mean_abs."
        ),
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
                    render_run_panel_bytes(
                        run,
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
            "tribev2_pipeline_summary": _build_pipeline_summary(),
            "comparison_notes": _build_modality_notes(run),
            "interpretation_contract": _build_interpretation_contract(run),
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

    LOGGER.info(
        "OpenAI request start | model=%s | include_context=%s | max_images=%s | comparison=%s",
        model,
        include_context,
        max_images,
        isinstance(run, ImageComparisonRun),
    )
    client = OpenAI(api_key=api_key)
    labels: list[str] = []
    content: list[dict[str, str]] = []
    input_items: list[dict[str, tp.Any]] = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": build_chat_system_prompt(run)}],
        }
    ]
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
                    "Contexte TRIBE v2 du run courant:\n"
                    + context_text
                    + "\n\nUtilise ce contexte et les images jointes pour repondre a la question suivante."
                ),
            }
        )
        content.extend(image_parts)
    content.append({"type": "input_text", "text": user_prompt})
    input_items.append({"role": "user", "content": content})
    response = client.responses.create(
        model=model,
        previous_response_id=previous_response_id,
        reasoning={"effort": reasoning_effort},
        input=input_items,
    )
    LOGGER.info(
        "OpenAI request complete | model=%s | response_id=%s",
        model,
        getattr(response, "id", None),
    )
    return extract_response_text(response), getattr(response, "id", None), labels
