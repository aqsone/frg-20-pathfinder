import json

from google import genai
from google.genai import types

from src.rag.utils import call_with_retry

JUDGE_MODEL = "gemini-2.5-flash"

_PROMPT = """\
Tu es un évaluateur expert en aéronautique. Compare la réponse générée à la réponse de référence.

Question : {question}

Réponse de référence : {reference_answer}

Réponse générée : {generated_answer}

Critères (1 à 5) :
1 = totalement incorrecte ou hors sujet
2 = en grande partie incorrecte
3 = partiellement correcte (information manquante ou imprécise)
4 = correcte avec de légères imprécisions
5 = correcte et complète

Réponds UNIQUEMENT avec un objet JSON : {{"score": <entier 1-5>, "justification": "<explication courte en français>"}}"""


def judge_answer(
    question: str,
    reference_answer: str,
    generated_answer: str,
    client: genai.Client,
    model: str = JUDGE_MODEL,
) -> dict:
    prompt = _PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )
    response = call_with_retry(
        client.models.generate_content,
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        result = json.loads(response.text)
        return {"score": int(result["score"]), "justification": result.get("justification", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": -1, "justification": f"erreur de parsing : {response.text[:100]}"}