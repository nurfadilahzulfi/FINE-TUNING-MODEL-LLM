"""
Evaluasi model AES (fine-tuned atau baseline prompting) terhadap test.jsonl.

Metrik utama: Quadratic Weighted Kappa (QWK) -- standar de-facto untuk
Automated Essay Scoring karena skor bersifat ordinal (0 < 5 < 10), bukan
kategori nominal biasa. Accuracy saja menyembunyikan seberapa jauh model
"meleset" (skor 0 diprediksi 10 jauh lebih buruk daripada diprediksi 5).

Cara pakai:
    1. Isi fungsi `predict_score()` di bawah untuk memanggil model kamu
       (Ollama endpoint fine-tuned ATAU prompting baseline -- jalankan
       script ini dua kali, ganti fungsi predict, lalu bandingkan QWK).
    2. python3 03_evaluate.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import requests
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TEST_PATH = Path("./data/test.jsonl")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama32-aes"  # ganti sesuai nama model hasil `ollama create`
VALID_SCORES = [0, 5, 10]


def predict_score(system_prompt: str, user_prompt: str) -> int | None:
    """Panggil Ollama, parse skor dari respons JSON. Return None jika gagal parse."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},  # deterministik untuk evaluasi
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
    except (requests.RequestException, KeyError) as exc:
        logger.warning("Gagal memanggil Ollama: %s", exc)
        return None

    match = re.search(r'"skor"\s*:\s*(\d+)', content)
    if not match:
        logger.warning("Tidak bisa parse skor dari respons: %r", content[:200])
        return None

    score = int(match.group(1))
    if score not in VALID_SCORES:
        logger.warning("Skor di luar skala valid (%s): %d", VALID_SCORES, score)
        return None
    return score


def load_test_set(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ditemukan -- jalankan 01_prepare_dataset.py dulu.")
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main() -> None:
    records = load_test_set(TEST_PATH)
    logger.info("Mengevaluasi %d contoh dari test set...", len(records))

    y_true: list[int] = []
    y_pred: list[int] = []
    n_failed = 0

    for i, rec in enumerate(records):
        system_msg = rec["messages"][0]["content"]
        user_msg = rec["messages"][1]["content"]
        true_score = json.loads(rec["messages"][2]["content"])["skor"]

        pred_score = predict_score(system_msg, user_msg)
        if pred_score is None:
            n_failed += 1
            continue

        y_true.append(true_score)
        y_pred.append(pred_score)

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d", i + 1, len(records))

    if n_failed:
        logger.warning("%d/%d prediksi gagal diparse dan dikecualikan dari metrik.", n_failed, len(records))

    if not y_true:
        logger.error("Tidak ada prediksi valid -- cek koneksi Ollama / nama model.")
        return

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    logger.info("Quadratic Weighted Kappa: %.4f", qwk)
    logger.info(
        "Interpretasi kasar QWK: <0.4 lemah | 0.4-0.6 cukup | 0.6-0.8 baik | >0.8 sangat baik"
    )

    logger.info("Confusion matrix (baris=asli, kolom=prediksi), label %s:\n%s",
                VALID_SCORES, confusion_matrix(y_true, y_pred, labels=VALID_SCORES))
    logger.info("Classification report:\n%s", classification_report(y_true, y_pred, labels=VALID_SCORES))


if __name__ == "__main__":
    main()
