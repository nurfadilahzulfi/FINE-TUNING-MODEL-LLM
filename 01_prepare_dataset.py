"""
Data preparation pipeline untuk fine-tuning LLM lokal (Llama 3.2 3B via Ollama)
sebagai Automated Essay Scoring (AES).

Input : Dataset_Machine_Learning.xlsx (sheet 'Soal' + 'Jawaban')
Output: train.jsonl, val.jsonl, test.jsonl (format chat SFT untuk Unsloth/TRL)
        + cleaned_dataset.csv (untuk audit manual / isi rubrik)

Catatan penting:
- RUBRIK belum ada di data asli. Kolom `rubrik` di cleaned_dataset.csv sengaja
  dikosongkan agar diisi manual (3-5 poin kunci per soal). Tanpa rubrik,
  fine-tuning hanya akan mengajarkan gaya/format penilaian, bukan kriteria akademik.
- Split dilakukan stratified per (Kode Soal x Skor) agar tiap kelas skor
  terwakili di train/val/test secara proporsional -> evaluasi QWK jadi valid.
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("/mnt/user-data/uploads/Dataset_Machine_Learning.xlsx")
OUTPUT_DIR = Path("/home/claude/aes_pipeline/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VALID_SCORES = {0.0, 5.0, 10.0}  # skala penilaian resmi sesuai kolom 'Skor (0/5/10)'

SYSTEM_PROMPT = (
    "Anda adalah asisten penilai esai untuk mata kuliah Machine Learning. "
    "Nilai jawaban mahasiswa HANYA berdasarkan rubrik yang diberikan. "
    "Berikan skor (0, 5, atau 10) dan alasan singkat dalam Bahasa Indonesia. "
    "Format jawaban WAJIB JSON: {\"skor\": <0|5|10>, \"alasan\": \"...\"}"
)


def clean_text(text: object) -> str | None:
    """Normalisasi teks jawaban: rapikan newline literal, whitespace, dan string kosong."""
    if not isinstance(text, str):
        return None
    text = text.replace("\\n", "\n").strip()
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    if text in {"", "-", "NaN", "nan"}:
        return None
    return text


def load_and_merge(path: Path) -> pd.DataFrame:
    """Load kedua sheet, join by Kode Soal, dan buang baris yang tidak lengkap."""
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    soal = pd.read_excel(path, sheet_name="Soal")
    jawaban = pd.read_excel(path, sheet_name="Jawaban")

    n_before = len(jawaban)

    jawaban["Jawaban"] = jawaban["Jawaban"].apply(clean_text)
    jawaban = jawaban.dropna(subset=["Kode Soal", "Jawaban", "Skor (0/5/10)"])
    jawaban = jawaban[jawaban["Skor (0/5/10)"].isin(VALID_SCORES)]  # buang outlier skor (mis. 3.0)
    jawaban["Kode Soal"] = jawaban["Kode Soal"].astype(int)
    jawaban = jawaban.drop_duplicates(subset=["Kode Soal", "Jawaban"])

    df = jawaban.merge(soal, left_on="Kode Soal", right_on="Kode Soal", how="left")
    df = df.rename(columns={"Soal": "pertanyaan", "Jawaban": "jawaban", "Skor (0/5/10)": "skor"})
    df["skor"] = df["skor"].astype(int)
    df["rubrik"] = ""  # placeholder wajib diisi manual sebelum training final

    n_after = len(df)
    logger.info(
        "Baris awal: %d | Baris valid setelah cleaning: %d | Dibuang: %d (%.1f%%)",
        n_before, n_after, n_before - n_after, 100 * (n_before - n_after) / n_before,
    )
    logger.info("Distribusi skor final:\n%s", df["skor"].value_counts().sort_index())
    logger.info("Distribusi soal final:\n%s", df["Kode Soal"].value_counts().sort_index())

    return df.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split stratified pada gabungan Kode Soal + skor agar distribusi tiap subset seimbang."""
    strata = df["Kode Soal"].astype(str) + "_" + df["skor"].astype(str)

    # buang strata dengan <2 anggota (tidak bisa distratifikasi)
    counts = strata.value_counts()
    valid_idx = strata.isin(counts[counts >= 2].index)
    dropped = (~valid_idx).sum()
    if dropped:
        logger.warning("Membuang %d baris dari strata langka (<2 anggota) sebelum split.", dropped)
    df, strata = df[valid_idx].reset_index(drop=True), strata[valid_idx].reset_index(drop=True)

    train_df, temp_df, strata_train, strata_temp = train_test_split(
        df, strata, test_size=val_size + test_size, stratify=strata, random_state=RANDOM_SEED
    )
    # re-check strata temp untuk split kedua
    temp_counts = strata_temp.value_counts()
    valid_temp = strata_temp.isin(temp_counts[temp_counts >= 2].index)
    temp_df, strata_temp = temp_df[valid_temp], strata_temp[valid_temp]

    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=strata_temp, random_state=RANDOM_SEED
    )
    return train_df, val_df, test_df


def to_chat_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Tulis dataset dalam format chat SFT (compatible Llama-3.2 instruct template)."""
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rubrik = row["rubrik"] or "(rubrik belum diisi - lengkapi sebelum training final)"
            user_msg = (
                f"Pertanyaan: {row['pertanyaan']}\n"
                f"Rubrik penilaian: {rubrik}\n"
                f"Jawaban mahasiswa: {row['jawaban']}\n\n"
                "Berikan skor dan alasan dalam format JSON."
            )
            assistant_msg = json.dumps(
                {"skor": int(row["skor"]), "alasan": ""}, ensure_ascii=False
            )
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Ditulis %d baris -> %s", len(df), path)


def main() -> None:
    df = load_and_merge(INPUT_PATH)
    df.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
    logger.info("cleaned_dataset.csv disimpan. ISI KOLOM 'rubrik' SEBELUM TRAINING FINAL.")

    train_df, val_df, test_df = stratified_split(df)
    logger.info(
        "Split -> train: %d | val: %d | test: %d", len(train_df), len(val_df), len(test_df)
    )

    to_chat_jsonl(train_df, OUTPUT_DIR / "train.jsonl")
    to_chat_jsonl(val_df, OUTPUT_DIR / "val.jsonl")
    to_chat_jsonl(test_df, OUTPUT_DIR / "test.jsonl")


if __name__ == "__main__":
    main()
