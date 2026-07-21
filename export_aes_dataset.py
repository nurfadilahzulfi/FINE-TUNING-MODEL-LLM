"""
Management command untuk export data AES dari database production ke JSONL
siap fine-tuning, menggantikan alur export Excel manual.

Lokasi file: apps/submissions/management/commands/export_aes_dataset.py
(buat folder management/commands/ dengan __init__.py kosong di masing-masing
level kalau belum ada -- struktur wajib Django management command)

Cara pakai:
    python manage.py export_aes_dataset --output ./aes_export --only-verified

Opsi:
    --output          folder output (default: ./aes_export)
    --only-verified   HANYA include baris yang sudah dikoreksi/divalidasi dosen.
                       Rekomendasi: pakai flag ini begitu field verifikasi
                       ditambahkan ke model Jawaban. Tanpa flag ini, semua
                       baris grading_status='done' ikut ter-export apa adanya
                       (termasuk yang murni hasil AI belum divalidasi).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from django.db.models import QuerySet
from sklearn.model_selection import train_test_split

from apps.exams.models import Soal
from apps.submissions.models import Jawaban

logger = logging.getLogger(__name__)

VALID_SCORES = {0, 5, 10}
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "Anda adalah asisten penilai esai untuk mata kuliah Machine Learning. "
    "Nilai jawaban mahasiswa HANYA berdasarkan rubrik yang diberikan. "
    "Berikan skor (0, 5, atau 10) dan alasan singkat dalam Bahasa Indonesia. "
    "Format jawaban WAJIB JSON: {\"skor\": <0|5|10>, \"alasan\": \"...\"}"
)


class Command(BaseCommand):
    help = "Export data Jawaban (soal + jawaban + nilai) ke JSONL untuk fine-tuning AES."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--output", type=str, default="./aes_export", help="Folder output JSONL."
        )
        parser.add_argument(
            "--only-verified",
            action="store_true",
            help="Hanya ambil baris yang sudah diverifikasi dosen (butuh field verifikasi di model Jawaban).",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        output_dir = Path(options["output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        only_verified: bool = options["only_verified"]

        queryset = self._build_queryset(only_verified)
        records = self._queryset_to_records(queryset)

        if not records:
            self.stderr.write(self.style.ERROR("Tidak ada data valid untuk diexport."))
            return

        self.stdout.write(f"Total baris valid: {len(records)}")

        train, val, test = self._stratified_split(records)
        self.stdout.write(
            f"Split -> train: {len(train)} | val: {len(val)} | test: {len(test)}"
        )

        self._write_jsonl(train, output_dir / "train.jsonl")
        self._write_jsonl(val, output_dir / "val.jsonl")
        self._write_jsonl(test, output_dir / "test.jsonl")

        self.stdout.write(self.style.SUCCESS(f"Selesai. File tersimpan di {output_dir}"))

    def _build_queryset(self, only_verified: bool) -> QuerySet[Jawaban]:
        """Ambil jawaban yang sudah selesai dinilai, dengan relasi soal ter-prefetch."""
        qs = Jawaban.objects.filter(
            grading_status="done", nilai__isnull=False
        ).select_related("soal")

        if only_verified:
            # NOTE: sesuaikan nama field ini setelah kamu tambahkan mekanisme
            # verifikasi dosen ke model Jawaban, misalnya:
            #   qs = qs.filter(diverifikasi_dosen=True)
            logger.warning(
                "--only-verified diminta tapi field verifikasi belum ada di model Jawaban. "
                "Tambahkan field tsb dulu, lalu update filter ini."
            )

        return qs.exclude(teks_jawaban="").exclude(teks_jawaban__isnull=True)

    def _queryset_to_records(self, queryset: QuerySet[Jawaban]) -> list[dict[str, Any]]:
        """Konversi queryset jadi list dict bersih, buang skor di luar skala valid."""
        records: list[dict[str, Any]] = []
        skipped = 0

        for jawaban in queryset.iterator():
            if jawaban.nilai not in VALID_SCORES:
                skipped += 1
                continue

            soal: Soal = jawaban.soal
            records.append(
                {
                    "soal_id": soal.id,
                    "pertanyaan": soal.pertanyaan.strip(),
                    "referensi_jawaban": (soal.referensi_jawaban or "").strip(),
                    "kata_kunci": (soal.kata_kunci or "").strip(),
                    "jawaban": jawaban.teks_jawaban.strip(),
                    "nilai": int(jawaban.nilai),
                    "alasan": (jawaban.alasan_nilai or "").strip(),
                }
            )

        if skipped:
            logger.warning("Melewati %d baris dengan nilai di luar skala valid %s.", skipped, VALID_SCORES)

        return records

    def _stratified_split(
        self, records: list[dict[str, Any]], val_size: float = 0.15, test_size: float = 0.15
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Split stratified per (soal_id x nilai), sama seperti pipeline Excel sebelumnya."""
        strata = [f"{r['soal_id']}_{r['nilai']}" for r in records]

        from collections import Counter

        counts = Counter(strata)
        filtered = [(r, s) for r, s in zip(records, strata) if counts[s] >= 2]
        dropped = len(records) - len(filtered)
        if dropped:
            logger.warning("Membuang %d baris dari strata langka (<2 anggota).", dropped)

        recs = [r for r, _ in filtered]
        strat_labels = [s for _, s in filtered]

        train, temp, strat_train, strat_temp = train_test_split(
            recs, strat_labels, test_size=val_size + test_size,
            stratify=strat_labels, random_state=RANDOM_SEED,
        )

        temp_counts = Counter(strat_temp)
        temp_filtered = [(r, s) for r, s in zip(temp, strat_temp) if temp_counts[s] >= 2]
        temp_recs = [r for r, _ in temp_filtered]
        temp_strat = [s for _, s in temp_filtered]

        rel_test = test_size / (val_size + test_size)
        val, test = train_test_split(
            temp_recs, test_size=rel_test, stratify=temp_strat, random_state=RANDOM_SEED
        )
        return train, val, test

    def _write_jsonl(self, records: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                rubrik = r["referensi_jawaban"] or "(rubrik belum diisi)"
                if r["kata_kunci"]:
                    rubrik += f"\nKata kunci: {r['kata_kunci']}"

                user_msg = (
                    f"Pertanyaan: {r['pertanyaan']}\n"
                    f"Rubrik penilaian: {rubrik}\n"
                    f"Jawaban mahasiswa: {r['jawaban']}\n\n"
                    "Berikan skor dan alasan dalam format JSON."
                )
                assistant_msg = json.dumps(
                    {"skor": r["nilai"], "alasan": r["alasan"]}, ensure_ascii=False
                )
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Ditulis %d baris -> %s", len(records), path)
