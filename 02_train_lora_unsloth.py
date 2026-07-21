"""
Fine-tuning Llama-3.2-3B-Instruct dengan QLoRA (Unsloth) untuk Automated Essay Scoring.

JALANKAN DI MESIN BER-GPU (min. 8GB VRAM, mis. RTX 3060/T4 Colab/Kaggle).
Tidak jalan di CPU-only environment ini -- ini SKELETON siap pakai untuk
dijalankan di lingkungan training kamu.

Install (sekali saja, di mesin GPU):
    pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes

Referensi: https://github.com/unslothai/unsloth (cek versi CUDA/torch yang cocok
dengan environment kamu di README mereka -- string ekstra di atas berubah-ubah).
"""

from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---- Hyperparameters (dokumentasikan semua untuk reproducibility) ----
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"  # quantized, hemat VRAM
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0  # 0 = optimized path di Unsloth
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3          # dataset kecil (~300 baris) -> awasi overfitting, cek val loss tiap epoch
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4          # effective batch size = 8
SEED = 42

DATA_DIR = Path("./data")     # hasil dari 01_prepare_dataset.py
OUTPUT_DIR = Path("./outputs/llama32_3b_aes_lora")
GGUF_OUTPUT_DIR = Path("./outputs/llama32_3b_aes_gguf")


def main() -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    logger.info("Loading base model: %s", BASE_MODEL)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {DATA_DIR}. Jalankan 01_prepare_dataset.py dulu."
        )

    dataset = load_dataset(
        "json", data_files={"train": str(train_path), "validation": str(val_path)}
    )

    def format_chat(example: dict) -> dict:
        # apply_chat_template menyusun prompt sesuai template resmi Llama-3.2-Instruct
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat, remove_columns=dataset["train"].column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            eval_strategy="epoch",       # pantau val loss tiap epoch -- stop kalau naik (overfit)
            save_strategy="epoch",
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=SEED,
            report_to="none",
        ),
    )

    logger.info("Mulai training...")
    trainer.train()

    logger.info("Menyimpan adapter LoRA + tokenizer ke %s", OUTPUT_DIR)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Ekspor langsung ke GGUF (quantized q4_k_m) supaya bisa langsung diimport Ollama
    logger.info("Mengekspor ke GGUF (q4_k_m) di %s", GGUF_OUTPUT_DIR)
    model.save_pretrained_gguf(
        str(GGUF_OUTPUT_DIR), tokenizer, quantization_method="q4_k_m"
    )
    logger.info("Selesai. File .gguf ada di %s -- lanjut ke Modelfile Ollama.", GGUF_OUTPUT_DIR)


if __name__ == "__main__":
    main()
