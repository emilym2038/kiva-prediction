import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import pipeline

OUTPUT_PATH = "emotion_analysis.csv"
CHUNK_SIZE = 1000

df = pd.read_csv("/project/rise-phishing/kiva-prediction/cleaned_csv/borrower_info.csv")
new_df = pd.DataFrame()
new_df["loan_id"] = df["loan_id"]
new_df["desc_en"] = df["desc_en"]


new_df["desc_en"] = new_df["desc_en"].fillna('')
new_df["desc_en"] = new_df["desc_en"].astype(str)



new_df["desc_char_count"] = new_df["desc_en"].str.len()

new_df["desc_word_count"] = new_df["desc_en"].str.split().str.len()



test_df = new_df.copy()


def pick_device():
    """Use the GPU only if this torch build can actually run a kernel on it.
    A GPU's exact arch (e.g. sm_89) may be missing from get_arch_list() while
    still working fine via PTX JIT from a nearby compiled arch -- and
    conversely (e.g. P100/sm_60) it may be visible via CUDA but truly
    unsupported. A real smoke-test op is the only reliable way to tell."""
    if not torch.cuda.is_available():
        return -1
    try:
        (torch.tensor([1.0], device="cuda") + 1).cpu()
        return 0
    except RuntimeError as e:
        major, minor = torch.cuda.get_device_capability(0)
        print(f"GPU sm_{major}{minor} present but unusable with installed torch "
              f"build ({e}) -- falling back to CPU.")
        return -1


device = pick_device()
batch_size = 64 if device == 0 else 32
print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}, batch_size={batch_size}")

emotion_pipeline = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    device=device
)

# Resume support: skip rows already scored in a previous (interrupted) run.
start_idx = 0
if os.path.exists(OUTPUT_PATH):
    done_df = pd.read_csv(OUTPUT_PATH)
    start_idx = len(done_df)
    print(f"Resuming from row {start_idx} (found existing {OUTPUT_PATH})")

texts = test_df["desc_en"].tolist()
write_header = start_idx == 0

# desc_en (and loan_id/desc_en duplicates) already live in sentiment_analysis.csv --
# only keep columns unique to this script, joinable back on loan_id.
output_cols = ["loan_id", "desc_char_count", "desc_word_count"]

for chunk_start in tqdm(range(start_idx, len(texts), CHUNK_SIZE), desc="Scoring emotion"):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(texts))
    chunk_texts = texts[chunk_start:chunk_end]

    results = emotion_pipeline(chunk_texts, batch_size=batch_size, truncation=True)

    chunk_df = test_df.iloc[chunk_start:chunk_end][output_cols].copy()
    chunk_df["emotion"] = [r["label"] for r in results]
    chunk_df["emotion_confidence"] = [round(r["score"], 4) for r in results]

    chunk_df.to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
    write_header = False