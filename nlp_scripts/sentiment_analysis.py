
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import pipeline

OUTPUT_PATH = "sentiment_analysis.csv"
CHUNK_SIZE = 1000

df = pd.read_csv("/project/rise-phishing/kiva-prediction/cleaned_csv/borrower_info.csv")
new_df = pd.DataFrame()
new_df["loan_id"] = df["loan_id"]
new_df["desc_en"] = df["desc_en"]


new_df["desc_en"] = new_df["desc_en"].fillna('')
new_df["desc_en"] = new_df["desc_en"].astype(str)



test_df = new_df.copy()

sentiment_pipeline = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis",
    device=0 if torch.cuda.is_available() else -1
)

# Resume support: skip rows already scored in a previous (interrupted) run.
start_idx = 0
if os.path.exists(OUTPUT_PATH):
    done_df = pd.read_csv(OUTPUT_PATH)
    start_idx = len(done_df)
    print(f"Resuming from row {start_idx} (found existing {OUTPUT_PATH})")

texts = test_df["desc_en"].tolist()
write_header = start_idx == 0

for chunk_start in tqdm(range(start_idx, len(texts), CHUNK_SIZE), desc="Scoring sentiment"):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(texts))
    chunk_texts = texts[chunk_start:chunk_end]

    results = sentiment_pipeline(chunk_texts, batch_size=32, truncation=True)

    chunk_df = test_df.iloc[chunk_start:chunk_end].copy()
    chunk_df["sentiment"] = [r["label"] for r in results]
    chunk_df["confidence"] = [round(r["score"], 4) for r in results]

    chunk_df.to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
    write_header = False



