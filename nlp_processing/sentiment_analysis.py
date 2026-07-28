
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import pipeline

df = pd.read_csv("borrower_info.csv")
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

results = sentiment_pipeline(
    test_df["desc_en"].tolist(),
    batch_size=32,
    truncation=True
)

test_df["sentiment"] = [r["label"] for r in results]
test_df["confidence"] = [round(r["score"], 4) for r in results]




