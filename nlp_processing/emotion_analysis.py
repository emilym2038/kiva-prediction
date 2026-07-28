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



new_df["desc_char_count"] = new_df["desc_en"].str.len()

new_df["desc_word_count"] = new_df["desc_en"].str.split().str.len()



test_df = new_df.copy()

emotion_pipeline = pipeline(
    "text-classification", 
    model="SamLowe/roberta-base-go_emotions"
)

emotion_results = emotion_pipeline(
    test_df["desc_en"].tolist(),
    batch_size=32,
    truncation=True
)

test_df["emotion"] = [r["label"] for r in emotion_results]
test_df["emotion_confidence"] = [round(r["score"], 4) for r in emotion_results]
