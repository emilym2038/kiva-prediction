import os
import pandas as pd
from tqdm import tqdm
import textstat

OUTPUT_PATH = "readability.csv"
CHUNK_SIZE = 5000

df = pd.read_csv("/project/rise-phishing/kiva-prediction/cleaned_csv/borrower_info.csv")
new_df = pd.DataFrame()
new_df["loan_id"] = df["loan_id"]
new_df["desc_en"] = df["desc_en"]

new_df["desc_en"] = new_df["desc_en"].fillna('')
new_df["desc_en"] = new_df["desc_en"].astype(str)

test_df = new_df.copy()

start_idx = 0
if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
    done_df = pd.read_csv(OUTPUT_PATH)
    start_idx = len(done_df)
    print(f"Resuming from row {start_idx} (found existing {OUTPUT_PATH})")

texts = test_df["desc_en"].tolist()
write_header = start_idx == 0

output_cols = ["loan_id"]

for chunk_start in tqdm(range(start_idx, len(texts), CHUNK_SIZE), desc="Scoring readability"):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(texts))
    chunk_texts = texts[chunk_start:chunk_end]

    fk_grade = [textstat.flesch_kincaid_grade(t) if t.strip() else None for t in chunk_texts]
    reading_ease = [textstat.flesch_reading_ease(t) if t.strip() else None for t in chunk_texts]

    chunk_df = test_df.iloc[chunk_start:chunk_end][output_cols].copy()
    chunk_df["flesch_kincaid_grade"] = fk_grade
    chunk_df["flesch_reading_ease"] = reading_ease

    chunk_df.to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
    write_header = False
