import os
import pandas as pd
from tqdm import tqdm
import spacy

OUTPUT_PATH = "entity_analysis.csv"
CHUNK_SIZE = 5000
N_PROCESS = int(os.environ.get("NSLOTS", "4"))

df = pd.read_csv("/project/rise-phishing/kiva-prediction/cleaned_csv/borrower_info.csv")
new_df = pd.DataFrame()
new_df["loan_id"] = df["loan_id"]
new_df["desc_en"] = df["desc_en"]

new_df["desc_en"] = new_df["desc_en"].fillna('')
new_df["desc_en"] = new_df["desc_en"].astype(str)

test_df = new_df.copy()

# Only the NER component (and its tok2vec dependency) is needed -- named
# entity count as a proxy for concreteness -- so drop the rest for speed.
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

# Resume support: skip rows already scored in a previous (interrupted) run.
start_idx = 0
if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
    done_df = pd.read_csv(OUTPUT_PATH)
    start_idx = len(done_df)
    print(f"Resuming from row {start_idx} (found existing {OUTPUT_PATH})")

texts = test_df["desc_en"].tolist()[start_idx:]
loan_ids = test_df["loan_id"].tolist()[start_idx:]
write_header = start_idx == 0

print(f"Using n_process={N_PROCESS}")

buffer = []
docs = nlp.pipe(texts, batch_size=64, n_process=N_PROCESS)
for loan_id, doc in tqdm(zip(loan_ids, docs), total=len(texts), desc="Counting entities"):
    buffer.append((loan_id, len(doc.ents)))

    if len(buffer) >= CHUNK_SIZE:
        chunk_df = pd.DataFrame(buffer, columns=["loan_id", "n_named_entities"])
        chunk_df.to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
        write_header = False
        buffer = []

if buffer:
    chunk_df = pd.DataFrame(buffer, columns=["loan_id", "n_named_entities"])
    chunk_df.to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
