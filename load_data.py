import json
def load_finqa_corpus(json_path: str) -> list:
    """Loads FinQA JSON and returns a list of reconstructed text documents."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    corpus = []
    for item in data:
        # Convert table 2D list to a markdown-like string
        table_rows = [" | ".join(map(str, row)) for row in item["table"]]
        table_str = "\n".join(table_rows)
        
        # Combine pre-text, table, and post-text into a single document string
        doc_text = f"{item['pre_text']}\n\n{table_str}\n\n{item['post_text']}"
        corpus.append(doc_text)
        
    return corpus
def load_tatqa_corpus(json_path: str) -> list:
    """Loads TAT-QA JSON and returns a list of reconstructed text documents."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    corpus = []
    for item in data:
        # Extract paragraphs text
        paragraphs_text = "\n\n".join([p["text"] for p in item["paragraphs"]])
        
        # Convert table 2D list to markdown-like string
        table_rows = [" | ".join(map(str, row)) for row in item["table"]]
        table_str = "\n".join(table_rows)
        
        # Combine paragraphs and table
        doc_text = f"{paragraphs_text}\n\n{table_str}"
        corpus.append(doc_text)
        
    return corpus

FinQA_data_path = "data/FinQA/train.json"
FinQA_corpus = load_finqa_corpus(FinQA_data_path)
print(f"FinQa train data length: {len(FinQA_corpus)}")

TatQA_data_path = "data/Tat-QA/tatqa_dataset_train.json"
TatQA_corpus = load_tatqa_corpus(TatQA_data_path)
print(f"TatQA train data length: {len(TatQA_corpus)}")

