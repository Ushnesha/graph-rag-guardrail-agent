import json
import os
import glob
import pickle
from llama_parse import LlamaParse

from dotenv import load_dotenv
load_dotenv()  # Load variables from .env file

api_key = os.getenv("LLAMA_CLOUD_API_KEY")

if not api_key:
    raise ValueError("Please set LLAMA_CLOUD_API_KEY in your .env file")
def parse_and_load_globs(data_dir: str = "data/FinBench", cache_path: str = "data/FinBench/parsed_corpus.pkl") -> list:
    if os.path.exists(cache_path):
        print(f"Loading parsed FinBench corpus from cache: {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 2. Find all PDF files
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}.")
        return []

    print(f"Found {len(pdf_files)} PDF files to parse. Initializing LlamaParse...")

    # Initialize LlamaParse in markdown mode
    # LlamaParse visual parsing handles tables and complex layout structures automatically
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",  # Outputs clean text and formatted markdown tables
        verbose=True,
        split_by_page=True       # Keeps each page as a separate document chunk
    )

    corpus = []

    # Parse each PDF file
    for pdf_path in sorted(pdf_files):
        print(f"Parsing file: {pdf_path}...")
        try:
            documents = parser.load_data(pdf_path)
            
            for doc in documents:
                page_text = doc.text.strip()
                if page_text:
                    corpus.append(page_text)
                    
        except Exception as e:
            print(f"Failed to parse {pdf_path}: {e}")

    # 3. Cache the parsed corpus
    print(f"Caching parsed corpus (length: {len(corpus)} pages) to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(corpus, f)

    return corpus


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

# FinQA_data_path = "data/FinQA/train.json"
# FinQA_corpus = load_finqa_corpus(FinQA_data_path)
# print(f"FinQa train data length: {len(FinQA_corpus)}")

# TatQA_data_path = "data/Tat-QA/tatqa_dataset_train.json"
# TatQA_corpus = load_tatqa_corpus(TatQA_data_path)
# print(f"TatQA train data length: {len(TatQA_corpus)}")

# FinBench_corpus = parse_and_load_globs("data/FinBench", "data/FinBench/parsed_corpus.pkl")
# print(f"FinBench corpus length: {len(FinBench_corpus)}")

