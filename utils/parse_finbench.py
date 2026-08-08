import os
import glob
import pickle
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Load environment variables (contains LLAMA_CLOUD_API_KEY)
load_dotenv()

# Verify API key is present
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
if not api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY is not set. Please add it to your .env file.")

def parse_finbench_pdfs(data_dir: str = "data/FinBench", cache_path: str = "data/FinBench/parsed_corpus.pkl") -> list:
    """
    Parses all PDFs in data_dir using LlamaParse and caches the output.
    Returns a list of strings where each element represents a page (containing text + markdown tables).
    
    This matches the format of FinQA_corpus and TatQA_corpus.
    """
    # 1. Check if cached corpus exists to prevent redundant API cost
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

if __name__ == "__main__":
    import time
    start = time.time()
    
    # Run the parsing job
    corpus = parse_finbench_pdfs()
    
    print(f"Successfully processed FinBench corpus.")
    print(f"Total parsed pages (list size): {len(corpus)}")
    print(f"Time taken: {time.time() - start:.2f} seconds")
    
    if corpus:
        print("\n--- SAMPLE PAGE OUTPUT (First 500 characters) ---")
        print(corpus[0][:500])
