# phase1_hybrid.py
import os
import hashlib
import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from load_data import FinQA_corpus, TatQA_corpus
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Define the Raw Corpus
# CORPUS = [
#     "Project Alpha is our primary cloud migration effort, managed by Sarah.",
#     "Sarah is the VP of Infrastructure and likes drinking green tea.",
#     "We have a strict $50,000 budget ceiling for cloud infrastructure operations.",
#     "The marketing team is preparing a campaign for Project Alpha launch."
# ]

CORPUS = FinQA_corpus

class HybridSearchEngine:
    def __init__(self, documents: list):
        self.documents = documents

        # Initialize Local Ollama Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device':'mps',
            }
        )
        
        # Initialize Qdrant
        self.qdrant = QdrantClient(path="./qdrant_db")
        self.collection_name = "local_chunks_minilm"
        
        # Generate a hash representing the current corpus content
        corpus_str = "".join(sorted(self.documents))
        current_hash = hashlib.md5(corpus_str.encode("utf-8")).hexdigest()

        
        self.recreate_needed = True
        if self.qdrant.collection_exists(self.collection_name):
            try:
                # Retrieve the special metadata point holding the corpus hash (ID: 999999)
                results = self.qdrant.retrieve(
                    collection_name=self.collection_name,
                    ids=[999999]
                )
                if results and results[0].payload.get("corpus_hash") == current_hash:
                    self.recreate_needed = False
            except Exception:
                pass
        
        if self.recreate_needed:
            # Delete and rebuild the collection if the corpus has changed
            print("Corpus has changed, recreating Qdrant collection...")
            if self.qdrant.collection_exists(self.collection_name):
                self.qdrant.delete_collection(self.collection_name)
            
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE) # miniLM vectors are 384 dim
            )
            self._initialize_qdrant(current_hash)

        # Initialize BM25 Index
        bm25_path = "./qdrant_db/bm25_index.pkl"
        if self.recreate_needed or not os.path.exists(bm25_path):
            print("Building and saving BM25 index...")
            self.tokenized_corpus = [doc.lower().split(" ") for doc in documents]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            # Save BM25 index to disk
            with open(bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)
        else:
            print("Loading BM25 index from cache...")
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)


        
        

    # called only when there are no documents or when the contents of the corpus documents change
    def _initialize_qdrant(self, corpus_hash: str):
        print("Initializing Qdrant collection...")
        # convert all documents into vectors
        vectors = self.embeddings.embed_documents(self.documents)

        # create points with vectors and documents
        points = [
            PointStruct(
                id=idx,
                vector=vector,
                payload={"text": text}
            )
            for idx, (text, vector) in enumerate(zip(self.documents, vectors))
        ]
        # Append a special control point containing the corpus hash
        points.append(
            PointStruct(
                id=999999,
                vector=[0.0] * len(vectors[0]),  # dummy vector matching nomic dimension
                payload={"corpus_hash": corpus_hash}
            )
        )

        # implementing batching for large vector datasets
        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=batch
            )

    def search(self, query: str, k: int = 60, limit: int = 2) -> list:
        # A. Vector Search
        query_vector = self.embeddings.embed_query(query)
        v_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=10
        ).points
        vector_ranks = {
            hit.payload["text"]: index
            for index, hit in enumerate(v_results)
            if hit.id != 999999 and hit.payload and "text" in hit.payload
        }

        # B. BM25 Search
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        ranked_bm25 = sorted(zip(self.documents, bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_ranks = {doc: index for index, (doc, score) in enumerate(ranked_bm25) if score > 0}

        # C. Reciprocal Rank Fusion (RRF) Calculation
        rrf_scores = {}
        all_candidates = set(list(vector_ranks.keys()) + list(bm25_ranks.keys()))
        for doc in all_candidates:
            score = 0.0
            if doc in vector_ranks:
                score += 1.0 / (k + vector_ranks[doc] + 1)
            if doc in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[doc] + 1)
            rrf_scores[doc] = score

        # Sort based on consolidated scores
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in sorted_docs[:limit]]

    def close(self):
        self.qdrant.close()

if __name__ == "__main__":
    from time import time
    initialization_start_time = time()
    engine = HybridSearchEngine(CORPUS)
    print("Hybrid Search Engine Initialized")
    initialization_end_time = time()
    print(f"Time taken for Hybrid Search Engine Initialization: {(initialization_end_time - initialization_start_time) * 1000} ms")
    query_srch_start_time = time()
    user_query = "what is the the interest expense in 2009?"
    refined_results = engine.search(user_query)
    query_srch_end_time = time()
    print(f"Time taken for Query Search: {(query_srch_end_time - query_srch_start_time) * 1000} ms")
    engine.close()
    print("--- REFINED SEARCH RESULTS (RRF) ---")
    print(f"Query: {user_query}")
    for idx, result in enumerate(refined_results):
        print(f"[{idx + 1}] {result}")