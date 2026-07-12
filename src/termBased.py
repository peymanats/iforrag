import re
from collections import Counter
import numpy as np
import json
import time
# from evaluate import CORPUS_PATH, load_corpus
CORPUS_PATH = "data/raw/corpus.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"

class Tokenizer:
    def tokenize(self, text):
        text = text.lower()
        text = text.replace("—", " ")
        text = text.replace("–", " ")
        text = re.sub(r"\s+/\s+", " ", text)

        pattern = r"[a-z]+\d*(?:[-/][a-z0-9]+)*|\d+(?:\.\d+)?"
        return re.findall(pattern, text)
    
class InvertedIndex:
    def __init__(self):
        # term -> {doc_id: term_frequency}
        self.index = {}

    def add(self, term, doc_id, tf):
        self.index.setdefault(term, {})[doc_id] = tf

    def __contains__(self, term):
        return term in self.index

    def __getitem__(self, term):
        return self.index[term]


class IndexBuilder:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.inverted_index = InvertedIndex()
        self.doc_lengths = {}
        self.documents = {}
        self.avgdl = 0.0

    def build(self, corpus):
        for doc in corpus:
            doc_id = doc["id"]

            # Index title + body
            document = f"{doc['title']} {doc['text']}"
            tokens = self.tokenizer.tokenize(document)

            self.documents[doc_id] = doc
            self.doc_lengths[doc_id] = len(tokens)

            tf = Counter(tokens)

            for term, freq in tf.items():
                self.inverted_index.add(term, doc_id, freq)

        if self.doc_lengths:
            self.avgdl = sum(self.doc_lengths.values()) / len(self.doc_lengths)

        return self.inverted_index

def split_sentences_punctuation(text):
    """
    Splits text into sentences using punctuation marks.
    
    Parameters:
    text (str): The input text to be split.
    
    Returns:
    list: A list of sentences.
    """
    # Regular expression to split sentences based on punctuation marks
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

def sliding_windows(text, window_size=100, step_size=50):
    """
    Splits text into overlapping windows of a specified size.
    
    Parameters:
    text (str): The input text to be split.
    window_size (int): The size of each window.
    step_size (int): The step size for sliding the window.
    
    Returns:
    list: A list of text windows.
    """
    windows = []
    for start in range(0, len(text), step_size):
        end = start + window_size
        windows.append(text[start:end])
        if end >= len(text):
            break
    return windows

def load_corpus(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs

def build_baseline_system(corpus_path,sliding_window=False,window_size=100,step_size=50):
    """Build the baseline index and return a retrieve(query, top_k) function."""
    from sentence_transformers import SentenceTransformer

    docs = load_corpus(corpus_path)
    model = SentenceTransformer(EMBED_MODEL)

    # Build chunks (same as baseline)
    chunks = []
    start = time.time()
    for d in docs:
        if sliding_window:
            windows = sliding_windows(d["text"], window_size, step_size)
            for i, c in enumerate(windows):
                chunks.append({"doc_id": d["id"], "sent_id": d["id"] + f"_w_{i}", "title": d["title"], "text": f"Title: {d['title']}\n{c}"})
        else:
            for sent_num, c in enumerate(split_sentences_punctuation(d["text"])):
                chunks.append({"doc_id": d["id"], "sent_id": d["id"] + f"_s_{sent_num}", "title": d["title"], "text": f"Title: {d['title']}\n{c}"})
    
    # Encode and normalize (same as baseline)
    vectors = model.encode([c["text"] for c in chunks])
    end = time.time()
    build_time = (end -start)/len(docs)
    vectors = np.asarray(vectors, dtype="float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    # print(chunks)
    def reranker(query, bm25, top_k=8):
        """Return list of (doc_id, score, text) ranked by descending score,
        deduplicated by doc_id."""
        # print(query)
        retrieved_docs = bm25(query)
        q = model.encode([query])[0].astype("float32")
        q = q / np.linalg.norm(q)
        retrieved_doc_ids = {doc_id for doc_id, score in retrieved_docs}
        candidate_indices = [
            idx for idx, chunk in enumerate(chunks)
            if chunk["doc_id"] in retrieved_doc_ids
        ]        
        # print(f"Candidate indices for query '{query}': {candidate_indices}")
        filtered_chunks = [chunks[idx] for idx in candidate_indices]
        sims = vectors[candidate_indices] @ q

        # Sort all chunks by score descending
        order = np.argsort(-sims)

        # Deduplicate by doc_id — keep only the highest-scoring chunk per doc
        seen_docs = set()
        results = []
        for idx in order:
            doc_id = filtered_chunks[idx]["doc_id"]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            results.append((doc_id, float(sims[idx]), filtered_chunks[idx]["text"]))
            if len(results) >= top_k:
                break
        return results

    def retrieve(query, top_k=5):
        """Return list of (doc_id, score, text) ranked by descending score,
        deduplicated by doc_id."""
        q = model.encode([query])[0].astype("float32")
        q = q / np.linalg.norm(q)
        sims = vectors @ q

        # Sort all chunks by score descending
        order = np.argsort(-sims)

        # Deduplicate by doc_id — keep only the highest-scoring chunk per doc
        seen_docs = set()
        results = []
        for idx in order:
            doc_id = chunks[idx]["doc_id"].split("_s_")[0]  # remove sentence suffix for deduplication
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            results.append((doc_id, float(sims[idx]), chunks[idx]["text"]))
            if len(results) >= top_k:
                break
        return results
    return reranker ,retrieve, build_time


def compute_idf(inverted_index, total_docs):
    idf = {}
    for term, postings in inverted_index.index.items():
        df = len(postings)
        idf[term] = np.log((total_docs - df + 0.5) / (df + 0.5) + 1)
    return idf

def compute_bm25_score(query, index_builder,inverted_index, idf, doc_lengths, avgdl, k1=1.5, b=0.75,tokenizer=None,top_k=8):
    query_terms = tokenizer.tokenize(query) if tokenizer else query
    scores={}
    # score = 0.0
    for doc_id in index_builder.documents.keys():
        doc_length = doc_lengths[doc_id]
        score = 0.0
        for term in query_terms:
            if term in inverted_index:
                postings = inverted_index[term]
                if doc_id in postings:
                    tf = postings[doc_id]
                    idf_term = idf[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))
                    score += idf_term * (numerator / denominator)
        scores[doc_id] = score
    top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return top_k


if __name__ == "__main__":
    # Example usage
    tokenizer = Tokenizer()
    index_builder = IndexBuilder(tokenizer)

    # Load your corpus here (list of documents)
    corpus = load_corpus(CORPUS_PATH)

    inverted_index = index_builder.build(corpus)
    total_docs = len(corpus)
    idf = compute_idf(inverted_index, total_docs)
    reranker ,_,_ = build_baseline_system(CORPUS_PATH)
    # Example query
    query = "What is the rated output of the C-100 compressor?"
    query_terms = tokenizer.tokenize(query)
    # Compute BM25 scores for each document
    bm25 = lambda q: compute_bm25_score(q, index_builder, inverted_index, idf, index_builder.doc_lengths, index_builder.avgdl, tokenizer=tokenizer, top_k=5)
    reranker_results = reranker(query, bm25, top_k=8)
    print(f"Reranker results for query '{query}':", reranker_results)
    # print(f"BM25 scores for query '{query}':", scores)