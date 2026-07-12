import re
from collections import Counter
import numpy as np
import json

from evaluate import CORPUS_PATH, load_corpus

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


def compute_idf(inverted_index, total_docs):
    idf = {}
    for term, postings in inverted_index.index.items():
        df = len(postings)
        idf[term] = np.log((total_docs - df + 0.5) / (df + 0.5) + 1)
    return idf

def compute_bm25_score(query_terms, doc_id, inverted_index, idf, doc_lengths, avgdl, k1=1.5, b=0.75):
    score = 0.0
    doc_length = doc_lengths[doc_id]

    for term in query_terms:
        if term in inverted_index:
            postings = inverted_index[term]
            if doc_id in postings:
                tf = postings[doc_id]
                idf_term = idf[term]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))
                score += idf_term * (numerator / denominator)

    return score

if __name__ == "__main__":
    # Example usage
    tokenizer = Tokenizer()
    index_builder = IndexBuilder(tokenizer)

    # Load your corpus here (list of documents)
    corpus = load_corpus(CORPUS_PATH)

    inverted_index = index_builder.build(corpus)
    total_docs = len(corpus)
    idf = compute_idf(inverted_index, total_docs)

    # Example query
    query = "What is the rated output of the C-100 compressor?"
    query_terms = tokenizer.tokenize(query)

    # Compute BM25 scores for each document
    scores = {}
    for doc_id in index_builder.documents.keys():
        score = compute_bm25_score(query_terms, doc_id, inverted_index, idf, index_builder.doc_lengths, index_builder.avgdl)
        scores[doc_id] = score

    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    for doc_id, score in top3:
        print(f"Doc {doc_id}: {score:.4f}")