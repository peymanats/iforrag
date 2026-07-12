import re
from collections import Counter

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