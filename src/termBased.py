import re
from collections import Counter


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
    def __init__(self):
        self.inverted_index = InvertedIndex()
        self.doc_lengths = {}
        self.documents = {}

    def tokenize(self, text):
        text = text.lower()

        # Normalize title separators
        text = text.replace("—", " ")
        text = text.replace("–", " ")

        # "Lockout / Tagout" -> "lockout tagout"
        text = re.sub(r"\s+/\s+", " ", text)

        # Keep:
        #   p-200
        #   e-207
        #   brg-4410
        #   m3/h
        #   mm/s
        #   4.5
        pattern = r"[a-z]+\d*(?:[-/][a-z0-9]+)*|\d+(?:\.\d+)?"

        return re.findall(pattern, text)

    def build(self, corpus):
        for doc in corpus:
            doc_id = doc["id"]

            # Index title + body
            document = f"{doc['title']} {doc['text']}"

            tokens = self.tokenize(document)

            self.documents[doc_id] = doc
            self.doc_lengths[doc_id] = len(tokens)

            tf = Counter(tokens)

            for term, freq in tf.items():
                self.inverted_index.add(term, doc_id, freq)

        return self.inverted_index