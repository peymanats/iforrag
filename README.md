# Pipeline Diagnostics & Optimization Log

## 1. Running the Baseline

After running the baseline, the results show that the baseline does not support unanswerable questions and retrieves the entire document even when only a specific answer is needed. 

### Initial Goals:
* Add support for unanswerable questions by introducing similarity thresholds.
* Improve chunking so the system can return answers that are more specifically relevant to the query.
* Generate an evaluation dataset (created using **GLM-5.2**).

After adding a threshold of `0.5` to the baseline and evaluating, the main issue identified is when a question is unanswerable but the model returns a document anyway, resulting in false positives. As shown in the score distribution plot below, there is no single threshold that can cleanly separate answerable and unanswerable queries.

![score distribution](results/validation/baseline_tresh/score_distribution.png)
![confusion matrix](results/validation/baseline_tresh/confusion_matrix.png)

> **Diagnostic Hypothesis:** The chunk contexts are too general and lack specificity. Consequently, any query that points even slightly toward the general topic receives a high similarity score, even if it is unrelated.

---

## 2. Sentence Chunking Strategy

To address the context specificity issue, we implemented a sentence chunking strategy. During evaluation, we mapped the retrieved sentences back to their parent document IDs (`doc_id`) to match the ground truth.

### Key Results:
* **Recall@1 and Recall@3** improved by **3%**.
* The minimum similarity scores of unanswerable samples increased by approximately `0.1`.

Below are the distribution and confusion matrix plots for this experiment:

![score distribution](results/validation/sentence_chunking/score_distribution.png)
![confusion matrix](results/validation/sentence_chunking/confusion_matrix.png)

---

## 3. Metadata Enrichment: Sentence Chunking with Titles

Simply prefixing the document title to each sentence chunk boosted the similarity scores of the True Positive (TP) samples. With this modification and a threshold of approximately `0.7`, unanswerable questions can be approximately separated from answerable ones.

![score distribution](results/validation/sentences_chunking_title/score_distribution.png)
![confusion matrix](results/validation/sentences_chunking_title/confusion_matrix.png)

---

## 4. Term-Based Search (BM25) & Reranking Trials

The term-based BM25 search algorithm performed well on our data for answerable questions. Because the **Recall@3 of BM25 reached 100%**, we can be confident that the correct document is within the top-5 results.

### Latency Profiling:
To monitor operational efficiency across different retrieval setups, we integrated execution time tracking to measure both:
* **Index Construction Speed (`build_time`)**
* **Query Execution/Retrieval Latency**

### Reranking Implementation:
We implemented a two-stage reranker:
1. Retrieve the top 8 candidate documents using BM25.
2. Rerank the sentence chunks within those 8 documents using cosine similarity of their embeddings.

**Result:** No significant performance improvement was observed. 

> **Design Decision:** Moving to hybrid search methods like Reciprocal Rank Fusion (RRF) may not be the best choice here, as they have little impact on filtering out unanswerable questions (though they might provide minor improvements on error-code queries). Instead, we prefer focusing on improving context using metadata and semantic search alone.

---

## 5. Duplicate and Conflict Detection

We created a diagnostic notebook, `near_duplicate.ipynb`, to detect similar sentences within the corpus to catch potential duplicates and knowledge conflicts:

* **Knowledge Conflicts:** Identified between `DOC-02_s_2` and `DOC-01_s_2`.
* **Knowledge Duplication:** Identified between `DOC-05_s_0` and `DOC-06_s_0`.

---

## 6. Sliding Window Chunking Experiment

We implemented sliding window chunking and tested it under both pure semantic search and the hybrid reranking setup.

### Findings:
* For pure semantic search, sentence-level chunking still yields better overall separation.
* For the reranking pipeline, the sliding window approach performs better on answerable queries.

![score distribution](results/validation/reranker_sliding_100_50/score_distribution.png)
![confusion matrix](results/validation/reranker_sliding_100_50/confusion_matrix.png)

---

## 7. Threshold Calibration & Generalizability Validation

After testing different thresholds and sliding window parameters, the semantic retriever alone configured with a threshold of `0.65` achieved **85% accuracy** on our validation dataset. 

To test the generalizability of this model, we evaluated it against an unseen test dataset of 100 samples.

### Generalizability Results:
* **Accuracy on unseen test samples:** Dropped to **65%**.
* **Conclusion:** On unseen data, the semantic retriever with a static `0.65` threshold struggles to make correct decisions. This demonstrates that a single fixed threshold is highly vulnerable to distribution shifts and cannot cleanly separate unanswerable questions on unseen data.
