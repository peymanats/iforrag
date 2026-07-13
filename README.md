# WELCOME
the documentation is availabe in **github wiki**

NOTE: the main sections are **Diagnostic-Log** that I wrote during the process of implmentation and research and AI just rewrite it to better format and **Benchmark Results** that i analysis the results and decisions then explain for AI and it rewrite (gemini flash).

## Step A: Install Dependencies

Install the required Python packages.

The local dense retriever and reranker use **sentence-transformers**, while the automated PDF reporting pipeline uses **reportlab**.

```bash
pip install numpy sentence-transformers reportlab
```

> **Note:** No GPU is required. Embedding computations automatically run efficiently on the CPU.

---

## Step B: Running the Unified Benchmark Suite (`evall_all.py`)

To evaluate all **22 configuration settings** covering:

- BM25 retrieval
- Dense retrieval
- Hybrid reranker
- Rule-based router
- Different thresholds
- Different chunking strategies

run the master evaluation script:

```bash
python src/evall_all.py
```

---

## Generated Artifacts

After completion, the script produces:

### 1. `benchmark_results.json`

Contains:

- Raw evaluation metadata
- Runtime information
- Latency measurements
- Evaluation metrics

---

### 2. `benchmark_results.csv`

A spreadsheet-compatible export of the complete benchmark results.

---

### 3. `benchmark_report.pdf`

A landscape-oriented PDF report containing:

- Metric summaries
- System configurations
- Benchmark comparisons

---
