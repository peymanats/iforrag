#!/usr/bin/env python3
"""
evaluate.py — Evaluation harness for the RAG retrieval pipeline.

Runs a set of evaluation questions (eval/eval.jsonl) through a retrieval
system and reports metrics:

  Answerable questions:
    - Hit-Rate@1   (is the expected doc at rank 1?)
    - Hit-Rate@3   (is the expected doc in top-3, deduped by doc_id?)
    - MRR          (mean reciprocal rank of first correct doc)
  Unanswerable questions:
    - Abstention rate  (fraction where the system says "not found")
  All:
    - Avg top-1 score for answerable vs unanswerable (threshold calibration)
  Per-category breakdown.

Usage:
    iforrag/Scripts/python.exe src/evaluate.py --system baseline
    iforrag/Scripts/python.exe src/evaluate.py --system baseline --verbose
"""

import argparse
import json
import os
import sys
import numpy as np
import re
import time
from termBased import Tokenizer, IndexBuilder, InvertedIndex, compute_idf, compute_bm25_score
# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "corpus.jsonl")
EVAL_PATH = os.path.join(PROJECT_ROOT, "data", "eval", "eval.jsonl")

# Make project root importable (so we can import baseline_rag)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
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


def load_corpus(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs

def load_eval(path):
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

# ──────────────────────────────────────────────
# Baseline retrieval wrapper
# ──────────────────────────────────────────────
#
# The baseline (baseline_rag.py) returns only the single best chunk via
# argmax. To compute Hit-Rate@k and MRR we need the full ranked list, so
# we replicate its scoring (cosine similarity on MiniLM embeddings over
# 400-char chunks) but return top-k instead of just the top-1.

def build_baseline_system(corpus_path):
    """Build the baseline index and return a retrieve(query, top_k) function."""
    from baseline_rag import load_docs, chunk_text, EMBED_MODEL, CHUNK_SIZE
    from sentence_transformers import SentenceTransformer

    docs = load_docs(corpus_path)
    model = SentenceTransformer(EMBED_MODEL)

    # Build chunks (same as baseline)
    chunks = []
    start = time.time()
    for d in docs:
        for sent_num, c in enumerate(split_sentences_punctuation(d["text"])):
            chunks.append({"doc_id": d["id"] + f"_s_{sent_num}", "title": d["title"], "text": f"Title: {d['title']}\n{c}"})
    # Encode and normalize (same as baseline)
    vectors = model.encode([c["text"] for c in chunks])
    end = time.time()
    build_time = (end -start)/len(docs)
    vectors = np.asarray(vectors, dtype="float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

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

    return retrieve , build_time

# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def evaluate(retrieve_fn, questions, verbose=False, abstain_threshold=0.7):
    """Run all questions through the retrieval function and compute metrics.

    abstain_threshold: if the top-1 score is below this value, the system
        abstains (returns "not found"). Set to 0.0 to disable abstention
        (pure baseline behaviour). Applied to BOTH answerable and
        unanswerable questions, so the trade-off is visible.
    """

    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    # ── Answerable questions ──
    hit1 = 0
    hit3 = 0
    mrr_sum = 0.0
    ans_scores = []
    ans_false_abstain = 0   # answerable questions wrongly rejected by threshold
    per_question = []
    retrieve_time = 0.0
    for q in answerable:
        expected = set(q["expected_docs"])
        start=time.time()
        results = retrieve_fn(q["question"])
        end=time.time()
        retrieve_time += (end-start)
        retrieved_docs = [result[0] for result in results]
        top_score = results[0][1] if results else 0.0
        ans_scores.append(top_score)

        # Threshold-based abstention check
        abstained = top_score < abstain_threshold if abstain_threshold > 0 else False

        # Hit-Rate@1
        h1 = retrieved_docs[0] in expected if retrieved_docs else False
        if h1 and not abstained:
            hit1 += 1

        # Hit-Rate@3 and MRR — find the rank of the first correct doc
        rank = None
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in expected:
                rank = i + 1
                break
        if rank is not None and rank <= 3 and not abstained:
            hit3 += 1
        if rank is not None and not abstained:
            mrr_sum += 1.0 / rank

        if abstained:
            ans_false_abstain += 1

        per_question.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected": q["expected_docs"],
            "retrieved": retrieved_docs[:3],
            "top_score": top_score,
            "hit@1": h1 and not abstained,
            "first_correct_rank": rank,
            "abstained": abstained,
            "answerable": True,
        })

        if verbose:
            if abstained:
                status = "FAB "  # false abstention (rejected an answerable q)
            elif h1:
                status = "OK  "
            else:
                status = "MISS"
            print(f"  {status} {q['id']} [{q['category']:16}] "
                  f"expected={q['expected_docs']}  "
                  f"got={retrieved_docs[:3]}  "
                  f"score={top_score:.3f}"
                  f"{'  <ABSTAIN' if abstained else ''}")
    
    retrieve_time=retrieve_time/len(answerable) if len(answerable)>0 else 0.0
    n_ans = len(answerable)
    # Effective hit-rate: correct retrievals among answerable that were NOT
    # falsely abstained. A high threshold lowers effective hit-rate because
    # it rejects legitimate questions.
    effective_n = n_ans - ans_false_abstain
    hit1_rate = hit1 / n_ans if n_ans else 0.0
    hit3_rate = hit3 / n_ans if n_ans else 0.0
    mrr = mrr_sum / n_ans if n_ans else 0.0
    false_abstain_rate = ans_false_abstain / n_ans if n_ans else 0.0

    # ── Unanswerable questions ──
    # With a threshold, the system abstains when top_score < threshold.
    # Correct abstention = top_score below threshold (true negative).
    # Incorrect = top_score >= threshold (fabricates a wrong answer).
    unans_scores = []
    correct_abstain = 0

    for q in unanswerable:
        results = retrieve_fn(q["question"])
        top_score = results[0][1] if results else 0.0
        unans_scores.append(top_score)

        abstained = top_score < abstain_threshold if abstain_threshold > 0 else False
        if abstained:
            correct_abstain += 1

        per_question.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected": [],
            "retrieved": [r[0] for r in results[:3]],
            "top_score": top_score,
            "abstained": abstained,
            "answerable": False,
        })

        if verbose:
            if abstained:
                status = "AB  "  # correctly abstained
            else:
                status = "FAB "  # false answer (fabricated)
            print(f"  {status} {q['id']} [{q['category']:16}] "
                  f"got={results[0][0] if results else 'none'}  "
                  f"score={top_score:.3f}"
                  f"{'  <ABSTAIN' if abstained else ''}")

    n_unans = len(unanswerable)
    abstention_rate = correct_abstain / n_unans if n_unans else 0.0
    fabrication_rate = (n_unans - correct_abstain) / n_unans if n_unans else 0.0

    # ── Score distributions (for threshold calibration) ──
    ans_scores = np.array(ans_scores) if ans_scores else np.array([0.0])
    unans_scores = np.array(unans_scores) if unans_scores else np.array([0.0])

    # ── Per-category breakdown ──
    categories = {}
    for pq in per_question:
        if not pq["answerable"]:
            continue
        cat = pq["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "hit1": 0}
        categories[cat]["total"] += 1
        if pq.get("hit@1"):
            categories[cat]["hit1"] += 1

    return {
        "n_answerable": n_ans,
        "n_unanswerable": n_unans,
        "abstain_threshold": abstain_threshold,
        "hit_rate_at_1": hit1_rate,
        "hit_rate_at_3": hit3_rate,
        "mrr": mrr,
        "abstention_rate": abstention_rate,
        "fabrication_rate": fabrication_rate,
        "false_abstain_rate": false_abstain_rate,
        "avg_score_answerable": float(np.mean(ans_scores)),
        "avg_score_unanswerable": float(np.mean(unans_scores)),
        "min_score_answerable": float(np.min(ans_scores)),
        "max_score_unanswerable": float(np.max(unans_scores)),
        "per_category": categories,
        "per_question": per_question,
    },retrieve_time

# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────

def print_report(metrics, system_name):
    w = 64
    print()
    print("=" * w)
    print(f"  EVALUATION REPORT — {system_name.upper()}")
    print("=" * w)
    print()
    print(f"  Questions:        {metrics['n_answerable']} answerable + "
          f"{metrics['n_unanswerable']} unanswerable = "
          f"{metrics['n_answerable'] + metrics['n_unanswerable']} total")
    print()
    if metrics.get("abstain_threshold", 0.0) > 0:
        print(f"  Abstention threshold: {metrics['abstain_threshold']:.3f}")
    print()
    print("  ── Retrieval Quality (answerable) ──")
    print(f"    Hit-Rate@1:     {metrics['hit_rate_at_1']:.1%}  "
          f"({int(metrics['hit_rate_at_1'] * metrics['n_answerable'])}/"
          f"{metrics['n_answerable']})")
    print(f"    Hit-Rate@3:     {metrics['hit_rate_at_3']:.1%}  "
          f"({int(metrics['hit_rate_at_3'] * metrics['n_answerable'])}/"
          f"{metrics['n_answerable']})")
    print(f"    MRR:            {metrics['mrr']:.3f}")
    if metrics.get("false_abstain_rate", 0) > 0:
        fa = metrics['false_abstain_rate']
        nfa = int(round(fa * metrics['n_answerable']))
        print(f"    False abstain:  {fa:.1%}  ({nfa}/{metrics['n_answerable']})"
              f"  <- answerable questions WRONGLY rejected")
    print()
    print("  ── Abstention (unanswerable) ──")
    print(f"    Abstention rate: {metrics['abstention_rate']:.1%}  "
          f"({int(metrics['abstention_rate'] * metrics['n_unanswerable'])}/"
          f"{metrics['n_unanswerable']})")
    if metrics.get("fabrication_rate", 0) > 0:
        fab = metrics['fabrication_rate']
        nfab = int(round(fab * metrics['n_unanswerable']))
        print(f"    Fabrication rate: {fab:.1%}  ({nfab}/{metrics['n_unanswerable']})"
              f"  <- unanswerable questions that WRONGLY got a fabricated answer")
    print()
    print("  ── Score Distribution (threshold calibration) ──")
    print(f"    Answerable   — avg={metrics['avg_score_answerable']:.3f}  "
          f"min={metrics['min_score_answerable']:.3f}")
    print(f"    Unanswerable — avg={metrics['avg_score_unanswerable']:.3f}  "
          f"max={metrics['max_score_unanswerable']:.3f}")
    gap = metrics["min_score_answerable"] - metrics["max_score_unanswerable"]
    if gap > 0:
        print(f"    Separability:  GOOD (gap = {gap:.3f} > 0)")
        print(f"                   A threshold of ~{metrics['max_score_unanswerable']:.3f}"
              f"–{metrics['min_score_answerable']:.3f} could separate the two sets.")
    else:
        print(f"    Separability:  POOR (gap = {gap:.3f} <= 0)")
        print(f"                   Score ranges overlap — threshold-based abstention")
        print(f"                   alone will not cleanly separate the two sets.")
    print()
    print("  ── Per-Category Breakdown (Hit-Rate@1) ──")
    for cat, vals in sorted(metrics["per_category"].items()):
        rate = vals["hit1"] / vals["total"] if vals["total"] else 0
        print(f"    {cat:22} {rate:.1%}  ({vals['hit1']}/{vals['total']})")
    print()
    print("=" * w)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval.")
    parser.add_argument("--system", choices=["baseline", "improved"],
                        default="bm25", help="Which system to evaluate")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question results")
    parser.add_argument("--threshold", type=float, default=0,
                        help="Abstention threshold: if top-1 score < this, "
                             "abstain. 0.0 = no abstention (pure baseline).")
    parser.add_argument("--corpus", default=CORPUS_PATH,
                        help="Path to corpus.jsonl")
    parser.add_argument("--eval", default=EVAL_PATH,
                        help="Path to eval.jsonl")
    args = parser.parse_args()

    questions = load_eval(args.eval)
    print(f"Loaded {len(questions)} evaluation questions from {args.eval}")

    if args.system == "baseline":
        print("Building baseline index (sentence-transformers MiniLM)...")
        retrieve_fn, build_time = build_baseline_system(args.corpus)
    elif args.system == "bm25":
        tokenizer = Tokenizer()
        index_builder = IndexBuilder(tokenizer)

        # Load your corpus here (list of documents)
        corpus = load_corpus(args.corpus)
        start = time.time()
        inverted_index = index_builder.build(corpus)
        end =  time.time()
        build_time = (end - start) / len(corpus)
        total_docs = len(corpus)
        idf = compute_idf(inverted_index, total_docs)
        retrieve_fn = lambda query: compute_bm25_score(
            query,
            index_builder,
            inverted_index,
            idf,
            index_builder.doc_lengths,
            index_builder.avgdl,
            tokenizer=tokenizer,
            top_k=5,
        )        # Improved system will be added in a later step


    if args.threshold > 0:
        print(f"Abstention threshold enabled: score < {args.threshold} -> abstain")
    print(f"build time for each index is: {build_time}")
    metrics,retrive_time = evaluate(retrieve_fn, questions, verbose=args.verbose,
                       abstain_threshold=args.threshold)
    print_report(metrics, args.system)

    # Save detailed results to JSON for the README
    suffix = f"_t{args.threshold}" if args.threshold > 0 else ""
    out_path = os.path.join(PROJECT_ROOT,"data", "eval",
                            f"results_{args.system}{suffix}.json")
    speed_metrics={"build_time" :build_time , "retrieve_time" : retrive_time}
    build_time_out_path = os.path.join(PROJECT_ROOT,"data", "eval",
                            f"results_{args.system}{suffix}_build_time.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(build_time_out_path, "w", encoding="utf-8") as f:
        json.dump(speed_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {out_path}")

if __name__ == "__main__":
    main()
