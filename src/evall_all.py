#!/usr/bin/env python3
"""
evaluate_all.py

A unified script to run evaluations for all retrieval configurations 
(BM25, Dense Retriever, and Hybrid Reranker) on both 'eval' and 'test' datasets.
Outputs results to the console, CSV, JSON, and a structured PDF report.
"""

import os
import sys
import time
import json
import csv

# ─────────────────────────────────────────────────────────────
# Path Configuration & Dynamic Imports
# ─────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Add 'src' directory to path if the script is placed in the project root
src_dir = os.path.join(script_dir, "src")
if os.path.isdir(src_dir):
    sys.path.insert(0, src_dir)

try:
    from evaluate import evaluate, load_eval, CORPUS_PATH, EVAL_PATH, TEST_PATH
    from termBased import (
        Tokenizer, IndexBuilder, compute_idf, compute_bm25_score, 
        build_baseline_system, load_corpus
    )
except ImportError as e:
    print(f"Error: Could not import required modules. Ensure 'evaluate.py' "
          f"and 'termBased.py' are in the same directory or within a 'src' folder.\n{e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Caching System Builders (To avoid redundant index building)
# ─────────────────────────────────────────────────────────────
system_cache = {}

def get_cached_system(system_name, sliding_window, corpus_path):
    """
    Initializes and caches retrieval systems to ensure embedding models
    and index building only occur once per setting.
    """
    cache_key = (system_name, sliding_window)
    if cache_key in system_cache:
        return system_cache[cache_key]
    
    print(f"Building/compiling system: {system_name} (sliding_window={sliding_window})...")
    
    if system_name == "bm25":
        tokenizer = Tokenizer()
        index_builder = IndexBuilder(tokenizer)
        corpus = load_corpus(corpus_path)
        start = time.time()
        inverted_index = index_builder.build(corpus)
        end = time.time()
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
            top_k=8,
        )
        
    elif system_name == "retriever":
        # Returns reranker, retrieve, build_time. We want retrieve here.
        _, retrieve_fn, build_time = build_baseline_system(
            corpus_path, sliding_window=sliding_window, window_size=100, step_size=50
        )
        
    elif system_name == "reranker":
        tokenizer = Tokenizer()
        index_builder = IndexBuilder(tokenizer)
        corpus = load_corpus(corpus_path)
        start = time.time()
        inverted_index = index_builder.build(corpus)
        end = time.time()
        bm25_build_time = (end - start) / len(corpus)
        total_docs = len(corpus)
        idf = compute_idf(inverted_index, total_docs)
        
        # We need the reranker core function and its encoding build time
        reranker_func, _, reranker_build_time = build_baseline_system(
            corpus_path, sliding_window=sliding_window, window_size=100, step_size=50
        )
        build_time = bm25_build_time + reranker_build_time
        
        bm_25 = lambda query: compute_bm25_score(
            query,
            index_builder,
            inverted_index,
            idf,
            index_builder.doc_lengths,
            index_builder.avgdl,
            tokenizer=tokenizer,
            top_k=8,
        )
        retrieve_fn = lambda query: reranker_func(query, bm_25, top_k=8)
    else:
        raise ValueError(f"Unknown system setting: {system_name}")
        
    system_cache[cache_key] = (retrieve_fn, build_time)
    return retrieve_fn, build_time


# ─────────────────────────────────────────────────────────────
# PDF Generation with ReportLab
# ─────────────────────────────────────────────────────────────
def generate_pdf_report(results, pdf_path):
    """
    Compiles a professional landscape PDF report using ReportLab.
    """
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
    except ImportError:
        print("\n[!] 'reportlab' package not found. Skipping PDF generation.")
        print("    To generate the PDF report, please run: pip install reportlab\n")
        return False

    print(f"Generating PDF report at '{pdf_path}'...")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Text Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=15
    )
    
    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=0
    )
    
    legend_title = ParagraphStyle(
        'LegendTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor("#2D3748")
    )
    
    legend_text = ParagraphStyle(
        'LegendText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor("#4A5568")
    )

    elements = []
    
    # Title & Metadata
    elements.append(Paragraph("RAG Retrieval Pipeline Benchmark Report", title_style))
    elements.append(Paragraph(
        f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Corpus Path: {CORPUS_PATH}", subtitle_style
    ))
    
    # Columns Header Configuration
    headers = [
        "System / Setting", "Dataset", "Threshold", "Hit-Rate@1", 
        "Hit-Rate@3", "MRR", "Abstention", "Fabrication", "Accuracy", "Latency"
    ]
    table_data = [headers]
    
    for r in results:
        table_data.append([
            r["setting"],
            r["dataset"],
            f"{r['threshold']:.2f}",
            f"{r['hit_rate_at_1']:.2%}",
            f"{r['hit_rate_at_3']:.2%}",
            f"{r['mrr']:.3f}",
            f"{r['abstention_rate']:.2%}",
            f"{r['fabrication_rate']:.2%}",
            f"{r['accuracy']:.2%}",
            f"{r['latency'] * 1000:.1f} ms"
        ])
        
    # Printable area width in Landscape letter is 792 - (36 * 2) = 720 points
    col_widths = [140, 45, 55, 55, 55, 45, 65, 65, 55, 65]
    
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    t_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor("#1A365D")),
    ])
    
    # Alternating Row Styles
    for i in range(1, len(table_data)):
        bg_color = colors.HexColor("#F7FAFC") if i % 2 == 0 else colors.white
        t_style.add('BACKGROUND', (0, i), (-1, i), bg_color)
        t_style.add('ALIGN', (1, i), (-1, i), 'CENTER')
        t_style.add('FONTNAME', (1, i), (-1, i), 'Helvetica')
        t_style.add('FONTSIZE', (1, i), (-1, i), 8)
        t_style.add('BOTTOMPADDING', (0, i), (-1, i), 5)
        t_style.add('TOPPADDING', (0, i), (-1, i), 5)
        
    t.setStyle(t_style)
    elements.append(t)
    
    # Legend Explanation block
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Metric Descriptions & Evaluation Notes:", legend_title))
    elements.append(Spacer(1, 3))
    elements.append(Paragraph(
        "• <b>Hit-Rate@1 / Hit-Rate@3:</b> Rate of answerable queries where the correct document was retrieved at rank 1, or within the top 3 (deduplicated by document ID) respectively.<br/>"
        "• <b>MRR (Mean Reciprocal Rank):</b> Reciprocal of the rank of the first correct document (1/rank), averaged over all answerable queries.<br/>"
        "• <b>Abstention Rate:</b> The fraction of unanswerable queries where the system correctly returned no document (top-1 score was below threshold).<br/>"
        "• <b>Fabrication Rate:</b> The fraction of unanswerable queries where the system incorrectly fabricated an answer (top-1 score met/exceeded threshold).<br/>"
        "• <b>Accuracy:</b> The overall percentage of correct behaviors (correct retrievals for answerable queries + correct abstentions for unanswerable ones).<br/>"
        "• <b>Latency:</b> The average retrieval / rerank execution time per query in milliseconds.",
        legend_text
    ))
    
    doc.build(elements)
    print(f"PDF report successfully saved to '{pdf_path}'")
    return True


# ─────────────────────────────────────────────────────────────
# Main Benchmark Loop
# ─────────────────────────────────────────────────────────────
def main():
    print(f"Starting pipeline-wide evaluation benchmarking...")
    print(f"Corpus path: {CORPUS_PATH}")
    print(f"Eval path:   {EVAL_PATH}")
    print(f"Test path:   {TEST_PATH}\n")

    # Layout: (system_name, sliding_window, threshold, dataset_name, dataset_path)
    configs = [
        # Baseline BM25
        ("bm25", False, 0.0, "Eval", EVAL_PATH),
        ("bm25", False, 0.0, "Test", TEST_PATH),
        
        # Dense Retriever (Sentence-Level Split)
        ("retriever", False, 0.5, "Eval", EVAL_PATH),
        ("retriever", False, 0.65, "Eval", EVAL_PATH),
        ("retriever", False, 0.5, "Test", TEST_PATH),
        ("retriever", False, 0.65, "Test", TEST_PATH),
        
        # Dense Retriever (Sliding Window Split)
        ("retriever", True, 0.5, "Eval", EVAL_PATH),
        ("retriever", True, 0.65, "Eval", EVAL_PATH),
        ("retriever", True, 0.5, "Test", TEST_PATH),
        ("retriever", True, 0.65, "Test", TEST_PATH),
        
        # Hybrid Reranker (Sentence-Level)
        ("reranker", False, 0.5, "Eval", EVAL_PATH),
        ("reranker", False, 0.65, "Eval", EVAL_PATH),
        ("reranker", False, 0.5, "Test", TEST_PATH),
        ("reranker", False, 0.65, "Test", TEST_PATH),
        
        # Hybrid Reranker (Sliding Window)
        ("reranker", True, 0.5, "Eval", EVAL_PATH),
        ("reranker", True, 0.65, "Eval", EVAL_PATH),
        ("reranker", True, 0.5, "Test", TEST_PATH),
        ("reranker", True, 0.65, "Test", TEST_PATH),
        
    ]

    results = []

    for idx, (sys_name, sliding, threshold, ds_name, ds_path) in enumerate(configs, 1):
        print(f"[{idx}/{len(configs)}] Processing: {sys_name} | Dataset: {ds_name} | Threshold: {threshold} (sliding={sliding})")
        
        try:
            questions = load_eval(ds_path)
        except Exception as e:
            print(f"  --> Error loading {ds_name} dataset: {e}. Skipping configuration.")
            continue
            
        try:
            retrieve_fn, build_time = get_cached_system(sys_name, sliding, CORPUS_PATH)
        except Exception as e:
            print(f"  --> Error preparing system: {e}. Skipping configuration.")
            continue
            
        # Execute Evaluation
        metrics, avg_latency = evaluate(retrieve_fn, questions, verbose=False, abstain_threshold=threshold)
        
        # Format a descriptive setting name
        label = sys_name.upper()
        if sliding:
            label += " (Sliding Window)"
        
        results.append({
            "setting": label,
            "dataset": ds_name,
            "threshold": threshold,
            "hit_rate_at_1": metrics["hit_rate_at_1"],
            "hit_rate_at_3": metrics["hit_rate_at_3"],
            "mrr": metrics["mrr"],
            "abstention_rate": metrics["abstention_rate"],
            "fabrication_rate": metrics["fabrication_rate"],
            "accuracy": metrics["accuracy"],
            "latency": avg_latency,
            "build_time": build_time
        })

    # ─────────────────────────────────────────────────────────────
    # Outputs & Serialization
    # ─────────────────────────────────────────────────────────────
    
    # 1. Print beautifully formatted Markdown table to Console
    print("\n" + "=" * 115)
    print(f"| {'System / Setting':<28} | {'Dataset':<7} | {'Thresh':<6} | {'Hit@1':<8} | {'Hit@3':<8} | "
          f"{'MRR':<6} | {'Abstain':<8} | {'Fabric.':<8} | {'Accuracy':<8} | {'Latency':<10} |")
    print("-" * 115)
    for r in results:
        print(f"| {r['setting']:<28} | {r['dataset']:<7} | {r['threshold']:<6.1f} | "
              f"{r['hit_rate_at_1']:<8.1%} | {r['hit_rate_at_3']:<8.1%} | {r['mrr']:<6.3f} | "
              f"{r['abstention_rate']:<8.1%} | {r['fabrication_rate']:<8.1%} | {r['accuracy']:<8.1%} | "
              f"{r['latency']*1000:<8.1f} ms |")
    print("=" * 115 + "\n")

    # 2. Save JSON file
    json_path = os.path.join(script_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved raw JSON results to '{json_path}'")

    # 3. Save CSV file
    csv_path = os.path.join(script_dir, "benchmark_results.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Setting", "Dataset", "Threshold", "Hit-Rate@1", 
            "Hit-Rate@3", "MRR", "Abstention Rate", "Fabrication Rate", "Accuracy", "Avg Latency (s)", "Index Build Time"
        ])
        for r in results:
            writer.writerow([
                r["setting"], r["dataset"], r["threshold"],
                f"{r['hit_rate_at_1']:.4f}", f"{r['hit_rate_at_3']:.4f}", f"{r['mrr']:.4f}",
                f"{r['abstention_rate']:.4f}", f"{r['fabrication_rate']:.4f}", f"{r['accuracy']:.4f}",
                f"{r['latency']:.6f}", f"{r['build_time']:.6f}"
            ])
    print(f"Saved spreadsheet CSV results to '{csv_path}'")

    # 4. Generate the PDF
    pdf_path = os.path.join(script_dir, "benchmark_report.pdf")
    generate_pdf_report(results, pdf_path)

if __name__ == "__main__":
    main()