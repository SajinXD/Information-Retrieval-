import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

class IRMetrics:
    
    def __init__(self):
        
        self.results = defaultdict(dict)
    
    def precision_at_k(self, retrieved_docs, relevant_docs, k=10):
        
        if k == 0 or len(retrieved_docs) == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        
        relevant_retrieved = len([doc for doc in top_k if doc in relevant_docs])
        
        precision = relevant_retrieved / k
        
        return precision
    
    def recall_at_k(self, retrieved_docs, relevant_docs, k=10):
        
        if len(relevant_docs) == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        
        relevant_retrieved = len([doc for doc in top_k if doc in relevant_docs])
        
        recall = relevant_retrieved / len(relevant_docs)
        
        return recall
    
    def f1_score_at_k(self, retrieved_docs, relevant_docs, k=10):
        
        precision = self.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def average_precision(self, retrieved_docs, relevant_docs):
        
        if len(relevant_docs) == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_retrieved = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                precision_sum += precision_at_i
        
        
        ap = precision_sum / len(relevant_docs)
        
        return ap
    
    def mean_average_precision(self, all_retrieved_docs, all_relevant_docs):
       
        ap_scores = []
        
        for query_id in all_retrieved_docs.keys():
            retrieved = all_retrieved_docs[query_id]
            relevant = all_relevant_docs.get(query_id, set())
            
            ap = self.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        # Calculate mean
        map_score = np.mean(ap_scores) if ap_scores else 0.0
        
        return map_score
    
    def dcg_at_k(self, retrieved_docs, relevant_docs, k=10):
        
        dcg = 0.0
        
        for i, doc in enumerate(retrieved_docs[:k]):
            relevance = 1 if doc in relevant_docs else 0
            
            dcg += relevance / np.log2(i + 2)
        
        return dcg
    
    def idcg_at_k(self, relevant_docs, k=10):
        
        ideal_ranking = list(relevant_docs)[:k]
        
        idcg = 0.0
        for i in range(len(ideal_ranking)):
            idcg += 1.0 / np.log2(i + 2)
        
        return idcg
    
    def ndcg_at_k(self, retrieved_docs, relevant_docs, k=10):
        
        dcg = self.dcg_at_k(retrieved_docs, relevant_docs, k)
        idcg = self.idcg_at_k(relevant_docs, k)
        
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        
        return ndcg
    
    def mean_ndcg(self, all_retrieved_docs, all_relevant_docs, k=10):
        
        ndcg_scores = []
        
        for query_id in all_retrieved_docs.keys():
            retrieved = all_retrieved_docs[query_id]
            relevant = all_relevant_docs.get(query_id, set())
            
            ndcg = self.ndcg_at_k(retrieved, relevant, k)
            ndcg_scores.append(ndcg)
        
        mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        return mean_ndcg
    
    def reciprocal_rank(self, retrieved_docs, relevant_docs):
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        
        return 0.0  # No relevant document found
    
    def mean_reciprocal_rank(self, all_retrieved_docs, all_relevant_docs):
        
        rr_scores = []
        
        for query_id in all_retrieved_docs.keys():
            retrieved = all_retrieved_docs[query_id]
            relevant = all_relevant_docs.get(query_id, set())
            
            rr = self.reciprocal_rank(retrieved, relevant)
            rr_scores.append(rr)
        
        mrr = np.mean(rr_scores) if rr_scores else 0.0
        
        return mrr
    
    def precision_recall_curve(self, retrieved_docs, relevant_docs, max_k=50):
        
        k_values = []
        precision_values = []
        recall_values = []
        
        max_k = min(max_k, len(retrieved_docs))
        
        for k in range(1, max_k + 1):
            k_values.append(k)
            precision_values.append(self.precision_at_k(retrieved_docs, relevant_docs, k))
            recall_values.append(self.recall_at_k(retrieved_docs, relevant_docs, k))
        
        return k_values, precision_values, recall_values


def load_cisi_relevance(relevance_file):
    
    relevance_dict = defaultdict(set)
    
    try:
        with open(relevance_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    query_id = int(parts[0])
                    doc_id = int(parts[1])
                    relevance_dict[query_id].add(doc_id)
    except FileNotFoundError:
        print(f"Warning: Relevance file '{relevance_file}' not found.")
        print("Using sample data for demonstration...")
        for i in range(1, 11):
            relevance_dict[i] = set(range(i, i + 5))
    
    return relevance_dict


def evaluate_retrieval_model(model_name, all_retrieved_docs, all_relevant_docs):
    
    metrics_calculator = IRMetrics()
    results = {}
    
    k_values = [5, 10, 20, 50]
    
    print(f"\n{'='*70}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'='*70}\n")
    
    for k in k_values:
        precisions = []
        recalls = []
        f1_scores = []
        
        for query_id in all_retrieved_docs.keys():
            retrieved = all_retrieved_docs[query_id]
            relevant = all_relevant_docs.get(query_id, set())
            
            precisions.append(metrics_calculator.precision_at_k(retrieved, relevant, k))
            recalls.append(metrics_calculator.recall_at_k(retrieved, relevant, k))
            f1_scores.append(metrics_calculator.f1_score_at_k(retrieved, relevant, k))
        
        results[f'Precision@{k}'] = np.mean(precisions)
        results[f'Recall@{k}'] = np.mean(recalls)
        results[f'F1-Score@{k}'] = np.mean(f1_scores)
        
        print(f"Metrics at K={k}:")
        print(f"  Precision@{k}: {results[f'Precision@{k}']:.4f}")
        print(f"  Recall@{k}:    {results[f'Recall@{k}']:.4f}")
        print(f"  F1-Score@{k}:  {results[f'F1-Score@{k}']:.4f}")
        print()
    
    map_score = metrics_calculator.mean_average_precision(all_retrieved_docs, all_relevant_docs)
    results['MAP'] = map_score
    print(f"Mean Average Precision (MAP): {map_score:.4f}\n")
    
    for k in k_values:
        ndcg = metrics_calculator.mean_ndcg(all_retrieved_docs, all_relevant_docs, k)
        results[f'nDCG@{k}'] = ndcg
        print(f"nDCG@{k}: {ndcg:.4f}")
    print()
    
    mrr = metrics_calculator.mean_reciprocal_rank(all_retrieved_docs, all_relevant_docs)
    results['MRR'] = mrr
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}\n")
    
    print(f"{'='*70}\n")
    
    return results


def visualize_metrics(all_model_results):
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Information Retrieval Model Performance Comparison', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    model_names = list(all_model_results.keys())
    
    ax1 = axes[0, 0]
    k_values = [5, 10, 20, 50]
    x = np.arange(len(k_values))
    width = 0.25
    
    for i, model in enumerate(model_names):
        precision_values = [all_model_results[model][f'Precision@{k}'] for k in k_values]
        ax1.bar(x + i*width, precision_values, width, label=model, alpha=0.8)
    
    ax1.set_xlabel('K Value', fontweight='bold')
    ax1.set_ylabel('Precision', fontweight='bold')
    ax1.set_title('Precision@K Comparison', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'K={k}' for k in k_values])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for i, model in enumerate(model_names):
        recall_values = [all_model_results[model][f'Recall@{k}'] for k in k_values]
        ax2.bar(x + i*width, recall_values, width, label=model, alpha=0.8)
    
    ax2.set_xlabel('K Value', fontweight='bold')
    ax2.set_ylabel('Recall', fontweight='bold')
    ax2.set_title('Recall@K Comparison', fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'K={k}' for k in k_values])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    for i, model in enumerate(model_names):
        ndcg_values = [all_model_results[model][f'nDCG@{k}'] for k in k_values]
        ax3.bar(x + i*width, ndcg_values, width, label=model, alpha=0.8)
    
    ax3.set_xlabel('K Value', fontweight='bold')
    ax3.set_ylabel('nDCG', fontweight='bold')
    ax3.set_title('nDCG@K Comparison', fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f'K={k}' for k in k_values])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    

    ax4 = axes[1, 1]
    metrics = ['MAP', 'MRR']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(model_names):
        values = [all_model_results[model][metric] for metric in metrics]
        ax4.bar(x + i*width, values, width, label=model, alpha=0.8)
    
    ax4.set_xlabel('Metric', fontweight='bold')
    ax4.set_ylabel('Score', fontweight='bold')
    ax4.set_title('MAP and MRR Comparison', fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{output_path}'")
    
    return fig


def create_results_table(all_model_results):
    
    df = pd.DataFrame(all_model_results).T
    
    df = df.round(4)
    
    column_order = ['Precision@5', 'Precision@10', 'Precision@20', 'Precision@50',
                    'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                    'F1-Score@5', 'F1-Score@10', 'F1-Score@20', 'F1-Score@50',
                    'MAP', 'MRR',
                    'nDCG@5', 'nDCG@10', 'nDCG@20', 'nDCG@50']
    
    df = df[column_order]
    
    return df


if __name__ == "__main__":
    
    
    print("\n" + "="*70)
    print("INFORMATION RETRIEVAL SYSTEM EVALUATION")
    print("Assignment 7 - Performance Metrics Analysis")
    print("="*70 + "\n")
    
    relevance_file = "CISI.REL"  
    all_relevant_docs = load_cisi_relevance(relevance_file)
    
    tfidf_results = {}
    for query_id in range(1, 11):
        
        tfidf_results[query_id] = list(range(query_id, query_id + 50))
    
    cosine_results = {}
    for query_id in range(1, 11):
        cosine_results[query_id] = list(range(query_id - 1, query_id + 49))
    
    boolean_results = {}
    for query_id in range(1, 11):
        boolean_results[query_id] = list(range(query_id + 1, query_id + 51))
    
    all_model_results = {}
    
    all_model_results['TF-IDF'] = evaluate_retrieval_model(
        'TF-IDF Model', tfidf_results, all_relevant_docs
    )
    
    all_model_results['Cosine Similarity'] = evaluate_retrieval_model(
        'Cosine Similarity Model', cosine_results, all_relevant_docs
    )
    
    all_model_results['Boolean AND'] = evaluate_retrieval_model(
        'Boolean AND Model', boolean_results, all_relevant_docs
    )
    
    
    print("\n" + "="*70)
    print("CONSOLIDATED RESULTS TABLE")
    print("="*70 + "\n")
    
    results_df = create_results_table(all_model_results)
    print(results_df.to_string())
    
    
    csv_path = os.path.join(OUTPUT_DIR, 'evaluation_results.csv')
    results_df.to_csv(csv_path)
    print(f"\nResults saved to '{csv_path}'")
    
    
    print("\nGenerating visualizations...")
    visualize_metrics(all_model_results)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70 + "\n")
    
   
    print("\nKEY FINDINGS:")
    print("-" * 70)
    
    
    best_map = max(all_model_results.items(), key=lambda x: x[1]['MAP'])
    best_mrr = max(all_model_results.items(), key=lambda x: x[1]['MRR'])
    best_p10 = max(all_model_results.items(), key=lambda x: x[1]['Precision@10'])
    best_ndcg10 = max(all_model_results.items(), key=lambda x: x[1]['nDCG@10'])
    
    print(f"1. Best MAP Score: {best_map[0]} ({best_map[1]['MAP']:.4f})")
    print(f"2. Best MRR Score: {best_mrr[0]} ({best_mrr[1]['MRR']:.4f})")
    print(f"3. Best Precision@10: {best_p10[0]} ({best_p10[1]['Precision@10']:.4f})")
    print(f"4. Best nDCG@10: {best_ndcg10[0]} ({best_ndcg10[1]['nDCG@10']:.4f})")
    print("-" * 70)
    
    print(f"\nAll output files have been saved to the '{OUTPUT_DIR}' directory")