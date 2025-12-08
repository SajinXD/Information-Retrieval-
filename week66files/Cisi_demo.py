import os
import urllib.request
import zipfile
from ir_system import IRSystem, Evaluator
import json
import time


def download_cisi_dataset(output_dir='data'):
    """Download CISI dataset from a public source"""
    print("Downloading CISI dataset...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
   
    base_url = "http://ir.dcs.gla.ac.uk/resources/test_collections/cisi/"
    files = {
        'CISI.ALL': 'documents',
        'CISI.QRY': 'queries',
        'CISI.REL': 'relevance judgments'
    }
    
    print(f"Creating data directory: {output_dir}")
    
    for filename, description in files.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists")
            continue
        
        try:
            url = base_url + filename
            print(f"Downloading {filename} ({description})...")
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            print(f"  Please manually download from: {url}")
    
    return output_dir


def load_cisi_relevance(filepath):
    """Load CISI relevance judgments"""
    relevance = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    query_id = int(parts[0])
                    doc_id = int(parts[1])
                    
                    if query_id not in relevance:
                        relevance[query_id] = set()
                    relevance[query_id].add(doc_id)
    except FileNotFoundError:
        print(f"Relevance file not found: {filepath}")
        return {}
    
    return relevance


def run_comprehensive_evaluation(ir_system, queries, relevance_judgments, output_file='results.json'):
    """Run comprehensive evaluation across multiple models"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 70)
    
    models = ['bm25', 'tf-idf', 'cosine']
    k_values = [5, 10, 20]
    
    all_results = {}
    
    for model in models:
        print(f"\nEvaluating model: {model.upper()}")
        print("-" * 70)
        
        model_results = {
            'model': model,
            'query_results': {},
            'metrics': {}
        }
        
        query_results = {}
        
        
        for query_id, query_text in queries.items():
            results = ir_system.search(query_text, model=model, top_k=20)
            retrieved_ids = [doc_id for doc_id, score in results]
            query_results[query_id] = retrieved_ids
            
            model_results['query_results'][query_id] = {
                'retrieved': retrieved_ids[:10],
                'scores': [float(score) for doc_id, score in results[:10]]
            }
        
        
        precisions_at_k = {k: [] for k in k_values}
        recalls_at_k = {k: [] for k in k_values}
        aps = []
        
        for query_id, retrieved in query_results.items():
            relevant = relevance_judgments.get(query_id, set())
            
            if relevant:  
                
                for k in k_values:
                    p = Evaluator.precision_at_k(retrieved, relevant, k)
                    r = Evaluator.recall_at_k(retrieved, relevant, k)
                    precisions_at_k[k].append(p)
                    recalls_at_k[k].append(r)
                
                
                ap = Evaluator.average_precision(retrieved, relevant)
                aps.append(ap)
        
        
        for k in k_values:
            avg_p = sum(precisions_at_k[k]) / len(precisions_at_k[k]) if precisions_at_k[k] else 0
            avg_r = sum(recalls_at_k[k]) / len(recalls_at_k[k]) if recalls_at_k[k] else 0
            model_results['metrics'][f'P@{k}'] = round(avg_p, 4)
            model_results['metrics'][f'R@{k}'] = round(avg_r, 4)
            print(f"  Precision@{k}: {avg_p:.4f}")
            print(f"  Recall@{k}: {avg_r:.4f}")
        
        map_score = sum(aps) / len(aps) if aps else 0
        model_results['metrics']['MAP'] = round(map_score, 4)
        print(f"  MAP: {map_score:.4f}")
        
        all_results[model] = model_results
    

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


def run_example_searches(ir_system, queries):
    """Run example searches and display results"""
    print("\n" + "=" * 70)
    print("EXAMPLE SEARCH RESULTS")
    print("=" * 70)
    
    example_queries = dict(list(queries.items())[:3])
    
    for query_id, query_text in example_queries.items():
        print(f"\nQuery {query_id}: {query_text[:100]}...")
        print("-" * 70)
        
        results = ir_system.search(query_text, model='bm25', top_k=5)
        
        if results:
            for rank, (doc_id, score) in enumerate(results, 1):
                doc_text = ir_system.get_document(doc_id)
                preview = doc_text[:150].replace('\n', ' ')
                print(f"\n{rank}. Document {doc_id} (BM25 Score: {score:.4f})")
                print(f"   Preview: {preview}...")
        else:
            print("No results found")


def create_visualization_data(results, output_file='visualization_data.json'):
    """Create data for visualization"""
    viz_data = {
        'comparison': {},
        'metrics': []
    }
    
    models = list(results.keys())
    metrics = ['MAP', 'P@5', 'P@10', 'R@5', 'R@10']
    
    for metric in metrics:
        metric_data = {
            'metric': metric,
            'values': {}
        }
        for model in models:
            metric_data['values'][model] = results[model]['metrics'].get(metric, 0)
        viz_data['metrics'].append(metric_data)
    
    with open(output_file, 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print(f"Visualization data saved to: {output_file}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("CISI INFORMATION RETRIEVAL SYSTEM - COMPLETE DEMO")
    print("=" * 70)
    

    data_dir = download_cisi_dataset()

    doc_file = os.path.join(data_dir, 'CISI.ALL')
    query_file = os.path.join(data_dir, 'CISI.QRY')
    rel_file = os.path.join(data_dir, 'CISI.REL')
    
    if not all(os.path.exists(f) for f in [doc_file, query_file, rel_file]):
        print("\n⚠ Warning: Could not download all CISI files automatically.")
        print("Please download manually from: http://ir.dcs.gla.ac.uk/resources/test_collections/cisi/")
        print("Required files: CISI.ALL, CISI.QRY, CISI.REL")
        print("\nUsing sample data for demonstration instead...")
        
        # Use sample data
        from ir_system import main as demo_main
        demo_main()
        return

    print("\n" + "=" * 70)
    print("INITIALIZING IR SYSTEM")
    print("=" * 70)
    
    ir_system = IRSystem()

    print("\nLoading CISI documents...")
    documents = ir_system.load_cisi_documents(doc_file)
    print(f"✓ Loaded {len(documents)} documents")
    

    print("\nBuilding IR system...")
    start_time = time.time()
    ir_system.build_system(documents)
    build_time = time.time() - start_time
    print(f"✓ System built in {build_time:.2f} seconds")
    
    
    stats = ir_system.get_statistics()
    print("\n" + "=" * 70)
    print("SYSTEM STATISTICS")
    print("=" * 70)
    print(f"Number of documents: {stats['num_documents']}")
    print(f"Unique terms: {stats['num_unique_terms']}")
    print(f"Average document length: {stats['avg_doc_length']:.2f} tokens")
    print(f"Total tokens: {stats['total_tokens']}")
    
    
    print("\nLoading queries...")
    queries = ir_system.load_cisi_queries(query_file)
    print(f"✓ Loaded {len(queries)} queries")
    
    
    print("\nLoading relevance judgments...")
    relevance = load_cisi_relevance(rel_file)
    print(f"✓ Loaded relevance judgments for {len(relevance)} queries")
    
    
    run_example_searches(ir_system, queries)
    
   
    results = run_comprehensive_evaluation(ir_system, queries, relevance)
    
    
    create_visualization_data(results)
    
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ IR System successfully built and evaluated")
    print("✓ Results saved to: results.json")
    print("✓ Visualization data saved to: visualization_data.json")
    print("\nBest performing model by MAP:")
    
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['MAP'])
    print(f"  {best_model[0].upper()}: MAP = {best_model[1]['metrics']['MAP']:.4f}")
    
    print("\n" + "=" * 70)
    print("Demo complete! Check the output files for detailed results.")
    print("=" * 70)


if __name__ == "__main__":
    main()