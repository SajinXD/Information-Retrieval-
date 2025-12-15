import numpy as np
from collections import defaultdict
import pickle
import os
import sys

try:
    from ir_evaluation import IRMetrics, evaluate_retrieval_model, load_cisi_relevance, visualize_metrics, create_results_table
    print("ir_evaluation module loaded successfully")
except ImportError:
    print("ERROR: Could not import ir_evaluation module.")
    print("Make sure 'ir_evaluation.py' is in the same directory as this script.")
    sys.exit(1)

def check_files_exist():
    
    required_files = {
        'CISI.ALL': 'Documents file',
        'CISI.QRY': 'Queries file', 
        'CISI.REL': 'Relevance judgments file'
    }
    
    missing_files = []
    for filename, description in required_files.items():
        if not os.path.exists(filename):
            missing_files.append(f"  - {filename} ({description})")
    
    if missing_files:
        print("\n" + "="*70)
        print("MISSING REQUIRED FILES")
        print("="*70)
        print("\nThe following CISI dataset files are missing:")
        for file in missing_files:
            print(file)
        print("\nPlease place these files in the current directory:")
        print(f"  {os.getcwd()}")
        print("\nYou can download the CISI dataset from:")
        print("  http://ir.dcs.gla.ac.uk/resources/test_collections/cisi/")
        print("\nOr run in DEMO MODE with sample data (see below)")
        print("="*70 + "\n")
        return False
    
    return True

def generate_sample_retrieval_results(num_queries=10):
    
    print("\n" + "="*70)
    print("RUNNING IN DEMO MODE - USING SAMPLE DATA")
    print("="*70)
    print("\nGenerating sample retrieval results for demonstration...")
    
    
    all_relevant_docs = {}
    for query_id in range(1, num_queries + 1):
        num_relevant = np.random.randint(5, 9)
        relevant_start = query_id * 10
        all_relevant_docs[query_id] = set(range(relevant_start, relevant_start + num_relevant))
    
    
    tfidf_results = {}
    for query_id in range(1, num_queries + 1):
        relevant = list(all_relevant_docs[query_id])
        non_relevant = list(range(1, 200))
        non_relevant = [doc for doc in non_relevant if doc not in relevant]
        
        top_relevant = np.random.choice(relevant, size=min(6, len(relevant)), replace=False).tolist()
        remaining = [doc for doc in relevant if doc not in top_relevant]
        filler = np.random.choice(non_relevant, size=50, replace=False).tolist()
        
        tfidf_results[query_id] = top_relevant + filler[:4] + remaining + filler[4:]
    
    cosine_results = {}
    for query_id in range(1, num_queries + 1):
        relevant = list(all_relevant_docs[query_id])
        non_relevant = list(range(1, 200))
        non_relevant = [doc for doc in non_relevant if doc not in relevant]
        
        top_relevant = np.random.choice(relevant, size=min(7, len(relevant)), replace=False).tolist()
        remaining = [doc for doc in relevant if doc not in top_relevant]
        filler = np.random.choice(non_relevant, size=50, replace=False).tolist()
        
        cosine_results[query_id] = top_relevant + filler[:3] + remaining + filler[3:]
    
    boolean_results = {}
    for query_id in range(1, num_queries + 1):
        relevant = list(all_relevant_docs[query_id])
        non_relevant = list(range(1, 200))
        non_relevant = [doc for doc in non_relevant if doc not in relevant]
        
        selected_relevant = np.random.choice(relevant, size=min(4, len(relevant)), replace=False).tolist()
        filler = np.random.choice(non_relevant, size=20, replace=False).tolist()
        
        boolean_results[query_id] = selected_relevant + filler
    
    print(f"✓ Generated sample data for {num_queries} queries")
    print(f"✓ TF-IDF results: ~60% relevant in top 10")
    print(f"✓ Cosine Similarity results: ~70% relevant in top 10")
    print(f"✓ Boolean AND results: ~50% relevant (strict matching)")
    print("="*70 + "\n")
    
    return tfidf_results, cosine_results, boolean_results, all_relevant_docs


def load_tfidf_model(documents_file, queries_file):
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("Loading documents...")
    documents = []
    doc_ids = []
    
    with open(documents_file, 'r', encoding='utf-8', errors='ignore') as f:
        current_doc = ""
        current_id = None
        
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                if current_doc and current_id:
                    documents.append(current_doc)
                    doc_ids.append(current_id)
                current_id = int(line.split()[1])
                current_doc = ""
            elif not line.startswith('.'):
                current_doc += line + " "
        
        if current_doc and current_id:
            documents.append(current_doc)
            doc_ids.append(current_id)
    
    print(f"✓ Loaded {len(documents)} documents")
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        lowercase=True,
        min_df=2
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"✓ Created TF-IDF matrix: {tfidf_matrix.shape}")
    
    return vectorizer, tfidf_matrix, doc_ids


def load_queries(queries_file):
    
    print("Loading queries...")
    queries = []
    
    with open(queries_file, 'r', encoding='utf-8', errors='ignore') as f:
        current_query = {"id": 0, "text": ""}
        
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                if current_query["text"]:
                    queries.append(current_query)
                current_query = {"id": int(line.split()[1]), "text": ""}
            elif line.startswith('.W'):
                continue
            else:
                current_query["text"] += line + " "
        
        if current_query["text"]:
            queries.append(current_query)
    
    print(f"✓ Loaded {len(queries)} queries")
    return queries


def retrieve_tfidf(query_text, vectorizer, tfidf_matrix, doc_ids, top_k=100):
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    query_vector = vectorizer.transform([query_text])
    
    
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    retrieved_doc_ids = [doc_ids[idx] for idx in top_indices]
    
    return retrieved_doc_ids


def process_queries_real_data(queries, vectorizer, tfidf_matrix, doc_ids):
    
    tfidf_results = {}
    cosine_results = {}  
    
    print(f"\nProcessing {len(queries)} queries...")
    
    for i, query in enumerate(queries):
        query_id = query["id"]
        query_text = query["text"]
        
        retrieved = retrieve_tfidf(query_text, vectorizer, tfidf_matrix, doc_ids)
        tfidf_results[query_id] = retrieved
        cosine_results[query_id] = retrieved  
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1} queries...")
    
    print(f"✓ Query processing complete!")
    
    boolean_results = {}
    for query_id, docs in tfidf_results.items():
        boolean_results[query_id] = docs[:30] 
    
    return tfidf_results, cosine_results, boolean_results

def run_with_real_data():
    
    print("\n" + "="*70)
    print("IR MODEL INTEGRATION AND EVALUATION")
    print("="*70)
    
    DOCUMENTS_FILE = "CISI.ALL"
    QUERIES_FILE = "CISI.QRY"
    RELEVANCE_FILE = "CISI.REL"
    
    print("\n[1/4] Loading retrieval models...")
    vectorizer, tfidf_matrix, doc_ids = load_tfidf_model(DOCUMENTS_FILE, QUERIES_FILE)
    
    print("\n[2/4] Loading queries...")
    queries = load_queries(QUERIES_FILE)
    
    print("\n[3/4] Processing queries with all models...")
    tfidf_results, cosine_results, boolean_results = process_queries_real_data(
        queries, vectorizer, tfidf_matrix, doc_ids
    )
    
    print("\n[4/4] Loading relevance judgments...")
    all_relevant_docs = load_cisi_relevance(RELEVANCE_FILE)
    print(f"✓ Loaded relevance judgments for {len(all_relevant_docs)} queries")
    
    return tfidf_results, cosine_results, boolean_results, all_relevant_docs


def run_with_sample_data():
    
    return generate_sample_retrieval_results(num_queries=10)


def main():

    files_exist = check_files_exist()
    
    if files_exist:
        try:
            tfidf_results, cosine_results, boolean_results, all_relevant_docs = run_with_real_data()
        except Exception as e:
            print(f"\nError processing CISI files: {e}")
            print("Falling back to DEMO MODE...")
            tfidf_results, cosine_results, boolean_results, all_relevant_docs = run_with_sample_data()
    else:
        print("\nOptions:")
        print("1. Place CISI files in current directory and run again")
        print("2. Run in DEMO MODE with sample data")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '2':
            tfidf_results, cosine_results, boolean_results, all_relevant_docs = run_with_sample_data()
        else:
            print("\nPlease add CISI files and run the script again.")
            sys.exit(0)
    
    print("\n" + "="*70)
    print("EVALUATING MODELS")
    print("="*70)
    
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
    print("GENERATING RESULTS")
    print("="*70)
    
    results_df = create_results_table(all_model_results)
    print("\nResults Summary:")
    print(results_df.to_string())
    
    OUTPUT_DIR = 'output'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    csv_path = os.path.join(OUTPUT_DIR, 'evaluation_results.csv')
    results_df.to_csv(csv_path)
    print(f"\n✓ Results saved to '{csv_path}'")
    
    visualize_metrics(all_model_results)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nAll output files saved to: {os.path.abspath(OUTPUT_DIR)}")

def print_quick_start_guide():
    """
    Print instructions for using this script
    """
    print("""
╔════════════════════════════════════════════════════════════════════╗
║           IR MODEL INTEGRATION - QUICK START GUIDE                 ║
╚════════════════════════════════════════════════════════════════════╝

OPTION 1: Run with Your CISI Dataset
────────────────────────────────────────────────────────────────────
1. Download the CISI dataset from:
   http://ir.dcs.gla.ac.uk/resources/test_collections/cisi/

2. Place these files in the same directory as this script:
   - CISI.ALL (documents)
   - CISI.QRY (queries)
   - CISI.REL (relevance judgments)

3. Run: python integration_example.py

OPTION 2: Run in Demo Mode
────────────────────────────────────────────────────────────────────
1. Run: python integration_example.py
2. Choose option 2 when prompted
3. The script will use sample data for demonstration

OUTPUT FILES
────────────────────────────────────────────────────────────────────
All results will be saved to the 'output' directory:
- evaluation_results.csv (metrics table)
- model_comparison.png (visualization charts)

CUSTOMIZATION
────────────────────────────────────────────────────────────────────
To integrate YOUR retrieval models:
1. Replace load_tfidf_model() with your TF-IDF implementation
2. Replace retrieve_tfidf() with your retrieval function
3. Add your Cosine Similarity and Boolean AND implementations

For detailed instructions, see the comments in the code.

════════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print_quick_start_guide()
        sys.exit(0)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nFor help, run: python integration_example.py --help")
        sys.exit(1)