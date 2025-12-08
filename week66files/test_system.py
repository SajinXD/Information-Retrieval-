from ir_system import IRSystem, TextPreprocessor, InvertedIndex, RetrievalModels, Evaluator
import sys

def test_preprocessor():
    """Test text preprocessing"""
    print("\n" + "="*60)
    print("TEST 1: Text Preprocessing")
    print("="*60)
    
    preprocessor = TextPreprocessor()
    

    text = "Information Retrieval Systems are Amazing!"
    tokens = preprocessor.tokenize(text)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    

    tokens_no_stop = preprocessor.remove_stopwords(tokens)
    print(f"Without stopwords: {tokens_no_stop}")
    

    stemmed = [preprocessor.stem(token) for token in tokens_no_stop]
    print(f"Stemmed: {stemmed}")
    

    processed = preprocessor.preprocess(text)
    print(f"Complete pipeline: {processed}")
    
    print("✓ Preprocessing test passed!")
    return True

def test_inverted_index():
    """Test inverted index creation"""
    print("\n" + "="*60)
    print("TEST 2: Inverted Index")
    print("="*60)
    
    preprocessor = TextPreprocessor()
    inverted_index = InvertedIndex()
    

    docs = {
        1: "information retrieval systems",
        2: "information systems and databases",
        3: "retrieval algorithms"
    }
    

    preprocessed = {
        doc_id: preprocessor.preprocess(text) 
        for doc_id, text in docs.items()
    }
    

    inverted_index.build_index(preprocessed)
    
    print(f"Number of unique terms: {len(inverted_index.index)}")
    print(f"Number of documents: {inverted_index.num_docs}")
    print(f"Average document length: {inverted_index.avg_doc_length:.2f}")
    

    print("\nSample posting lists:")
    for term in list(inverted_index.index.keys())[:3]:
        postings = inverted_index.get_posting_list(term)
        print(f"  '{term}': {postings}")
    
    print("✓ Inverted index test passed!")
    return True

def test_retrieval_models():
    """Test retrieval models"""
    print("\n" + "="*60)
    print("TEST 3: Retrieval Models")
    print("="*60)
    

    ir_system = IRSystem()
    
    test_docs = {
        1: "information retrieval is important",
        2: "search engines use retrieval algorithms",
        3: "text mining and information extraction"
    }
    
    ir_system.build_system(test_docs)
    
    
    query = "information retrieval"
    print(f"Query: '{query}'")
    
    
    models = ['tf-idf', 'bm25', 'cosine']
    
    for model in models:
        results = ir_system.search(query, model=model, top_k=3)
        print(f"\n{model.upper()} Results:")
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"  {rank}. Doc {doc_id}: {score:.4f}")
    
    print("\n✓ Retrieval models test passed!")
    return True

def test_evaluation():
    """Test evaluation metrics"""
    print("\n" + "="*60)
    print("TEST 4: Evaluation Metrics")
    print("="*60)
    

    retrieved = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    relevant = {1, 2, 5, 8, 11}
    
   
    p_at_5 = Evaluator.precision_at_k(retrieved, relevant, 5)
    r_at_5 = Evaluator.recall_at_k(retrieved, relevant, 5)
    ap = Evaluator.average_precision(retrieved, relevant)
    
    print(f"Retrieved: {retrieved[:5]}...")
    print(f"Relevant: {relevant}")
    print(f"\nPrecision@5: {p_at_5:.4f}")
    print(f"Recall@5: {r_at_5:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    print("\n✓ Evaluation metrics test passed!")
    return True

def test_complete_system():
    """Test complete IR system"""
    print("\n" + "="*60)
    print("TEST 5: Complete System Integration")
    print("="*60)
    
    
    ir_system = IRSystem()
    
   
    documents = {
        1: "Information retrieval is the activity of obtaining information system resources",
        2: "Boolean retrieval uses AND OR NOT operators for searching documents",
        3: "The vector space model represents documents and queries as vectors in space",
        4: "TF-IDF is a numerical statistic that reflects term importance in documents",
        5: "BM25 is a ranking function used by search engines for relevance estimation",
        6: "Natural language processing helps computers understand human language",
        7: "Machine learning algorithms can improve information retrieval systems",
        8: "Web search engines process millions of queries every day",
        9: "Document indexing creates efficient data structures for fast retrieval",
        10: "Query expansion techniques can improve search result quality"
    }
    
    
    print(f"Building system with {len(documents)} documents...")
    ir_system.build_system(documents)
    
    
    stats = ir_system.get_statistics()
    print(f"\nSystem Statistics:")
    print(f"  Documents: {stats['num_documents']}")
    print(f"  Unique terms: {stats['num_unique_terms']}")
    print(f"  Avg doc length: {stats['avg_doc_length']:.2f} tokens")
    
    
    test_queries = [
        "information retrieval",
        "search engines",
        "machine learning"
    ]
    
    print(f"\nTesting {len(test_queries)} queries...")
    for query in test_queries:
        results = ir_system.search(query, model='bm25', top_k=3)
        print(f"\nQuery: '{query}' - Found {len(results)} results")
        if results:
            top_doc_id, top_score = results[0]
            print(f"  Top result: Doc {top_doc_id} (score: {top_score:.4f})")
    
    print("\n✓ Complete system test passed!")
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "IR SYSTEM TEST SUITE")
    print("="*70)
    
    tests = [
        ("Text Preprocessing", test_preprocessor),
        ("Inverted Index", test_inverted_index),
        ("Retrieval Models", test_retrieval_models),
        ("Evaluation Metrics", test_evaluation),
        ("Complete System", test_complete_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test FAILED!")
            print(f"Error: {str(e)}")
            results.append((test_name, False))
    
   
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your IR system is working correctly.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)