import re
import math
import string
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import json
import time


class TextPreprocessor:
    """Handles all text preprocessing operations"""
    
    def __init__(self):
       
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Convert text to lowercase and split into tokens"""
        text = text.lower()
        
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common stopwords from token list"""
        return [token for token in tokens if token not in self.stopwords]
    
    def stem(self, token: str) -> str:
        """Simple suffix stripping stemmer"""
        
        suffixes = ['ing', 'ed', 'es', 's', 'ly', 'tion', 'ness']
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[:-len(suffix)]
        return token
    
    def preprocess(self, text: str, use_stemming=True) -> List[str]:
        """Complete preprocessing pipeline"""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if use_stemming:
            tokens = [self.stem(token) for token in tokens]
        return tokens


class InvertedIndex:
    """Inverted index structure for efficient document retrieval"""
    
    def __init__(self):
        
        self.index = defaultdict(lambda: defaultdict(int))
        
        self.doc_lengths = {}
        
        self.doc_vectors = {}
       
        self.num_docs = 0
        
        self.avg_doc_length = 0
    
    def build_index(self, documents: Dict[int, List[str]]):
        """Build inverted index from preprocessed documents"""
        self.num_docs = len(documents)
        total_length = 0
        
        for doc_id, tokens in documents.items():
            
            term_counts = Counter(tokens)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            
            for term, count in term_counts.items():
                self.index[term][doc_id] = count
            
            
            self.doc_vectors[doc_id] = term_counts
        
        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0
    
    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """Get frequency of term in document"""
        return self.index[term].get(doc_id, 0)
    
    def get_document_frequency(self, term: str) -> int:
        """Get number of documents containing term"""
        return len(self.index[term])
    
    def get_posting_list(self, term: str) -> Dict[int, int]:
        """Get all documents containing term with frequencies"""
        return dict(self.index[term])


class RetrievalModels:
    """Implementation of various retrieval models"""
    
    def __init__(self, inverted_index: InvertedIndex):
        self.index = inverted_index
    
    def tf_idf_score(self, query_terms: List[str], doc_id: int) -> float:
        """Calculate TF-IDF score for document given query"""
        score = 0.0
        N = self.index.num_docs
        
        for term in query_terms:
            tf = self.index.get_term_frequency(term, doc_id)
            if tf > 0:
                df = self.index.get_document_frequency(term)
                idf = math.log(N / df) if df > 0 else 0
                
                score += (1 + math.log(tf)) * idf
        
        return score
    
    def bm25_score(self, query_terms: List[str], doc_id: int, 
                   k1: float = 1.5, b: float = 0.75) -> float:
        """
        Calculate BM25 score for document given query
        k1: term frequency saturation parameter (typically 1.2-2.0)
        b: length normalization parameter (typically 0.75)
        """
        score = 0.0
        N = self.index.num_docs
        avgdl = self.index.avg_doc_length
        dl = self.index.doc_lengths.get(doc_id, 0)
        
        for term in query_terms:
            tf = self.index.get_term_frequency(term, doc_id)
            if tf > 0:
                df = self.index.get_document_frequency(term)
                if df > 0:
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (dl / avgdl))
                    score += idf * (numerator / denominator)
        
        return score
    
    def boolean_retrieval(self, query_terms: List[str]) -> Set[int]:
        """Simple AND boolean retrieval"""
        if not query_terms:
            return set()
        
        
        result = set(self.index.get_posting_list(query_terms[0]).keys())
        
        
        for term in query_terms[1:]:
            result &= set(self.index.get_posting_list(term).keys())
        
        return result
    
    def cosine_similarity(self, query_terms: List[str], doc_id: int) -> float:
        """Calculate cosine similarity between query and document"""
        query_vec = Counter(query_terms)
        doc_vec = self.index.doc_vectors.get(doc_id, {})
        
        
        dot_product = sum(query_vec[term] * doc_vec.get(term, 0) 
                         for term in query_vec)
        
        
        query_magnitude = math.sqrt(sum(count ** 2 for count in query_vec.values()))
        doc_magnitude = math.sqrt(sum(count ** 2 for count in doc_vec.values()))
        
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0
        
        return dot_product / (query_magnitude * doc_magnitude)


class IRSystem:
    """Complete Information Retrieval System"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.inverted_index = InvertedIndex()
        self.retrieval_models = None
        self.documents = {}  
        self.preprocessed_docs = {}  
    
    def load_cisi_documents(self, filepath: str) -> Dict[int, str]:
        """Load CISI dataset documents"""
        documents = {}
        current_doc_id = None
        current_text = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('.I'):
                        
                        if current_doc_id is not None:
                            documents[current_doc_id] = ' '.join(current_text)
                        
                        current_doc_id = int(line.split()[1])
                        current_text = []
                    elif line.startswith('.T') or line.startswith('.W') or line.startswith('.A') or line.startswith('.B'):
                        continue  
                    elif line and not line.startswith('.'):
                        current_text.append(line)
                
                
                if current_doc_id is not None:
                    documents[current_doc_id] = ' '.join(current_text)
        
        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
            return {}
        
        return documents
    
    def load_cisi_queries(self, filepath: str) -> Dict[int, str]:
        """Load CISI dataset queries"""
        queries = {}
        current_query_id = None
        current_text = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('.I'):
                        if current_query_id is not None:
                            queries[current_query_id] = ' '.join(current_text)
                        current_query_id = int(line.split()[1])
                        current_text = []
                    elif line.startswith('.W'):
                        continue
                    elif line and not line.startswith('.'):
                        current_text.append(line)
                
                if current_query_id is not None:
                    queries[current_query_id] = ' '.join(current_text)
        
        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
            return {}
        
        return queries
    
    def build_system(self, documents: Dict[int, str]):
        """Build the complete IR system"""
        print("Building Information Retrieval System...")
        print(f"Processing {len(documents)} documents...")
        
        self.documents = documents
        
        
        for doc_id, text in documents.items():
            self.preprocessed_docs[doc_id] = self.preprocessor.preprocess(text)
        
        
        self.inverted_index.build_index(self.preprocessed_docs)
        
        
        self.retrieval_models = RetrievalModels(self.inverted_index)
        
        print("System built successfully!")
        print(f"Index contains {len(self.inverted_index.index)} unique terms")
        print(f"Average document length: {self.inverted_index.avg_doc_length:.2f} tokens")
    
    def search(self, query: str, model: str = 'bm25', top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for documents matching query
        
        Args:
            query: Search query string
            model: Retrieval model ('tf-idf', 'bm25', 'cosine', 'boolean')
            top_k: Number of results to return
        
        Returns:
            List of (doc_id, score) tuples
        """
        # Preprocess query
        query_terms = self.preprocessor.preprocess(query)
        
        if not query_terms:
            return []
        
        results = []
        
        if model == 'boolean':
            doc_ids = self.retrieval_models.boolean_retrieval(query_terms)
            results = [(doc_id, 1.0) for doc_id in doc_ids]
        else:
            
            for doc_id in self.documents.keys():
                if model == 'tf-idf':
                    score = self.retrieval_models.tf_idf_score(query_terms, doc_id)
                elif model == 'bm25':
                    score = self.retrieval_models.bm25_score(query_terms, doc_id)
                elif model == 'cosine':
                    score = self.retrieval_models.cosine_similarity(query_terms, doc_id)
                else:
                    raise ValueError(f"Unknown model: {model}")
                
                if score > 0:
                    results.append((doc_id, score))
        
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_document(self, doc_id: int) -> str:
        """Retrieve original document text"""
        return self.documents.get(doc_id, "Document not found")
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            'num_documents': self.inverted_index.num_docs,
            'num_unique_terms': len(self.inverted_index.index),
            'avg_doc_length': self.inverted_index.avg_doc_length,
            'total_tokens': sum(self.inverted_index.doc_lengths.values())
        }


class Evaluator:
    """Evaluation metrics for IR system"""
    
    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate Precision@K"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & relevant)
        return relevant_retrieved / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate Recall@K"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & relevant)
        return relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0
    
    @staticmethod
    def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
        """Calculate Average Precision"""
        if not relevant:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                relevant_count += 1
                precision_sum += relevant_count / i
        
        return precision_sum / len(relevant) if len(relevant) > 0 else 0.0
    
    @staticmethod
    def mean_average_precision(query_results: Dict[int, List[int]], 
                               relevance_judgments: Dict[int, Set[int]]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        aps = []
        for query_id, retrieved in query_results.items():
            relevant = relevance_judgments.get(query_id, set())
            ap = Evaluator.average_precision(retrieved, relevant)
            aps.append(ap)
        
        return sum(aps) / len(aps) if aps else 0.0


def main():
    """Main function to demonstrate the IR system"""
    print("=" * 60)
    print("INFORMATION RETRIEVAL SYSTEM - CISI Dataset")
    print("=" * 60)
    
    
    ir_system = IRSystem()
    
    
    sample_docs = {
        1: "Information retrieval is the process of obtaining information system resources",
        2: "Boolean retrieval uses AND OR NOT operators for document searching",
        3: "Vector space model represents documents and queries as vectors",
        4: "TF-IDF is a numerical statistic for information retrieval",
        5: "BM25 is a ranking function used by search engines to estimate relevance"
    }
    
    print("\nUsing sample documents for demonstration...")
    ir_system.build_system(sample_docs)
    
    
    stats = ir_system.get_statistics()
    print("\n" + "=" * 60)
    print("SYSTEM STATISTICS")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    
    print("\n" + "=" * 60)
    print("SEARCH EXAMPLES")
    print("=" * 60)
    
    test_queries = [
        "information retrieval",
        "boolean search",
        "vector space model"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        
        for model in ['bm25', 'tf-idf', 'cosine']:
            results = ir_system.search(query, model=model, top_k=3)
            print(f"\nModel: {model.upper()}")
            if results:
                for rank, (doc_id, score) in enumerate(results, 1):
                    doc_preview = ir_system.get_document(doc_id)[:80] + "..."
                    print(f"  {rank}. Doc {doc_id} (Score: {score:.4f})")
                    print(f"     {doc_preview}")
            else:
                print("  No results found")
    
    print("\n" + "=" * 60)
    print("System demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()