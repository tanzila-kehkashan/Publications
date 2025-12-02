"""
UPD: Urdu Plagiarism Detection Tool
Authors: M. Hassaan Rafiq, Saad Razzaq, Tanzella Kehkashan
Paper: UPD: A Plagiarism Detection Tool for Urdu Language Documents
Journal: International Journal of Multidisciplinary Sciences and Engineering (2018)

This implementation provides plagiarism detection for Urdu language documents
using tokenization, stop word removal, trigram chunking, and absolute hashing.
"""

import re
from typing import List, Set, Tuple, Dict
import hashlib
from collections import Counter


class UrduPlagiarismDetector:
    """Main class for detecting plagiarism in Urdu documents"""

    def __init__(self):
        self.stop_words = self._load_urdu_stop_words()
        self.chunk_size = 3  # Trigram model (n=3)

    def _load_urdu_stop_words(self) -> Set[str]:
        """Load common Urdu stop words"""
        # Common Urdu stop words
        stop_words = {
            'کا', 'کی', 'کے', 'کو', 'نے', 'میں', 'سے', 'پر',
            'اور', 'یہ', 'وہ', 'ہے', 'ہیں', 'تھا', 'تھی', 'تھے',
            'گا', 'گی', 'گے', 'ہو', 'ہوں', 'ہوئی', 'ہوئے', 'ہوا',
            'کر', 'کرنے', 'کیا', 'کیوں', 'کہ', 'جو', 'جب', 'جہاں',
            'ایک', 'دو', 'تین', 'یا', 'بھی', 'نہیں', 'اس', 'ان',
            'تو', 'ہی', 'ابھی', 'لیکن', 'مگر', 'اگر', 'تاکہ'
        }
        return stop_words

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Urdu text into words

        Args:
            text: Urdu text string

        Returns:
            List of tokens (words)
        """
        # Remove punctuation and extra whitespace
        text = re.sub(r'[۔؛،؍٪٫٬۝]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Split into words
        tokens = text.split()

        return tokens

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove Urdu stop words from token list

        Args:
            tokens: List of word tokens

        Returns:
            Filtered list without stop words
        """
        filtered_tokens = [
            token for token in tokens
            if token not in self.stop_words
        ]
        return filtered_tokens

    def create_trigrams(self, tokens: List[str]) -> List[Tuple[str, str, str]]:
        """
        Create trigrams (3-word chunks) from token list

        Args:
            tokens: List of filtered tokens

        Returns:
            List of trigrams as tuples
        """
        if len(tokens) < self.chunk_size:
            return [(tokens[0], tokens[1] if len(tokens) > 1 else '',
                    tokens[2] if len(tokens) > 2 else '')]

        trigrams = []
        for i in range(len(tokens) - self.chunk_size + 1):
            trigram = tuple(tokens[i:i + self.chunk_size])
            trigrams.append(trigram)

        return trigrams

    def absolute_hash(self, trigram: Tuple[str, str, str]) -> int:
        """
        Compute absolute hash value for a trigram
        Position of character matters in this hash function

        Args:
            trigram: Tuple of three words

        Returns:
            Integer hash value
        """
        hash_value = 0
        position_multiplier = 1

        for word in trigram:
            for char in word:
                # Get Unicode code point of character
                char_value = ord(char)
                # Multiply by position for absolute hashing
                hash_value += char_value * position_multiplier
                position_multiplier += 1

        return hash_value

    def compute_fingerprint(self, trigrams: List[Tuple[str, str, str]]) -> Set[int]:
        """
        Compute fingerprint set from trigrams using absolute hashing

        Args:
            trigrams: List of trigrams

        Returns:
            Set of hash values (fingerprint)
        """
        fingerprint = set()
        for trigram in trigrams:
            hash_val = self.absolute_hash(trigram)
            fingerprint.add(hash_val)

        return fingerprint

    def calculate_resemblance(self, fp_a: Set[int], fp_b: Set[int]) -> float:
        """
        Calculate resemblance measure R between two fingerprints

        R = |S(A) ∩ S(B)| / |S(A) ∪ S(B)|

        Args:
            fp_a: Fingerprint set of document A
            fp_b: Fingerprint set of document B

        Returns:
            Resemblance value between 0 and 1
        """
        if len(fp_a) == 0 and len(fp_b) == 0:
            return 1.0  # Both empty, considered identical

        if len(fp_a) == 0 or len(fp_b) == 0:
            return 0.0  # One empty, no similarity

        # Intersection: matched trigrams
        matched = fp_a & fp_b
        M = len(matched)

        # Union: total unique trigrams
        total = fp_a | fp_b
        N = len(total)

        # Resemblance measure
        R = M / N if N > 0 else 0.0

        return R

    def detect_plagiarism(self, doc1: str, doc2: str) -> Dict:
        """
        Main method to detect plagiarism between two Urdu documents

        Args:
            doc1: First Urdu document text
            doc2: Second Urdu document text

        Returns:
            Dictionary with plagiarism detection results
        """
        # Step 1: Tokenization
        tokens1 = self.tokenize(doc1)
        tokens2 = self.tokenize(doc2)

        # Step 2: Stop word removal
        filtered1 = self.remove_stop_words(tokens1)
        filtered2 = self.remove_stop_words(tokens2)

        # Step 3: Create trigrams
        trigrams1 = self.create_trigrams(filtered1)
        trigrams2 = self.create_trigrams(filtered2)

        # Step 4: Compute fingerprints using absolute hashing
        fingerprint1 = self.compute_fingerprint(trigrams1)
        fingerprint2 = self.compute_fingerprint(trigrams2)

        # Step 5: Calculate resemblance
        similarity = self.calculate_resemblance(fingerprint1, fingerprint2)

        # Prepare results
        results = {
            'similarity_percentage': similarity * 100,
            'doc1_tokens': len(tokens1),
            'doc2_tokens': len(tokens2),
            'doc1_filtered_tokens': len(filtered1),
            'doc2_filtered_tokens': len(filtered2),
            'doc1_trigrams': len(trigrams1),
            'doc2_trigrams': len(trigrams2),
            'doc1_fingerprint_size': len(fingerprint1),
            'doc2_fingerprint_size': len(fingerprint2),
            'matched_trigrams': len(fingerprint1 & fingerprint2),
            'total_unique_trigrams': len(fingerprint1 | fingerprint2)
        }

        return results

    def load_document(self, file_path: str) -> str:
        """
        Load Urdu document from file

        Args:
            file_path: Path to text file

        Returns:
            Document content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"Document not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading document: {str(e)}")


def print_results(results: Dict):
    """Pretty print plagiarism detection results"""
    print("\n" + "=" * 60)
    print("URDU PLAGIARISM DETECTION RESULTS")
    print("=" * 60)

    print(f"\n{'Similarity:':<30} {results['similarity_percentage']:.2f}%")

    print(f"\n{'DOCUMENT 1 STATISTICS':^60}")
    print("-" * 60)
    print(f"{'Total Tokens:':<30} {results['doc1_tokens']}")
    print(f"{'Filtered Tokens:':<30} {results['doc1_filtered_tokens']}")
    print(f"{'Trigrams Generated:':<30} {results['doc1_trigrams']}")
    print(f"{'Fingerprint Size:':<30} {results['doc1_fingerprint_size']}")

    print(f"\n{'DOCUMENT 2 STATISTICS':^60}")
    print("-" * 60)
    print(f"{'Total Tokens:':<30} {results['doc2_tokens']}")
    print(f"{'Filtered Tokens:':<30} {results['doc2_filtered_tokens']}")
    print(f"{'Trigrams Generated:':<30} {results['doc2_trigrams']}")
    print(f"{'Fingerprint Size:':<30} {results['doc2_fingerprint_size']}")

    print(f"\n{'COMPARISON METRICS':^60}")
    print("-" * 60)
    print(f"{'Matched Trigrams (M):':<30} {results['matched_trigrams']}")
    print(f"{'Total Unique Trigrams (N):':<30} {results['total_unique_trigrams']}")

    print("\n" + "=" * 60)

    # Interpretation
    similarity = results['similarity_percentage']
    if similarity >= 70:
        print("⚠️  HIGH SIMILARITY - Potential plagiarism detected!")
    elif similarity >= 30:
        print("⚡ MODERATE SIMILARITY - Review recommended")
    else:
        print("✅ LOW SIMILARITY - Documents appear to be original")

    print("=" * 60 + "\n")


def main():
    """Example usage demonstration"""

    print("UPD: Urdu Plagiarism Detection Tool")
    print("=" * 60)

    # Initialize detector
    detector = UrduPlagiarismDetector()

    # Example Urdu documents
    doc1 = """
    پاکستان جنوبی ایشیا میں واقع ایک خوبصورت ملک ہے۔ یہاں کی ثقافت بہت
    متنوع ہے اور لوگ مہمان نواز ہیں۔ پاکستان میں بہت سے تاریخی مقامات
    موجود ہیں۔ یہاں کی قدرتی خوبصورتی دنیا بھر میں مشہور ہے۔
    """

    doc2 = """
    پاکستان جنوبی ایشیا میں واقع ایک پرفضا ملک ہے۔ یہاں کی روایات بہت
    متنوع ہیں اور شہری مہمان نواز ہیں۔ پاکستان میں کئی تاریخی مقامات
    ملتے ہیں۔ یہاں کی فطری حسن عالمی سطح پر معروف ہے۔
    """

    doc3 = """
    تعلیم ہر معاشرے کی ترقی کی بنیاد ہے۔ اچھی تعلیم سے نوجوان نسل کو
    بہتر مستقبل مل سکتا ہے۔ ہمیں اپنے تعلیمی نظام کو مزید بہتر بنانے
    کی ضرورت ہے۔ کیونکہ علم ہی قوموں کی ترقی کا راز ہے۔
    """

    # Test Case 1: High similarity
    print("\n📊 Test Case 1: Comparing similar documents")
    results1 = detector.detect_plagiarism(doc1, doc2)
    print_results(results1)

    # Test Case 2: Low similarity
    print("\n📊 Test Case 2: Comparing different documents")
    results2 = detector.detect_plagiarism(doc1, doc3)
    print_results(results2)

    # Test Case 3: Identical documents
    print("\n📊 Test Case 3: Comparing identical documents")
    results3 = detector.detect_plagiarism(doc1, doc1)
    print_results(results3)


if __name__ == "__main__":
    main()
