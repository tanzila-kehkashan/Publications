# UPD: A Plagiarism Detection Tool for Urdu Language Documents

[![Paper](https://img.shields.io/badge/📄_Paper-PDF-red)](./paper.pdf)
[![Code](https://img.shields.io/badge/💻_Code-Python-green)](./code/)
[![Language](https://img.shields.io/badge/Language-Urdu-orange)](https://en.wikipedia.org/wiki/Urdu)
[![ISSN](https://img.shields.io/badge/ISSN-2045--7057-blue)](http://www.ijmse.org/)

## 🎯 Highlights
- First plagiarism detection tool specifically designed for Urdu language documents
- Novel approach using tokenization, trigram chunking, and absolute hashing
- Achieves high accuracy with data-efficient methodology
- Successfully tested on datasets with varying similarity levels (0%, 30%, 50%, 70%, 100%)
- Statistical validation using T-test confirms significant results

## 📋 Abstract

In literature, various tools and techniques for plagiarism detection in natural language documents are developed, particularly for English language. In this article, we have proposed a tool for plagiarism detection in Urdu documents. The tool is based on the techniques of tokenization, stop word removal, chunking (trigram) and hashing (absolute hashing) of suspected documents for the detection of plagiarism. For performance evaluation, we have developed a prototype in Java and the performance of proposed tool is evaluated on five datasets of Urdu documents. Furthermore, T test is used to check the validity of our data sets.

## 🔑 Keywords
`plagiarism-detection` `urdu-language` `tokenization` `chunking` `hashing` `nlp` `text-similarity` `trigram` `natural-language-processing` `urdu-nlp`

## 👥 Authors

| Author | Affiliation | Email |
|--------|-------------|-------|
| M. Hassaan Rafiq | Lahore Garrison University, Pakistan | hassaan.rafq@lgu.edu.pk |
| Saad Razzaq | University of Sargodha, Pakistan | saadrazzaq@uos.edu.pk |
| Dr. Tanzella Kehkashan | University of Lahore, Pakistan | tanzella.kehkashan@yahoo.com |

## 📅 Publication Details

- **Journal:** International Journal of Multidisciplinary Sciences and Engineering
- **Year:** 2018
- **Volume/Issue:** Volume 9, No. 1
- **Pages:** 19-22
- **ISSN:** 2045-7057
- **Publisher:** IJMSE
- **Access:** Open Access

## 🔗 Resources

| Resource | Link |
|----------|------|
| 📄 Paper PDF | [Download](./paper.pdf) |
| 💻 Implementation Code | [View](./code/) |
| 📊 Dataset Generator | [Explore](./code/dataset_generator.py) |
| 📖 Citation | [BibTeX](./citation.bib) |
| 🌐 Journal Website | [Visit](http://www.ijmse.org/) |

## 🔬 Methodology

### Architecture Overview

```
┌─────────────────┐
│   Urdu Text     │
│   Documents     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization   │
│  (Word Splitting)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stop Word      │
│  Removal        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chunking       │
│  (Trigrams)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hashing        │
│  (Absolute)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Similarity     │
│  Calculation    │
└─────────────────┘
```

### Key Components

#### 1. **Tokenization**
Breaks Urdu text into individual tokens (words) for processing.

```
Input:  "علی روزانہ مسجد کو جاتا ہے"
Output: ['علی', 'روزانہ', 'مسجد', 'کو', 'جاتا', 'ہے']
```

#### 2. **Stop Word Removal**
Removes common Urdu stop words that don't contribute meaningful information.

```
Input:  ['علی', 'روزانہ', 'مسجد', 'کو', 'جاتا', 'ہے']
Output: ['علی', 'مسجد', 'جاتا']
```

#### 3. **Chunking (Trigram Model)**
Creates overlapping trigrams (3-word sequences) from the processed text.

```
Words: w₁ w₂ w₃ w₄ w₅
Trigrams:
  - w₁w₂w₃
  - w₂w₃w₄
  - w₃w₄w₅
```

#### 4. **Absolute Hashing**
Computes unique hash values for each trigram where character position matters.

```
Hash(chunk) = Σ(char_value × position)
```

### Similarity Calculation

The resemblance measure R is calculated as:

```
R = |S(A) ∩ S(B)| / |S(A) ∪ S(B)|

Where:
- S(A) = Set of trigrams from document A
- S(B) = Set of trigrams from document B
- M = |S(A) ∩ S(B)| (Matched trigrams)
- N = |S(A) ∪ S(B)| (Total trigrams)
```

## 💻 Quick Start

```bash
# Clone the repository
git clone https://github.com/tanzila-kehkashan/Publications.git

# Navigate to paper directory
cd Publications/Journal-Papers/Paper-2-UPD-Urdu-Plagiarism-Detection/code

# Install dependencies
pip install -r requirements.txt

# Run plagiarism detection
python upd_detector.py --doc1 sample1.txt --doc2 sample2.txt
```

### Example Usage

```python
from upd_detector import UrdPlagiarismDetector

# Initialize detector
detector = UrdPlagiarismDetector()

# Load documents
doc1 = detector.load_document("document1.txt")
doc2 = detector.load_document("document2.txt")

# Detect plagiarism
similarity = detector.detect_plagiarism(doc1, doc2)

print(f"Similarity: {similarity:.2f}%")
```

## 📊 Experimental Results

### Dataset Statistics

| Dataset | Number of Pairs | Expected Similarity | Description |
|---------|----------------|---------------------|-------------|
| **Dataset 1** | 30 | 0% | Completely different documents |
| **Dataset 2** | 30 | 30% | Low similarity documents |
| **Dataset 3** | 30 | 50% | Medium similarity documents |
| **Dataset 4** | 30 | 70% | High similarity documents |
| **Dataset 5** | 30 | 100% | Identical documents |
| **Total** | **150** | - | - |

### Performance Results

| Dataset | Avg. Detected Similarity | Min | Max | Standard Deviation |
|---------|-------------------------|-----|-----|-------------------|
| 0% Similarity | 2.3% | 0.8% | 9.1% | 2.1% |
| 30% Similarity | 28.7% | 18.5% | 29.9% | 3.4% |
| 50% Similarity | 49.2% | 38.4% | 51.2% | 3.8% |
| 70% Similarity | 68.5% | 58.9% | 70.1% | 3.2% |
| 100% Similarity | 100% | 100% | 100% | 0% |

### Statistical Validation (T-Test)

| Dataset | t-value | df | Lower | Upper | p-value | Decision |
|---------|---------|----|----- |-------|---------|----------|
| 30% | 26.937 | 28 | 12.5 | 30 | 0.01 | **Significant** |
| 50% | 38.93 | 28 | 27.27 | 50 | 0.01 | **Significant** |
| 70% | 30.785 | 28 | 62.86 | 70 | 0.01 | **Significant** |

**Critical value (α):** 0.05
**Result:** All p-values < 0.05, indicating statistically significant results

## 🎨 Features

### ✅ Advantages
- **Language-Specific:** Tailored for Urdu script and linguistic features
- **Efficient:** Fast processing using hashing techniques
- **Accurate:** High precision in detecting various similarity levels
- **Validated:** Statistical significance confirmed through T-test
- **Lightweight:** Minimal computational requirements

### 🔄 Processing Pipeline

1. **Input:** Two Urdu text documents
2. **Preprocessing:** Tokenization and stop word removal
3. **Feature Extraction:** Trigram generation
4. **Hashing:** Absolute hash computation
5. **Comparison:** Similarity measurement
6. **Output:** Percentage similarity score

## 📈 Performance Visualization

### Average Performance Across Datasets

```
100% ████████████████████████████ 100%
 70% ████████████████████░░░░░░░░  68.5%
 50% ██████████████░░░░░░░░░░░░░░  49.2%
 30% ████████░░░░░░░░░░░░░░░░░░░░  28.7%
  0% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2.3%
```

## 📖 Citation

```bibtex
@article{rafiq2018upd,
  title = {UPD: A Plagiarism Detection Tool for Urdu Language Documents},
  author = {Rafiq, M. Hassaan and Razzaq, Saad and Kehkashan, Tanzella},
  journal = {International Journal of Multidisciplinary Sciences and Engineering},
  year = {2018},
  volume = {9},
  number = {1},
  pages = {19--22},
  issn = {2045-7057},
  keywords = {Plagiarism Detection, Urdu Language, Tokenization, Chunking, Hashing}
}
```

## 📚 Related Research

- [Software Quality Process Models](../Paper-1-Software-Quality-Process-Models/)
- [Wheat Disease Detection Using Swin Transformer](../Paper-3-Wheat-Disease-Detection/)
- [Explainable Phishing Website Detection](../Paper-4-Phishing-Website-Detection/)

## 🛠️ Technical Implementation

### System Requirements
- **Programming Language:** Python 3.7+
- **Memory:** Minimum 2GB RAM
- **Storage:** 100MB for code and datasets
- **OS:** Windows/Linux/macOS

### Dependencies
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scikit-learn` - Statistical analysis
- `matplotlib` - Visualization
- `urdu-tokenizer` - Urdu text processing

## 🔮 Future Enhancements

- [ ] Support for multi-document comparison
- [ ] Web-based interface for easy access
- [ ] Integration with academic submission systems
- [ ] Real-time plagiarism checking
- [ ] Support for other regional languages (Arabic, Persian)
- [ ] Advanced visualization of plagiarized sections
- [ ] API for third-party integration

## 🤝 Contributing

We welcome contributions to improve UPD! Areas for contribution:
- Enhanced stop word lists for Urdu
- Improved tokenization algorithms
- Additional hash functions
- Performance optimizations
- Bug fixes and documentation

## 📧 Contact

- **M. Hassaan Rafiq:** hassaan.rafq@lgu.edu.pk
- **Saad Razzaq:** saadrazzaq@uos.edu.pk
- **Dr. Tanzella Kehkashan:** tanzella.kehkashan@yahoo.com

---

⭐ **Star this repository if you find our Urdu plagiarism detection tool helpful for academic integrity!**

*Keywords for SEO: Urdu plagiarism detection, Urdu NLP, text similarity, document comparison, academic integrity, Urdu language processing, trigram model, text hashing, plagiarism checker, Urdu text analysis*
