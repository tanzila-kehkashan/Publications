# Explainable Phishing Website Detection for Secure and Sustainable Cyber Infrastructure

[![DOI](https://img.shields.io/badge/DOI-10.1038/s41598--025--27984--w-blue)](https://doi.org/10.1038/s41598-025-27984-w)
[![Paper](https://img.shields.io/badge/📄_Paper-PDF-red)](./paper.pdf)
[![Code](https://img.shields.io/badge/💻_Code-Python-green)](./code/)
[![Nature](https://img.shields.io/badge/Journal-Scientific_Reports-orange)](https://www.nature.com/srep/)
[![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen)](./code/)

## 🎯 Highlights
- **97% accuracy** in phishing detection using Random Forest with SHAP explainability
- SHAP (Shapley Additive Explanations) integration for feature-level interpretability
- Comprehensive evaluation of 5 ML models: RF, SVM, DT, LR, and KNN
- Dataset: 11,055 URLs with 30 diverse features from UCI/Kaggle repository
- URL-based detection suitable for resource-constrained environments
- Addresses critical gap of explainability in cybersecurity models
- Outperforms state-of-the-art methods including CNN+LSTM and DNN approaches

## 📋 Abstract

Phishing is a social engineering attack and a type of cybercrime that is dangerously and constantly on the rise. Phishing attacks can impact various sectors, including governmental, social, financial, and individual businesses. Traditional methods of identifying phishing websites, such as blacklist and heuristic approaches, often fail to provide sufficient protection. Moreover, traditional techniques that combine URLs, webpage content, and external features are time-consuming, require substantial computing power, and are unsuitable for devices with limited resources. To overcome this issue, this research applies feature selection techniques, specifically Shapley Additive Explanations (SHAP), with each model based primarily on the URL to improve the detection process. The models, namely Support Vector Machine (SVM), Random Forest (RF), Decision Tree (DT), Logistic Regression (LR), and K-Nearest Neighbor (KNN), were trained and tested. Each model used SHAP to improve precision and interpretability by highlighting the most important features. The Random Forest model achieved 97% accuracy, offering an overall and interpretable solution for phishing detection that contributes to a safer digital environment.

## 🔑 Keywords
`phishing-detection` `machine-learning` `random-forest` `shap` `explainable-ai` `url-analysis` `cybersecurity` `feature-selection` `svm` `decision-tree` `interpretability` `cyber-infrastructure`

## 👥 Authors

| Author | Affiliation | ORCID |
|--------|-------------|-------|
| Dr. Tanzila Kehkashan | Universiti Teknologi Malaysia & University of Lahore | Equal contribution |
| Dr. Maha Abdelhaq | Princess Nourah bint Abdulrahman University, Saudi Arabia | - |
| Dr. Ahmad Sami Al-Shamayleh | Al-Ahliyya Amman University, Jordan | - |
| Nazish Huda | University of Lahore, Pakistan | Equal contribution |
| Imran Ashraf Yaseen | University of Lahore, Pakistan | Equal contribution |
| Dr. Abdelmuttlib Ibrahim Abdalla Ahmed | Omdurman Islamic University, Sudan | - |
| Dr. Adnan Akhunzada | University of Doha for Science and Technology, Qatar | - |

## 📅 Publication Details

- **Journal:** Scientific Reports (Nature Portfolio)
- **Publisher:** Springer Nature
- **Year:** 2025
- **Volume:** 15
- **Article Number:** 41751
- **DOI:** [10.1038/s41598-025-27984-w](https://doi.org/10.1038/s41598-025-27984-w)
- **Impact Factor:** High-impact open access journal
- **License:** Creative Commons Attribution 4.0 International

## 🔗 Resources

| Resource | Link |
|----------|------|
| 📄 Paper PDF | [Download](./paper.pdf) |
| 💻 Code Implementation | [View](./code/) |
| 📊 Dataset | [UCI Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites) |
| 📖 Citation | [BibTeX](./citation.bib) |
| 🌐 Scientific Reports | [View Article](https://doi.org/10.1038/s41598-025-27984-w) |

## 🌐 Phishing Attack Overview

### What is Phishing?
Phishing is a **social engineering cyber attack** where attackers trick individuals into disclosing sensitive information like usernames, passwords, and financial data.

### Impact Statistics
- **80%+** of firms experience phishing attacks annually
- Significant financial and operational consequences
- Targets: Government, banks, social media, personal users
- Distribution channels: Email, SMS, social media

### Attack Vectors
- Misleading URLs
- Malicious links in emails
- SMS phishing (smishing)
- Social media spoofing

## 📊 Dataset Details

### Source
**UCI Machine Learning Repository** - Phishing Website Detection Dataset
Originally contributed by Mohammad et al. (2012)

### Statistics

| Metric | Value |
|--------|-------|
| **Total URLs** | 11,055 |
| **Features** | 30 |
| **Classes** | 2 (Legitimate = -1, Phishing = 1) |
| **Training Set** | 80% (8,844 URLs) |
| **Testing Set** | 20% (2,211 URLs) |
| **Validation Method** | 5-fold cross-validation |

### Key Features

**URL-Based Features:**
- `UsingIP` - Using IP address instead of domain name
- `LongURL` - URL length
- `ShortURL` - Using URL shortening service
- `Symbol@` - Presence of @ symbol
- `PrefixSuffix-` - Dash in domain name

**Domain-Based Features:**
- `SubDomains` - Number of subdomains
- `HTTPS` - HTTPS usage
- `DomainRegLen` - Domain registration length
- `DNSRecording` - DNS record existence
- `AgeofDomain` - Domain age

**Web Content Features:**
- `Favicon` - Favicon loading from external domain
- `NonStdPort` - Using non-standard port
- `RequestURL` - External resources percentage
- `AnchorURL` - Anchor URL characteristics
- `IframeRedirection` - Iframe redirection usage

**External Features:**
- `WebsiteTraffic` - Website traffic ranking
- `PageRank` - Google PageRank
- `LinksPointingToPage` - Number of links pointing to page

## 🏗️ Methodology

### Architecture Overview

```
┌─────────────────────┐
│  Data Collection    │
│  (11,055 URLs)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Data Preprocessing │
│  - Cleaning         │
│  - Encoding         │
│  - Normalization    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Model Training     │
│  - Random Forest    │
│  - SVM              │
│  - Decision Tree    │
│  - Logistic Reg.    │
│  - KNN              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  SHAP Analysis      │
│  - Feature Ranking  │
│  - Interpretability │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Performance Eval.  │
│  - Accuracy: 97%    │
│  - Precision        │
│  - Recall           │
│  - F1-Score         │
└─────────────────────┘
```

### Models Evaluated

#### 1. **Random Forest (RF)**
- Ensemble of decision trees
- Majority voting mechanism
- Best performer: **97.0% accuracy**
- Formula: F(z) = (1/M) × Σ gⱼ(z)

#### 2. **Support Vector Machine (SVM)**
- High-dimensional classification
- Optimal hyperplane separation
- Decision function: D(u) = α·u + β

#### 3. **K-Nearest Neighbors (KNN)**
- K=5 nearest neighbors
- Euclidean distance metric
- Distance: δ(p,q) = √Σ(pₖ - qₖ)²

#### 4. **Decision Tree (DT)**
- Recursive dataset partitioning
- Tree-like decision structure
- Interpretable rules

#### 5. **Logistic Regression (LR)**
- Binary probability estimation
- Logistic function: Pr(y=1|u) = 1/(1 + e^-(α·u+β))

### SHAP Integration

**Shapley Additive Explanations (SHAP)** quantifies each feature's contribution to predictions:

- **Transparency**: Explains why a model flags a URL as phishing
- **Feature Ranking**: Identifies most influential features
- **Trust Building**: Enables cybersecurity professionals to understand decisions
- **Performance Boost**: Improves accuracy by 5-7% across all models

## 💻 Quick Start

```bash
# Clone repository
git clone https://github.com/tanzila-kehkashan/Publications.git

# Navigate to project
cd Publications/Journal-Papers/Paper-4-Phishing-Website-Detection/code

# Install dependencies
pip install -r requirements.txt

# Run training
python train_phishing_detector.py --model rf --use-shap

# Run inference
python detect_phishing.py --url "https://example-suspicious-site.com"
```

### Example Usage

```python
import pandas as pd
from phishing_detector import PhishingDetector
from sklearn.model_selection import train_test_split

# Initialize detector with SHAP
detector = PhishingDetector(model='rf', use_shap=True)

# Load dataset
data = pd.read_csv('phishing_dataset.csv')
X = data.drop('Result', axis=1)
y = data['Result']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
detector.train(X_train, y_train)

# Predict
predictions = detector.predict(X_test)

# Get SHAP explanations
shap_values = detector.explain_predictions(X_test)

# Evaluate
accuracy = detector.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

## 📈 Performance Results

### Model Comparison (with SHAP)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------|-------------|---------------|-----------|-------------|
| **Random Forest (RF)** | **97.0** | **97.0** | **97.6** | **97.3** |
| Decision Tree (DT) | 96.0 | 96.7 | 96.1 | 96.4 |
| K-Nearest Neighbors (KNN) | 94.3 | 94.6 | 95.1 | 94.9 |
| Support Vector Machine (SVM) | 93.5 | 92.9 | 95.5 | 94.2 |
| Logistic Regression (LR) | 93.3 | 92.9 | 95.3 | 94.1 |

### Baseline Performance (without SHAP)

| Model | Accuracy (%) | Improvement with SHAP |
|-------|-------------|----------------------|
| Random Forest | 90.0 | **+7.0%** |
| Decision Tree | 90.0 | **+6.0%** |
| KNN | 86.0 | **+8.3%** |
| SVM | 86.0 | **+7.5%** |
| Logistic Regression | 85.0 | **+8.3%** |

### Confusion Matrix Analysis

**Random Forest (Best Model):**
- True Negatives: 937
- True Positives: 1,206
- False Positives: 39
- False Negatives: 29

**Key Metrics:**
- Very low false positive rate (3.9%)
- Minimal false negatives (2.4%)
- Balanced performance across both classes

### Comparison with State-of-the-Art

| Method | Accuracy (%) | F1-Score (%) | Year |
|--------|-------------|-------------|------|
| **RF + SHAP (Proposed)** | **97.0** | **97.3** | 2025 |
| RF (Baseline) | 96.25 | 96.2 | 2023 |
| LightGBM | 95.1 | 95.3 | 2024 |
| SVM | 94.2 | 94.8 | - |
| CNN+LSTM | 93.28 | 93.29 | 2023 |
| DNN | 92.89 | 92.21 | 2023 |
| DeepSeek R1 Distill | 75.0 | 76.0 | 2025 |

## 🔬 Key Innovations

### 1. SHAP-Based Feature Explainability
- **First phishing detection system** to systematically integrate SHAP across multiple models
- Provides transparent feature contribution analysis
- Enables cybersecurity professionals to understand and trust predictions
- Identifies most impactful features: HTTPS, AnchorURL, WebsiteTraffic

### 2. URL-Only Feature Set
- **Lightweight and efficient** detection
- No need for webpage content parsing
- Suitable for **resource-constrained devices**
- Fast inference time: ~120ms per URL

### 3. Comprehensive Benchmark Study
- Systematic comparison of 5 supervised ML models
- Both with and without SHAP (ablation study)
- Comparison with deep learning and LLM-based approaches
- Demonstrates practical superiority of interpretable models

### 4. Sustainable Cyber Infrastructure
- Addresses UN Sustainable Development Goals
- Contributes to digital safety and security
- Scalable to large-scale deployments
- Open access research promoting reproducibility

## 🎨 SHAP Feature Importance

### Top Features Identified by SHAP

```
HTTPS               ████████████████████████████ (0.40)
AnchorURL           ██████████████████████░░░░░░ (0.35)
WebsiteTraffic      ████████████████░░░░░░░░░░░░ (0.25)
SubDomains          ██████████████░░░░░░░░░░░░░░ (0.22)
LinksInScriptTags   ████████████░░░░░░░░░░░░░░░░ (0.20)
PrefixSuffix        ██████████░░░░░░░░░░░░░░░░░░ (0.18)
UsingIP             ████████░░░░░░░░░░░░░░░░░░░░ (0.15)
DNSRecording        ██████░░░░░░░░░░░░░░░░░░░░░░ (0.12)
PageRank            ████░░░░░░░░░░░░░░░░░░░░░░░░ (0.08)
```

### Interpretation
- **Red regions**: Features that strongly indicate phishing
- **Blue regions**: Features suggesting legitimate website
- **SHAP values**: Quantify contribution magnitude
- **Feature ranking**: Top 15 features selected empirically

## 📖 Citation

```bibtex
@article{kehkashan2025phishing,
  title = {Explainable phishing website detection for secure and sustainable cyber infrastructure},
  author = {Kehkashan, Tanzila and Abdelhaq, Maha and Al-Shamayleh, Ahmad Sami and Huda, Nazish and Yaseen, Imran Ashraf and Ahmed, Abdelmuttlib Ibrahim Abdalla and Akhunzada, Adnan},
  journal = {Scientific Reports},
  year = {2025},
  volume = {15},
  pages = {41751},
  doi = {10.1038/s41598-025-27984-w},
  url = {https://doi.org/10.1038/s41598-025-27984-w},
  publisher = {Springer Nature},
  keywords = {Machine learning, Phishing website detection, RF, SHAP, URL}
}
```

## 📚 Related Research

- [Software Quality Process Models](../Paper-1-Software-Quality-Process-Models/)
- [UPD: Urdu Plagiarism Detection Tool](../Paper-2-UPD-Urdu-Plagiarism-Detection/)
- [Wheat Disease Detection Using Swin Transformer](../Paper-3-Wheat-Disease-Detection/)

## 🔮 Future Work

- [ ] Integration with real-time email gateways and browser plug-ins
- [ ] Expansion to multi-lingual phishing detection
- [ ] Mobile phishing detection (SMS, WhatsApp)
- [ ] Adversarial robustness testing and defense mechanisms
- [ ] Lightweight SHAP variants for edge devices
- [ ] Integration with transformer-based deep learning models
- [ ] Zero-day phishing attack detection
- [ ] Federated learning for privacy-preserving detection
- [ ] Deployment on IoT and edge computing devices

## 🌍 Impact on Cybersecurity

This research contributes to:
- **Enhanced Digital Safety**: Protects users from financial and identity theft
- **Explainable AI in Security**: Builds trust in automated detection systems
- **Resource Efficiency**: Enables deployment on low-power devices
- **Sustainable Cyber Infrastructure**: Scalable and maintainable security solutions
- **Open Science**: Promotes reproducibility and collaboration

## 🛠️ Technical Implementation

### System Requirements
- **Programming Language:** Python 3.8+
- **Memory:** Minimum 4GB RAM
- **Storage:** 500MB for code and datasets
- **OS:** Windows/Linux/macOS
- **GPU:** Optional (for large-scale training)

### Key Dependencies
- `scikit-learn` - Machine learning models
- `shap` - Explainability framework
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `seaborn` - Advanced plotting

### Computational Environment
- **OS:** Windows 10
- **Processor:** 2.4 GHz
- **GPU:** Google Colab GPU (for experiments)
- **Training Time:** ~10 minutes for RF model
- **Inference Time:** ~120ms per URL

## 🏆 Funding & Acknowledgments

This work was supported by:
- **Princess Nourah bint Abdulrahman University** (Project PNURSP2025R97)
- **VLCMatrix Lab**, University of Lahore

## 📧 Contact

- **Dr. Tanzila Kehkashan:** tanzila.kehkashan@gmail.com
- **Imran Ashraf Yaseen:** imranashraf.yaseen@gmail.com
- **Dr. Abdelmuttlib Ibrahim Abdalla Ahmed:** abdelmuttlib@oiu.edu.sd
- **Dr. Adnan Akhunzada:** adnan.akhunzada@udst.edu.qa

---

⭐ **Star this repository if you find our explainable phishing detection research helpful for cybersecurity!**

*Keywords for SEO: phishing detection, machine learning cybersecurity, SHAP explainability, random forest classifier, URL analysis, interpretable AI, cyber threat detection, secure cyber infrastructure, feature importance, supervised learning*
