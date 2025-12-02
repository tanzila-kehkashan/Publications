# Data-Efficient Wheat Disease Detection Using Shifted Window Transformer

[![DOI](https://img.shields.io/badge/DOI-10.1109/TCE.2025.3582267-blue)](https://doi.org/10.1109/TCE.2025.3582267)
[![Paper](https://img.shields.io/badge/📄_Paper-PDF-red)](./paper.pdf)
[![Code](https://img.shields.io/badge/💻_Code-Python-green)](./code/)
[![IEEE](https://img.shields.io/badge/IEEE-TCE-orange)](https://ieeexplore.ieee.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.3%25-brightgreen)](./code/)

## 🎯 Highlights
- **99.3% accuracy** in wheat disease classification using Swin Transformer
- Hierarchical attention mechanism for both local and global feature extraction
- Trained on 9,346 wheat leaf images across 9 classes (8 diseases + healthy)
- Data-efficient approach suitable for resource-constrained environments
- Explainable AI using Grad-CAM visualization for model interpretability
- Addresses global food security challenges through precision agriculture

## 📋 Abstract

Wheat is an essential crop that plays a vital role in global food security, but is susceptible to a variety of diseases which have the potential to drastically decrease crop productivity. Detection of disease at an early stage and in an accurate manner is crucial to minimize crop losses. This research presents a deep learning technique based on the Shifted Window (Swin) Transformer, a powerful attention-based model that effectively captures both local and global information for enhanced classification output. Unlike conventional CNN-based methods which often face limitations in feature extraction, the Swin Transformer utilizes hierarchical attention mechanisms to improve disease detection accuracy.

## 🔑 Keywords
`wheat-disease-detection` `deep-learning` `swin-transformer` `image-classification` `precision-agriculture` `computer-vision` `food-security` `plant-pathology` `attention-mechanism` `agricultural-ai`

## 👥 Authors

| Author | Affiliation | ORCID |
|--------|-------------|-------|
| Muhammad Khubaib | University of Lahore, Pakistan | - |
| Dr. Tanzila Kehkashan | Universiti Teknologi Malaysia & University of Lahore | - |
| Dr. Maha Abdelhaq | Princess Nourah bint Abdulrahman University, Saudi Arabia | - |
| Dr. Muhammad Asghar Khan | Prince Muhammad Bin Fahd University, Saudi Arabia | - |
| Dr. Muhammad Zaman | COMSATS University Islamabad & Superior University | IEEE Member |
| Imran Ashraf | University of Lahore, Pakistan | - |
| Abdul Rehman | University of Lahore, Pakistan | - |
| Dr. Adnan Akhunzada | University of Doha for Science and Technology, Qatar | IEEE Senior Member |

## 📅 Publication Details

- **Journal:** IEEE Transactions on Consumer Electronics
- **Publisher:** IEEE
- **Year:** 2025
- **Volume/Issue:** Volume 71, No. 3
- **Pages:** 9006-9020
- **DOI:** [10.1109/TCE.2025.3582267](https://doi.org/10.1109/TCE.2025.3582267)
- **Impact Factor:** High-impact IEEE journal
- **License:** Creative Commons Attribution 4.0

## 🔗 Resources

| Resource | Link |
|----------|------|
| 📄 Paper PDF | [Download](./paper.pdf) |
| 💻 Code Implementation | [View](./code/) |
| 📊 Trained Models | [Download](./code/models/) |
| 🤗 Hugging Face Model | [Coming Soon] |
| 📖 Citation | [BibTeX](./citation.bib) |
| 🌐 IEEE Xplore | [View](https://ieeexplore.ieee.org/) |

## 🌾 Wheat Disease Classes

| Class ID | Disease Name | Pathogen Type | Characteristics |
|----------|-------------|---------------|-----------------|
| 1 | **Brown Rust** | Fungal | Brown pustules on leaves, high severity variation |
| 2 | **Crown and Root Rot** | Fungal | Affects plant base, consistent symptoms |
| 3 | **Fusarium Head Blight** | Fungal | Affects grain, less variation, mostly similar symptoms |
| 4 | **Healthy** | - | No disease symptoms, highest variability in healthy conditions |
| 5 | **Leaf Rust** | Fungal | Orange-red pustules, moderate variation in severity |
| 6 | **Loose Smut** | Fungal | Limited variation in appearance of infected plants |
| 7 | **Septoria** | Fungal | Low variation, primarily localized on disease stages |
| 8 | **Stripe Rust** | Fungal | Yellow stripes, considerable variation in manifestation |
| 9 | **Yellow Rust** | Fungal | Moderate variation in disease progression |

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Images** | 9,346 |
| **Number of Classes** | 9 (8 diseases + 1 healthy) |
| **Image Resolution** | 224×224 pixels |
| **Training Set** | 70% (6,542 images) |
| **Validation Set** | 15% (1,402 images) |
| **Test Set** | 15% (1,402 images) |
| **Data Sources** | LWDCD 2020, Wheat Disease Dataset, Wheat Leaf Dataset |

### Class Distribution

```
Healthy         ████████████████████ 2,460 images (26.3%)
Brown Rust      ██████████████ 1,256 images (13.4%)
Yellow Rust     ████████████ 1,395 images (14.9%)
Crown Root Rot  ████████ 1,021 images (10.9%)
Leaf Rust       ████████ 1,041 images (11.1%)
Loose Smut      ███████ 930 images (9.9%)
Fusarium        ████████ 607 images (6.5%)
Septoria        ████ 446 images (4.8%)
Stripe Rust     ██ 208 images (2.2%)
```

## 🏗️ Model Architecture

### Swin Transformer Overview

The Swin Transformer (Shifted Window Transformer) introduces a hierarchical structure with shifted window-based self-attention, making it highly efficient for vision tasks.

**Key Components:**

1. **Patch Embedding**
   - Input: 224×224×3 RGB images
   - Patch Size: 4×4
   - Embedding Dimension: 96

2. **Shifted Window Attention**
   - Local attention within windows
   - Cross-window connections via shifting
   - Computational complexity: O(N) instead of O(N²)

3. **Multi-Layer Perceptron (MLP)**
   - Two fully connected layers
   - GELU activation function
   - Dropout for regularization

4. **Hierarchical Structure**
   - Four stages with progressive downsampling
   - Feature dimensions: 96 → 192 → 384 → 768

5. **Classification Head**
   - Global average pooling
   - Fully connected layer
   - Softmax for 9-class output

### Mathematical Formulation

**Patch Embedding:**
```
z₀ = Flatten(X) · Wₑ
```

**Shifted Window Attention:**
```
Attention(Q, K, V) = softmax(QKᵀ/√dₖ) · V
```

**MLP Layer:**
```
y = GELU(W₁x + b₁) · W₂ + b₂
```

**Residual Connection:**
```
z_output = LayerNorm(z + f(z))
```

## 💻 Quick Start

```bash
# Clone repository
git clone https://github.com/tanzila-kehkashan/Publications.git

# Navigate to project
cd Publications/Journal-Papers/Paper-3-Wheat-Disease-Detection/code

# Install dependencies
pip install -r requirements.txt

# Run training
python train_swin_transformer.py --epochs 10 --batch-size 32 --lr 0.0001

# Run inference
python inference.py --image path/to/wheat_leaf.jpg --model checkpoints/best_model.pth
```

### Example Usage

```python
import torch
from swin_transformer import SwinTransformer
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = SwinTransformer(num_classes=9)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Predict
image = Image.open('wheat_leaf.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)

print(f"Predicted Disease: {class_names[prediction.item()]}")
```

## 📈 Performance Results

### Model Comparison

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Parameters (M) | Inference Time (ms) |
|-------|-------------|---------------|-----------|-------------|---------------|-------------------|
| **Swin Transformer (Proposed)** | **99.3** | **98.4** | **98.8** | **98.6** | 50.6 | 120 |
| Vision Transformer (ViT) | 95.5 | 94.2 | 94.8 | 94.5 | 86.5 | 150 |
| DeiT | 97.2 | 96.5 | 96.8 | 96.6 | 86.5 | 145 |
| ResNet50 | 95.0 | 94.1 | 94.5 | 94.3 | 25.6 | 80 |
| DenseNet121 | 96.8 | 95.9 | 96.2 | 96.0 | 8.0 | 95 |
| EfficientNetB0 | 96.5 | 95.7 | 96.0 | 95.8 | 5.3 | 70 |

### Training Dynamics

- **Optimizer:** Adam
- **Learning Rate:** 0.0001
- **Batch Size:** 32
- **Epochs:** 10
- **Early Stopping:** Patience of 3 epochs
- **Best Validation Accuracy:** 99.30%

### Confusion Matrix Highlights

**Best Performing Classes:**
- Healthy: 99.5% accuracy
- Brown Rust: 99.2% accuracy
- Yellow Rust: 99.0% accuracy

**Challenging Classes:**
- Stripe Rust: 96.5% (limited training samples)
- Septoria: 97.8%

## 🔬 Key Innovations

### 1. Shifted Window Mechanism
- Reduces computational complexity from quadratic to linear
- Enables cross-window connections without sacrificing local attention
- Hierarchical feature learning at multiple scales

### 2. Bayesian Hyperparameter Optimization
- Systematic tuning of learning rate, batch size, dropout rate
- Achieved optimal configuration:
  - Learning Rate: 0.0001
  - Batch Size: 32
  - Dropout: 0.3

### 3. Explainable AI Integration
- **Grad-CAM Visualization** highlights disease-affected regions
- Enhances model interpretability for agricultural experts
- Builds trust in automated disease diagnosis

### 4. Data Efficiency
- Effective performance with 9,346 images (vs. millions in typical deep learning)
- Suitable for agricultural domains with limited labeled data
- Transfer learning from ImageNet pre-training

## 🎨 Grad-CAM Visualizations

The model uses Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize which image regions contribute most to predictions:

```
Original Image → Grad-CAM Heatmap → Overlay
    [Leaf]     →   [Hot regions]  → [Highlighted disease areas]
```

**Interpretation:**
- Red regions: High contribution to disease classification
- Blue regions: Low relevance
- Validates model focuses on actual lesions, not background

## 📖 Citation

```bibtex
@article{khubaib2025wheat,
  title = {Data-Efficient Wheat Disease Detection Using Shifted Window Transformer: Enhancing Accuracy, Sustainability, and Global Food Security},
  author = {Khubaib, Muhammad and Kehkashan, Tanzila and Abdelhaq, Maha and Khan, Muhammad Asghar and Zaman, Muhammad and Ashraf, Imran and Rehman, Abdul and Akhunzada, Adnan},
  journal = {IEEE Transactions on Consumer Electronics},
  year = {2025},
  volume = {71},
  number = {3},
  pages = {9006--9020},
  doi = {10.1109/TCE.2025.3582267},
  publisher = {IEEE},
  keywords = {Wheat disease detection, Deep learning, Swin transformer, Image classification, Precision agriculture}
}
```

## 📚 Related Research

- [Software Quality Process Models](../Paper-1-Software-Quality-Process-Models/)
- [UPD: Urdu Plagiarism Detection Tool](../Paper-2-UPD-Urdu-Plagiarism-Detection/)
- [Explainable Phishing Website Detection](../Paper-4-Phishing-Website-Detection/)

## 🔮 Future Work

- [ ] Multi-crop disease detection (rice, corn, vegetables)
- [ ] Mobile app deployment for real-time field diagnosis
- [ ] Integration with UAV/drone imaging systems
- [ ] Severity grading (mild, moderate, severe)
- [ ] Multi-disease simultaneous detection
- [ ] Edge device optimization (TensorRT, ONNX)
- [ ] Federated learning for privacy-preserving collaboration

## 🌍 Impact on Global Food Security

This research contributes to:
- **Early Disease Detection:** Prevents large-scale crop losses
- **Precision Agriculture:** Targeted pesticide application reduces environmental impact
- **Resource Optimization:** Saves water, fertilizer, and labor
- **Sustainable Farming:** Supports UN Sustainable Development Goals (SDG 2: Zero Hunger)

## 🏆 Funding & Acknowledgments

This work was supported by:
- Princess Nourah bint Abdulrahman University (Project PNURSP2025R97)
- Vision and Language Computing Matrix Lab, University of Lahore

## 📧 Contact

- **Muhammad Khubaib:** (Lead Author)
- **Dr. Tanzila Kehkashan:** tanzila.kehkashan@gmail.com
- **Dr. Adnan Akhunzada:** adnan.akhunzada@udst.edu.qa

---

⭐ **Star this repository if you find our wheat disease detection research helpful for precision agriculture!**

*Keywords for SEO: wheat disease detection, Swin Transformer, deep learning agriculture, plant disease classification, precision farming, computer vision crops, AI food security, agricultural image analysis, transformer plant pathology*
