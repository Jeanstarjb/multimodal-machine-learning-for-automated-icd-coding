# Multimodal Machine Learning for Automated ICD Coding

This repository provides an implementation of the research paper **"Multimodal Machine Learning for Automated ICD Coding"** by Keyang Xu et al. The study introduces a multimodal machine learning framework to predict ICD-10 diagnostic codes by combining insights from unstructured text, semi-structured text, and structured tabular data. The model leverages an ensemble approach to integrate predictions from modality-specific models, ensuring robust and interpretable results.

---

## Core Concept

ICD-10 (International Classification of Diseases, 10th Edition) coding is a critical task in healthcare for documenting diagnoses and procedures. However, manual ICD coding is time-consuming and error-prone. The proposed framework automates ICD-10 coding by:

1. **Multimodal Data Fusion**: Processing three modalities of data:
   - **Unstructured Text**: Clinical notes, discharge summaries, etc.
   - **Semi-Structured Text**: Key-value pairs in medical records.
   - **Structured Tabular Data**: Lab results, vitals, demographics, etc.

2. **Modality-Specific Models**:
   - Each data modality is handled by a specialized machine learning model tailored to its structure and characteristics.

3. **Ensemble Method**:
   - Predictions from all modality-specific models are combined to generate the final ICD-10 codes.

4. **Explainability**:
   - The model extracts key evidence (e.g., text spans, important features) to explain predictions, enhancing trust and interpretability.

---

## Repository Contents

This repository contains the Python/PyTorch implementation of the multimodal machine learning framework described in the paper. Below is an overview of the key components:

### 1. **Data Preprocessing**
   - Scripts to process the **MIMIC-III** dataset, including:
     - Tokenizing unstructured text.
     - Parsing semi-structured text.
     - Normalizing and encoding structured tabular data.
   - Handles missing data and ensures modality alignment.

### 2. **Modality-Specific Models**
   - **Text Model**: A deep learning model (e.g., BiLSTM, TextCNN) for processing unstructured clinical notes.
   - **Semi-Structured Model**: A specialized architecture tailored for key-value pairs or categorical text data.
   - **Tabular Model**: Gradient boosting or neural network models optimized for structured numerical data.

### 3. **Ensemble Model**
   - Combines individual modality predictions using techniques such as:
     - Weighted averaging.
     - Stacking with a meta-learner.

### 4. **Explainability Module**
   - Extracts key evidence for ICD code predictions:
     - Highlights important text spans in clinical notes.
     - Identifies influential features in structured data.

### 5. **Evaluation Metrics**
   - Implements metrics for assessing model performance:
     - **Micro-F1 Score**: Measures overall classification accuracy.
     - **Micro-AUC**: Evaluates ranking performance.
     - **Jaccard Similarity Coefficient (JSC)**: Assesses the overlap between predicted and true codes.
   - Compares results with baseline models like TF-IDF and Text-CNN.

---

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- Access to the **MIMIC-III** dataset (requires proper credentials due to privacy regulations).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multimodal-icd-coding.git
   cd multimodal-icd-coding
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the MIMIC-III dataset:
   - Download and preprocess the dataset using the provided scripts in `data/`.

### Usage
1. **Preprocess the Data**:
   ```bash
   python preprocess.py --data_path /path/to/mimic-iii
   ```

2. **Train Modality-Specific Models**:
   ```bash
   python train.py --modality text --epochs 10
   python train.py --modality semi_structured --epochs 10
   python train.py --modality tabular --epochs 10
   ```

3. **Run Ensemble Model**:
   ```bash
   python ensemble.py --models text_model.pt semi_structured_model.pt tabular_model.pt
   ```

4. **Evaluate the Model**:
   ```bash
   python evaluate.py --ensemble_model ensemble_model.pt
   ```

---

## Results

The performance of the proposed multimodal model on the **MIMIC-III** dataset is summarized below:

| Model               | Micro-F1 | Micro-AUC | JSC (Text) | JSC (Tabular) |
|---------------------|----------|-----------|------------|---------------|
| TF-IDF              | 0.6721   | 0.7879    | -          | -             |
| Text-CNN            | 0.6569   | 0.9235    | -          | -             |
| **Proposed Model**  | **0.7633** | **0.9541** | 0.1806     | 0.3105        |
| Human Physicians    | -        | -         | 0.2780     | 0.5002        |

---

## Key Features
- **Multimodal Integration**: Combines insights from diverse data sources.
- **Explainability**: Highlights key evidence for predictions.
- **High Performance**: Outperforms baseline models significantly in both accuracy and interpretability.

---

## References
- **Paper**: [Multimodal Machine Learning for Automated ICD Coding](https://arxiv.org/pdf/1810.13348v4)
- **Dataset**: [MIMIC-III Database](https://physionet.org/content/mimiciii/1.4/)

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for discussion.

---

## Acknowledgments
The implementation is based on the methods and findings described in the research paper by Keyang Xu et al. Special thanks to the authors for their work on advancing multimodal machine learning in the healthcare domain.