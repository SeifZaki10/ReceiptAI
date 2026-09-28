# 🧾 ReceiptAI — Automated Receipt Understanding System

ReceiptAI is an end-to-end deep learning system for automatically extracting structured information from receipt images.

The system combines **computer vision, OCR, and transformer-based information extraction** to detect and recognize receipt text and extract four key fields:

- Company
- Address
- Date
- Total Amount

The project was developed as a complete receipt understanding pipeline, from dataset preparation and model evaluation to real-world deployment using Streamlit.

---

## 🚀 System Pipeline

```text
Receipt Image
      ↓
Adaptive Orientation Correction
      ↓
YOLO11 Text Detection
      ↓
Text Region Cropping
      ↓
PaddleOCR Text Recognition
      ↓
Reading Order Reconstruction
      ↓
T5 Transformer
      ↓
Structured Receipt Information
```

The orientation preprocessing automatically tests different receipt orientations and only rotates the image when there is sufficient evidence that another orientation produces better OCR results.

---

## 🧠 Models

### 1. YOLO11 — Text Detection

YOLO11 is used to detect text regions within receipt images.

The SROIE text annotations were converted from polygon coordinates into YOLO bounding-box format.

**Test Results**

| Metric | Result |
|---|---:|
| Precision | 95.05% |
| Recall | 93.72% |
| mAP@50 | 95.87% |
| mAP@50–95 | 66.57% |

---

### 2. PaddleOCR — Text Recognition

Detected text regions are cropped and passed to the English PaddleOCR recognition model:

```text
en_PP-OCRv4_mobile_rec
```

The detected regions are reconstructed into reading order from top-to-bottom and left-to-right.

The complete YOLO + PaddleOCR OCR pipeline was evaluated using Character Error Rate (CER) and Word Error Rate (WER).

| Metric | Result |
|---|---:|
| CER | 11.74% |
| WER | 41.43% |

---

### 3. T5 — Information Extraction

A pretrained **T5-small** model was fine-tuned to transform receipt text into structured information.

Example:

```text
Input:
ABC STORE
123 MAIN STREET
25/12/2018
TOTAL 19.50

Output:
company: ABC STORE
address: 123 MAIN STREET
date: 25/12/2018
total: 19.50
```

The final T5 V2 model was trained using the original SROIE training data together with additional receipt samples from the standardized receipt dataset.

### T5 V2 Test Results

The model was evaluated on the held-out SROIE test set using clean ground-truth receipt text.

| Field | Exact Match Accuracy |
|---|---:|
| Company | 89.71% |
| Address | 66.18% |
| Date | 97.06% |
| Total | 85.29% |

---

## 📊 End-to-End Evaluation

The complete pipeline was also evaluated from the original receipt image to the final structured fields:

```text
Image → YOLO → PaddleOCR → T5 → Structured Fields
```

This evaluation measures the combined effect of detection, OCR, reading-order reconstruction, and information extraction.

| Field | Exact Match | CER |
|---|---:|---:|
| Company | 33.82% | 17.26% |
| Address | 1.47% | 25.20% |
| Date | 82.35% | 12.46% |
| Total | 67.65% | 16.82% |

Exact-match evaluation is strict, especially for long fields such as addresses. Therefore, CER is also reported to measure partial textual correctness.

---

## 📂 Datasets

### SROIE 2019

The primary dataset used in the project is the **ICDAR 2019 SROIE dataset**.

It provides:

- Receipt images
- Text-region annotations
- OCR transcriptions
- Structured receipt fields

The YOLO dataset was created by converting SROIE polygon annotations into normalized bounding boxes.

### Additional T5 Training Data

To improve the generalization of the information extraction model, additional receipt data from the standardized receipt dataset was used.

The final T5 V2 training set combines:

- Original SROIE training samples
- Zenodo receipt samples
- Express receipt samples

The original SROIE validation and test sets were kept separate from the additional training data.

---

## 🗂️ Project Structure

```text
ReceiptAI/
│
├── app.py
├── pipeline.py
├── requirements.txt
├── packages.txt
├── styles.css
│
├── models/
│   ├── best (3).pt
│   └── final_t5_receipt_model_v2/
│
└── notebooks/
    └── ReceiptAI_Training_and_Evaluation.ipynb
```

---

## 💻 Running the Application

### 1. Clone the repository

```bash
git clone https://github.com/SeifZaki10/ReceiptAI.git
cd ReceiptAI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run ReceiptAI

```bash
streamlit run app.py
```

The application will start locally and allow receipt images to be uploaded and analyzed.

---

## 🌐 Streamlit Application

The Streamlit interface allows users to:

- Upload receipt images
- Automatically analyze receipts
- View extracted company, address, date, and total
- Inspect recognized OCR text
- Inspect raw T5 output
- Edit extracted information
- Maintain receipt history
- Export results as JSON or CSV

---

## 📓 Training & Evaluation Notebook

The complete training and evaluation workflow is available in:

```text
notebooks/ReceiptAI_Training_and_Evaluation.ipynb
```

The notebook includes:

- Dataset exploration
- SROIE preprocessing
- YOLO dataset conversion
- YOLO evaluation
- Adaptive orientation preprocessing
- PaddleOCR integration
- OCR CER/WER evaluation
- T5 preprocessing
- T5 training and evaluation
- External training data integration
- T5 V2 evaluation
- End-to-end system evaluation

---

## ⚠️ Limitations

The system may experience reduced accuracy when processing:

- Receipts with severe perspective distortion
- Low-quality or blurred images
- Receipt layouts significantly different from the training data
- Long address fields
- OCR errors that propagate to the information extraction model

Adaptive orientation correction improves robustness to rotated receipt images, while severe perspective correction remains outside the current scope.

---

## 🛠️ Technologies

- Python
- PyTorch
- Ultralytics YOLO11
- PaddleOCR
- Hugging Face Transformers
- T5
- OpenCV
- Streamlit
- SROIE 2019

---

## 👤 Author

**Seif Mohamed Zaki**  
Computer & Communication Engineering  
Faculty of Engineering, Alexandria University