# 📦 ShipmentSure: On-Time Delivery Predictor

An AI-powered machine learning application that predicts whether e-commerce shipments will arrive on time or experience delays. Built with **XGBoost**, **scikit-learn**, and **Streamlit** for real-time predictions.

## 🎯 Overview

ShipmentSure addresses a critical challenge in e-commerce logistics: predicting shipment delays before they happen. By analyzing shipment characteristics (warehouse location, shipping mode, customer data, and product metrics), the model can forecast delivery status with high accuracy.

The application provides:
- **Real-time predictions** via an interactive Streamlit web interface
- **Probability scores** for both on-time and delayed deliveries
- **Optimized decision thresholds** based on precision-recall analysis
- **Feature engineering** including cost-to-weight ratios for enhanced model performance

## ✨ Key Features

- 🎯 **Binary Classification**: Predicts "On Time" vs "Not On Time" delivery status
- 🤖 **XGBoost Model**: High-performance gradient boosting classifier
- 🧪 **Feature Engineering**: Automatic handling of missing values, categorical encoding, and scaling
- 📊 **Interactive Dashboard**: User-friendly Streamlit interface with real-time predictions
- 📈 **Probability Scores**: View confidence levels for each prediction
- ⚙️ **Threshold Optimization**: Decision threshold tuned via precision-recall analysis
- 📋 **Model Metrics**: Detailed information about model performance and pipeline

## 🏗️ System Architecture

```text
User Input (Shipment Details)
           ↓
    Streamlit Interface
           ↓
Feature Engineering
           ↓
Missing Value Imputation
           ↓
Categorical Encoding
           ↓
Standard Scaling
           ↓
XGBoost Classifier
           ↓
Probability Output
           ↓
Decision Threshold
           ↓
Prediction (On Time / Not On Time)
```

## 📋 Input Features

The model accepts the following shipment and customer information:

| Feature | Type | Example Values |
|---------|------|-----------------|
| **Warehouse Block** | Categorical | A, B, C, D, E, F |
| **Mode of Shipment** | Categorical | Flight, Road, Ship |
| **Product Importance** | Categorical | Low, Medium, High |
| **Customer Gender** | Categorical | M, F |
| **Cost of Product** | Numeric | $10 - $500 |
| **Weight (grams)** | Numeric | 100 - 8000 |
| **Customer Care Calls** | Numeric | 1 - 7 |
| **Customer Rating** | Numeric | 1 - 5 |
| **Prior Purchases** | Numeric | 1 - 10 |
| **Discount Offered (%)** | Numeric | 0 - 65 |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Shipment-Sure
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present**
   - `shipment_xgboost_pipeline.pkl` - Pre-trained XGBoost pipeline
   - `decision_threshold.pkl` - Optimized decision threshold

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 Model Details

- **Algorithm**: XGBoost Classifier
- **Target Classes**: 
  - `0` → Not On Time (Delayed)
  - `1` → On Time (Delivered on schedule)
- **Pipeline Components**:
  - Missing value imputation
  - One-hot encoding for categorical features
  - Standard scaling for numerical features
  - Gradient boosting classification

## 📁 Project Structure

```
Shipment-Sure/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── dataset.csv                         # Training dataset
├── Shipmentt_model_training.ipynb      # Model training notebook
├── shipment_xgboost_pipeline.pkl       # Pre-trained model (not in repo)
├── decision_threshold.pkl              # Optimized threshold (not in repo)
├── catboost_info/                      # CatBoost training artifacts
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   ├── time_left.tsv
│   └── learn/
├── README.md                           # This file
└── .gitignore                          # Git ignore rules
```

## 📦 Dependencies

Core dependencies (see `requirements.txt` for full list):

- **streamlit** - Web interface framework
- **pandas** - Data manipulation
- **scikit-learn** - ML pipeline and preprocessing
- **xgboost** - Gradient boosting classifier
- **joblib** - Model serialization
- **numpy** - Numerical computations
- **lightgbm** - Alternative gradient boosting
- **catboost** - Alternative gradient boosting
- **imbalanced-learn** - Handling imbalanced datasets
- **matplotlib & seaborn** - Visualization

## 🔍 Model Training

The model was trained on e-commerce shipment data using the Jupyter notebook `Shipmentt_model_training.ipynb`. 

To retrain the model:

1. Open `Shipmentt_model_training.ipynb` in Jupyter
2. Update the training data if needed
3. Run all cells to train and evaluate
4. Export the new model files (`shipment_xgboost_pipeline.pkl` and `decision_threshold.pkl`)

## 💡 Usage Example

### Via Streamlit Interface

1. Launch the app: `streamlit run app.py`
2. Fill in shipment details:
   - Select warehouse block and shipping mode
   - Enter product cost and weight
   - Provide customer metrics
3. Click **"🔮 Predict Delivery Status"**
4. View the prediction result and probability scores

### Programmatic Usage

```python
import joblib
import pandas as pd

# Load model and threshold
pipeline = joblib.load("shipment_xgboost_pipeline.pkl")
threshold = joblib.load("decision_threshold.pkl")

# Create sample data
data = pd.DataFrame({
    "Warehouse_block": ["A"],
    "Mode_of_Shipment": ["Flight"],
    "Customer_care_calls": [3],
    "Customer_rating": [4],
    "Cost_of_the_Product": [200],
    "Prior_purchases": [2],
    "Product_importance": ["high"],
    "Gender": ["M"],
    "Discount_offered": [10],
    "Weight_in_gms": [2500]
})

# Get prediction
probabilities = pipeline.predict_proba(data)
prediction = 1 if probabilities[0][1] >= threshold else 0
print(f"Prediction: {'On Time' if prediction == 1 else 'Not On Time'}")
print(f"Probability: {probabilities[0][1]:.2%}")
```

## 📈 Model Performance

The model was optimized using precision-recall analysis on the validation dataset. The decision threshold was selected to balance false positives and false negatives based on business requirements.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Model files not found
**Error**: ❌ Required model files are missing.
**Solution**: Ensure `shipment_xgboost_pipeline.pkl` and `decision_threshold.pkl` are in the project root directory.

### Streamlit not installed
**Solution**: Run `pip install -r requirements.txt`

### Port already in use
**Solution**: Specify a different port: `streamlit run app.py --server.port 8502`

## 📞 Support

For issues, questions, or feedback, please open an issue on the repository or contact the development team.

---

**Built with ❤️ for better e-commerce logistics predictions**
                    ┌─────────────────┐
                    │ FAISS Top-K     │
                    │ Retrieval       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Context Builder │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Google Gemini   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Grounded Answer │
                    └─────────────────┘
```

## How It Works

### 1. PDF Ingestion

The user uploads a research paper through Streamlit.

### 2. PDF Parsing

PyMuPDF extracts text, page information, font characteristics, text blocks, and metadata.

### 3. Heading Detection

Formatting information is used to identify likely headings and reconstruct the logical organization of the paper.

Example:

```text
Research Paper
├── Abstract
├── 1. Introduction
├── 2. Related Work
├── 3. Methodology
│   ├── 3.1 Dataset
│   ├── 3.2 Model Architecture
│   └── 3.3 Training
├── 4. Experiments
├── 5. Results
└── 6. Conclusion
```

### 4. Structure-Aware Chunking

Instead of blindly splitting text by character count, chunks retain their parent section and page information.

Example chunk:

```text
{
    "text": "...",
    "sectionTitle": "3.2 Model Architecture",
    "pageNumber": 5
}
```

This preserves provenance and improves contextual retrieval.

### 5. Embedding Generation

Chunks are converted into dense semantic vectors using:

```text
BAAI/bge-small-en-v1.5
```

through Sentence Transformers.

The embedding dimension is 384.

### 6. Vector Storage

Embeddings are indexed using:

```text
FAISS IndexFlatIP
```

The embeddings are normalized, so inner product is equivalent to cosine similarity.

```text
cosine(A,B) = A · B
```

when both vectors have unit norm.

### 7. Query Retrieval

The user's question is converted into the same embedding space.

```text
Question
   ↓
BGE Embedding
   ↓
FAISS Similarity Search
   ↓
Top-K Relevant Chunks
```

### 8. Answer Generation

The retrieved chunks are supplied as context to Google Gemini.

```text
User Question
+
Retrieved Context
↓
Gemini
↓
Grounded Answer
```

## Why RAG?

Without RAG:

```text
Question → LLM → Answer
```

The model may rely on general knowledge or hallucinate.

With RAG:

```text
Question
 ↓
Retriever
 ↓
Relevant source passages
 ↓
LLM
 ↓
Grounded answer
```

RAG gives the model access to information retrieved directly from the uploaded document.

## Why Structure-Aware RAG?

Basic RAG often follows:

```text
PDF → Text → Fixed Chunks → Embeddings → Retrieval
```

This project follows:

```text
PDF
 ↓
Text + Layout
 ↓
Heading Detection
 ↓
Document Structure
 ↓
Section-Aware Chunks
 ↓
Embeddings
 ↓
Retrieval
```

The important distinction is that section and page context are preserved with retrieved chunks.

## Traditional RAG vs Structure-Aware RAG

| Feature | Traditional RAG | Structure-Aware RAG |
|---|---|---|
| Text extraction | Basic | Structure-aware |
| Heading information | May be lost | Preserved |
| Chunking | Fixed-size | Section-aware |
| Page information | May be lost | Preserved |
| Semantic retrieval | Yes | Yes |
| Vector search | Yes | FAISS |
| LLM generation | Yes | Gemini |
| Provenance | Limited | Section + page context |

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Streamlit | Interactive application |
| PyMuPDF | PDF parsing and layout extraction |
| Sentence Transformers | Embedding generation |
| BAAI/bge-small-en-v1.5 | Semantic embeddings |
| FAISS | Vector similarity search |
| Google Gemini | Answer generation |

## Project Structure

> Keep this section synchronized with the actual repository. Replace or remove placeholder entries if your current repository uses different filenames.

```text
Structure-Aware-Research-Paper-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── parsing/
│   ├── pdf_parser.py
│   ├── heading_scorer.py
│   └── ...
│
├── chunking/
│   ├── section_chunker.py
│   └── ...
│
├── retrieval/
│   ├── embeddings.py
│   ├── vector_store.py
│   └── ...
│
├── generation/
│   ├── gemini_client.py
│   └── ...
│
├── vector_stores/
│   └── ...
│
└── utils/
    └── ...
```

## Installation

### Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Structure-Aware-Research-Paper-Assistant
```

### Create Virtual Environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file if your implementation uses environment variables:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit API keys or secrets to GitHub.

## Run the Application

```bash
streamlit run app.py
```

The application normally runs at:

```text
http://localhost:8501
```

## Example Usage

1. Upload a research paper PDF.
2. Allow the document to be parsed and indexed.
3. Enter a natural-language question.
4. The question is embedded using BGE.
5. FAISS retrieves the most relevant chunks.
6. Retrieved context is passed to Gemini.
7. Gemini generates a context-grounded answer.

Example questions:

```text
What methodology does the paper propose?
What dataset was used?
What are the main experimental results?
What limitations do the authors mention?
What is the architecture of the proposed model?
```

## Advantages

### Context Preservation
Section and page information remain associated with chunks.

### Semantic Search
Queries are matched by meaning rather than exact keywords.

### Grounded Generation
Gemini receives retrieved source context before generating the answer.

### Modular Design
Document processing, retrieval, and generation are separated, making the system easier to extend.

## Limitations

- Heading detection can depend on PDF formatting.
- Ambiguous or multi-hop questions can reduce retrieval quality.
- Chunk size affects context completeness versus noise.
- Scanned PDFs require OCR support.
- Tables and figures are not fully represented in plain-text retrieval.
- Current design is primarily focused on individual research papers.

## Future Improvements

### Cross-Encoder Reranking

```text
FAISS
 ↓
Top 20 Candidates
 ↓
Cross-Encoder Reranker
 ↓
Top 5 Chunks
 ↓
Gemini
```

### Adaptive Chunking

Dynamically determine chunk boundaries using semantic and section-level information.

### Hierarchical Retrieval

```text
Document
 ↓
Section
 ↓
Subsection
 ↓
Relevant Chunk
```

### OCR Support

Add OCR for scanned research papers.

### Multimodal Document Understanding

Extend retrieval to tables, figures, captions, and equations.

### Multi-Document Search

Allow users to upload and search across multiple research papers.

### Citation Highlighting

Show the exact source section and page used to generate an answer.

### Conversation Memory

Maintain context across multiple questions about the same paper.

## Interview Explanation

### 1-Minute Pitch

> I built a Structure-Aware Research Paper Assistant using Retrieval-Augmented Generation. The system allows users to upload research papers and ask questions using natural language. I use PyMuPDF to extract text, metadata, and layout information, then detect headings and preserve the document structure during semantic chunking. The chunks are converted into embeddings using BAAI/bge-small-en-v1.5 and stored in a FAISS vector index. When a user asks a question, the query is embedded and FAISS retrieves the most relevant chunks. These retrieved sections are passed to Google Gemini as context, and Gemini generates a grounded answer. The main difference from basic RAG is that section and page information are preserved during retrieval, which improves contextual understanding and provenance.

## Key Interview Questions

### Why RAG?

RAG lets the system retrieve information directly from the uploaded research paper instead of depending only on the LLM's pretrained knowledge.

### Why not send the entire PDF directly to Gemini?

Retrieval selects only the most relevant content for the current question, reducing unnecessary context and making the system more targeted.

### Why embeddings?

Embeddings capture semantic meaning, allowing the system to retrieve conceptually similar passages even when the query does not use the exact words found in the paper.

### Why BGE?

BGE is designed for strong semantic representations and retrieval-oriented embedding tasks.

### Why FAISS?

FAISS provides efficient vector similarity search and is lightweight and practical for this type of local RAG system.

### Why IndexFlatIP?

The embeddings are normalized, so inner product is equivalent to cosine similarity:

```text
Cosine Similarity = A · B
```

for unit-normalized vectors.

### What makes the system structure-aware?

The system detects document headings and preserves section and page metadata with each chunk instead of treating the entire PDF as unstructured text.

### What happens if retrieval is poor?

Potential improvements include better chunking, query rewriting, metadata filtering, hybrid search, increasing/decreasing Top-K, and cross-encoder reranking.

### What is the main limitation?

Retrieval quality depends on PDF parsing, heading detection, chunking strategy, and embedding quality. Poor document structure can reduce retrieval accuracy.

## Core RAG Pipeline

### Indexing

```text
Research Paper
      ↓
PyMuPDF
      ↓
Text + Layout + Metadata
      ↓
Heading Detection
      ↓
Document Structure
      ↓
Structure-Aware Chunking
      ↓
BGE Embeddings
      ↓
FAISS Index
```

### Querying

```text
User Question
      ↓
BGE Embedding
      ↓
FAISS Similarity Search
      ↓
Top-K Chunks
      ↓
Context Construction
      ↓
Google Gemini
      ↓
Grounded Answer
```

## Project Knowledge Base

```text
Project:
Structure-Aware Research Paper Assistant

Primary Goal:
Research-paper question answering using Structure-Aware RAG

Interface:
Streamlit

PDF Processing:
PyMuPDF

Document Processing:
Heading detection
Metadata extraction
Structure-aware chunking

Embedding Model:
BAAI/bge-small-en-v1.5

Embedding Library:
Sentence Transformers

Vector Search:
FAISS

Index:
IndexFlatIP

Similarity:
Cosine similarity through normalized inner product

LLM:
Google Gemini

Pipeline:
PDF → Parse → Structure → Chunk → Embed → FAISS
→ Retrieve → Context → Gemini → Answer
```

## Authors

**Meghana Sri Kondeti**

**Surya Tej Meka**

## License

This project is intended for educational, research, and demonstration purposes.
