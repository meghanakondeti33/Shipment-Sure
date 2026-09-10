# Structure-Aware Research Paper Assistant

An AI-powered Retrieval-Augmented Generation (RAG) system for asking natural-language questions over research papers. The system preserves document structure during parsing and chunking, performs semantic retrieval using BGE embeddings and FAISS, and generates context-grounded answers using Google Gemini.

## Overview

Research papers are long and information-dense. Traditional keyword search can miss semantically related information, while naive fixed-size chunking can separate content from its section context.

This project addresses these problems with a structure-aware RAG pipeline:

```text
PDF
 ↓
PyMuPDF Parsing
 ↓
Heading Detection + Metadata
 ↓
Structure-Aware Chunking
 ↓
BGE Embeddings
 ↓
FAISS Vector Search
 ↓
Top-K Relevant Chunks
 ↓
Google Gemini
 ↓
Grounded Answer
```

## Key Features

- PDF research-paper upload
- Text and metadata extraction
- Heading and section detection
- Structure-aware semantic chunking
- BGE embeddings using Sentence Transformers
- FAISS vector similarity search
- Google Gemini answer generation
- Section and page context preservation
- Interactive Streamlit interface
- Modular document-processing, retrieval, and generation pipeline

## System Architecture

```text
                    ┌─────────────────┐
                    │      User       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Streamlit    │
                    │       UI        │
                    └────────┬────────┘
                             │
                       Upload PDF
                             │
                             ▼
                    ┌─────────────────┐
                    │    PyMuPDF      │
                    │   PDF Parser    │
                    └────────┬────────┘
                             │
                    Text + Layout + Metadata
                             │
                             ▼
                    ┌─────────────────┐
                    │ Heading /       │
                    │ Structure       │
                    │ Detection       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Section-Aware   │
                    │ Chunking        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ BGE Embeddings  │
                    │ Sentence        │
                    │ Transformers    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ FAISS           │
                    │ Vector Index    │
                    └────────┬────────┘
                             │
                       User Question
                             │
                             ▼
                    ┌─────────────────┐
                    │ Query Embedding │
                    └────────┬────────┘
                             │
                             ▼
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
