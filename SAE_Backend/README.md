# ConceptViz Backend

The backend service for ConceptViz - providing SAE model processing, feature analysis, and activation steering capabilities for Large Language Model interpretability research.

## 🛠️ Tech Stack

- **Framework**: Flask with Python 3.8+
- **ML Libraries**: PyTorch, Transformers, NumPy, Scikit-learn
- **Vector Operations**: FAISS for similarity search
- **Embeddings**: OpenAI text-embedding-3-large
- **Data Processing**: UMAP, Pandas, JSON handling
- **API**: RESTful APIs with JSON responses

## 📁 Project Structure

```
backend/
├── SAE_Backend/          # Submodule for SAE processing logic
├── app/                  # Main Flask application
│   └── routes/           # API route handlers
│       ├── __init__.py   # Routes package initialization
│       ├── input.py      # API for Identification
│       ├── explore.py    # API for Interpretation  
│       └── validate.py   # API for Validation
├── utils/                # Utility modules and helpers
│   ├── __init__.py       
│   ├── db_manager.py     # Database and file management
│   ├── errors.py         # Custom error handling
│   ├── global_state.py   # Application-wide state management
│   ├── openai_service.py # OpenAI API integration service
│   └── vector_db.py      # Vector database operations
├── .gitignore            # Git ignore rules
├── ...
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or later
- pip or conda package manager
- OpenAI API key (for embeddings)
- CUDA (optional, for GPU acceleration)


### Running the Server

```bash
python run.py
```

The server will start on `http://localhost:5000`



For more information about the overall project, see the [main repository](../README.md).