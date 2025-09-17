🔍 Advanced Document Search Engine
This project is a complete solution for managing and performing semantic searches on documents. It combines robust file processing with an ultra-fast vector search engine. The system integrates the file management features from the original ds2.py with the high-performance search capabilities of vector_search.py.

🚀 Key Features
Document Processing: Extracts text from various file formats including PDF, DOCX, HTML, CSV, XLSX, and TXT.

Semantic Embeddings Generation: Converts document text into meaningful numerical vectors using an advanced language model like all-MiniLM-L6-v2.

Rapid Vector Search: Utilizes the hnswlib library to create an HNSW index, which allows for finding the most relevant documents for a query in a very short amount of time.

Index Management: Creates, saves, and loads the index and embeddings to avoid re-calculation on subsequent runs.

Simple Interface: An intuitive command-line interface is provided to interact with the system.

📦 Project Structure
mon_projet_combine/
├── src/
│   ├── document_processor.py      # Manages text extraction and embeddings creation.
│   ├── vector_search_engine.py    # Manages HNSW indexing and searching.
│   └── main.py                    # The program's entry point, orchestrating the workflow.
├── data/
│   ├── raw_documents/             # Directory to place your documents (PDFs, DOCX, etc.).
│   └── index_data/                # Directory where index files are saved.
├── requirements.txt               # A list of necessary dependencies.
└── README.md
⚙️ Installation and Usage
Clone the repository and navigate into the project directory.

Install dependencies using the requirements.txt file. This file contains all necessary libraries, including hnswlib, sentence-transformers, PyPDF2, and others.

Bash

pip install -r requirements.txt
Add your documents: Place the files you want to search through in the data/raw_documents/ directory.

Launch the application: Run the main script from your terminal.

Bash

python src/main.py
The program will display its progress and prompt you to enter a query to find the most relevant documents in your collection.