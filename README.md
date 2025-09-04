# LangGraphOCRChatBot





OCR RAG Chat Agent with LangChain + LangGraph (single-file example)



New features:

\- PNG-to-text conversion using Tesseract OCR

\- \*\*PDF-to-text conversion\*\* (native text extraction) with optional OCR fallback for scanned PDFs



What it does

------------

\- Converts a .png file into a plain text file with OCR

\- Reads a .pdf and converts it to a text file (and can OCR if it's a scanned PDF)

\- Ingests OCR/text (from .png, .pdf, .txt, or a raw string)

\- Indexes it with embeddings for semantic search

\- Lets a user ask questions in a chat loop

\- Retrieves relevant chunks, answers with RAG, then summarizes the answer



Prereqs

-------

Core:

&nbsp; pip install -U "langchain>=0.2" langchain-community langchain-openai langgraph faiss-cpu tiktoken pillow pytesseract pypdf



Optional (for OCR'ing scanned PDFs):

&nbsp; pip install pdf2image

&nbsp; # And install Poppler system package so pdf2image can rasterize PDFs

&nbsp; # Windows: https://github.com/oschwartz10612/poppler-windows (add bin folder to PATH)

&nbsp; # macOS:   brew install poppler

&nbsp; # Linux:   sudo apt-get install poppler-utils



You must also have Tesseract installed on your system:

\- Windows: https://github.com/UB-Mannheim/tesseract/wiki

\- Linux/macOS: `sudo apt-get install tesseract-ocr` or `brew install tesseract`



Run via Bash or Git Bash on Windows

---

py -3.11 OCRChatBot2.py --ocr-file path/to/ocr.txt

py -3.11 OCRChatBot2.py --ocr-text "...big OCR text..."

py -3.11 OCRChatBot2.py --ocr-png path/to/image.png

py -3.11 OCRChatBot2.py --pdf path/to/doc.pdf

py -3.11 OCRChatBot2.py --ocr-pdf path/to/scanned.pdf



Type questions at the prompt. Type :quit to exit.

