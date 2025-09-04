"""
OCR RAG Chat Agent with LangChain + LangGraph (single-file example)

New features:
- PNG-to-text conversion using Tesseract OCR
- **PDF-to-text conversion** (native text extraction) with optional OCR fallback for scanned PDFs

What it does
------------
- Converts a .png file into a plain text file with OCR
- Reads a .pdf and converts it to a text file (and can OCR if it's a scanned PDF)
- Ingests OCR/text (from .png, .pdf, .txt, or a raw string)
- Indexes it with embeddings for semantic search
- Lets a user ask questions in a chat loop
- Retrieves relevant chunks, answers with RAG, then summarizes the answer

Prereqs
-------
Core:
  pip install -U "langchain>=0.2" langchain-community langchain-openai langgraph faiss-cpu tiktoken pillow pytesseract pypdf

Optional (for OCR'ing scanned PDFs):
  pip install pdf2image
  # And install Poppler system package so pdf2image can rasterize PDFs
  # Windows: https://github.com/oschwartz10612/poppler-windows (add bin folder to PATH)
  # macOS:   brew install poppler
  # Linux:   sudo apt-get install poppler-utils

You must also have Tesseract installed on your system:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux/macOS: `sudo apt-get install tesseract-ocr` or `brew install tesseract`

Run via Bash or Git Bash on Windows
---
py -3.11 OCRChatBot2.py --ocr-file path/to/ocr.txt
py -3.11 OCRChatBot2.py --ocr-text "...big OCR text..."
py -3.11 OCRChatBot2.py --ocr-png path/to/image.png
py -3.11 OCRChatBot2.py --pdf path/to/doc.pdf
py -3.11 OCRChatBot2.py --ocr-pdf path/to/scanned.pdf

Type questions at the prompt. Type :quit to exit.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, TypedDict, Optional

# Reads the api_key from the .env file at same directory level as this file
from dotenv import load_dotenv
load_dotenv()

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

# OCR imports
from PIL import Image
import pytesseract
# ***************************************************
# *    Modify this line for correct path to .exe    *
# ***************************************************
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

from pypdf import PdfReader
# Optional OCR fallback for scanned PDFs
try:
    from pdf2image import convert_from_path  # requires Poppler
    _HAS_PDF2IMAGE = True
except Exception:
    _HAS_PDF2IMAGE = False

# ---------------------------
# Configuration
# ---------------------------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
TOP_K = int(os.environ.get("TOP_K", 4))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# OCR Utility (PNG)
# ---------------------------
def ocr_png_to_text(png_path: str, txt_out: Optional[str] = None) -> str:
    """Convert a .png image into OCR text and optionally save to a text file."""
    image = Image.open(png_path)
    text = pytesseract.image_to_string(image)
    if txt_out:
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(text)
    return text

# ---------------------------
# PDF Utilities
# ---------------------------
def extract_text_from_pdf(pdf_path: str, txt_out: Optional[str] = None) -> str:
    """Extract native text from a PDF (non-scanned)."""
    reader = PdfReader(pdf_path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    text = "\n\n".join(pages_text)
    if txt_out:
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def ocr_pdf_to_text(pdf_path: str, txt_out: Optional[str] = None, dpi: int = 300) -> str:
    """OCR a scanned PDF by rasterizing each page to an image, then running Tesseract.
    Requires pdf2image + Poppler + Tesseract.
    """
    if not _HAS_PDF2IMAGE:
        raise RuntimeError("pdf2image not installed or Poppler missing. Install pdf2image and Poppler to OCR scanned PDFs.")

    pages = convert_from_path(pdf_path, dpi=dpi)
    texts = []
    for idx, img in enumerate(pages, start=1):
        txt = pytesseract.image_to_string(img)
        texts.append(txt)
    full_text = "\n\n".join(texts)
    if txt_out:
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(full_text)
    return full_text

# ---------------------------
# Ingestion / Indexing
# ---------------------------
def build_vectorstore_from_text(ocr_text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(ocr_text)]
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = FAISS.from_documents(docs, embeddings)
    return vs

# ---------------------------
# RAG Prompts
# ---------------------------
RAG_SYSTEM_PROMPT = (
    "You are a careful assistant. Use the provided context to answer the user's question.\n"
    "- If the answer cannot be found in the context, say that clearly.\n"
    "- Be concise and cite short quoted snippets from the context where helpful."
)

RAG_USER_PROMPT = (
    "<question>\n{question}\n</question>\n\n"
    "<context>\n{context}\n</context>\n\n"
    "Answer based only on the context above."
)

SUMMARY_SYSTEM_PROMPT = (
    "You produce a brief executive summary for a non-technical reader, \n"
    "highlighting the direct answer and any key caveats in 3-5 bullet points."
)

SUMMARY_USER_PROMPT = (
    "Summarize the following answer succinctly in bullets:\n\n{answer}"
)

# ---------------------------
# LangGraph State
# ---------------------------
class GraphState(TypedDict):
    question: str
    chat_history: List[Tuple[str, str]]  # (user, assistant)
    context: str
    answer: str
    summary: str

# ---------------------------
# Node Implementations
# ---------------------------
@dataclass
class RAGComponents:
    retriever: any
    llm: any


def make_retrieve_node(rag: RAGComponents):
    def _retrieve(state: GraphState) -> GraphState:
        question = state["question"]
        docs: List[Document] = rag.retriever.get_relevant_documents(question)
        context = "\n\n".join(d.page_content.strip() for d in docs)
        return {**state, "context": context}

    return RunnableLambda(_retrieve)


def make_answer_node(rag: RAGComponents):
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("user", RAG_USER_PROMPT),
    ])

    chain = prompt | rag.llm

    def _answer(state: GraphState) -> GraphState:
        resp = chain.invoke({
            "question": state["question"],
            "context": state.get("context", ""),
        })
        content = resp.content if hasattr(resp, "content") else str(resp)
        return {**state, "answer": content}

    return RunnableLambda(_answer)


def make_summarize_node(rag: RAGComponents):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("user", SUMMARY_USER_PROMPT),
    ])
    chain = prompt | rag.llm

    def _summarize(state: GraphState) -> GraphState:
        ans = state.get("answer", "")
        resp = chain.invoke({"answer": ans})
        content = resp.content if hasattr(resp, "content") else str(resp)
        return {**state, "summary": content}

    return RunnableLambda(_summarize)

# ---------------------------
# Graph Builder
# ---------------------------
def build_graph(retriever, llm) -> any:
    rag = RAGComponents(retriever=retriever, llm=llm)

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", make_retrieve_node(rag))
    graph.add_node("answer", make_answer_node(rag))
    graph.add_node("summarize", make_summarize_node(rag))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()

# ---------------------------
# Chat Loop / Interface
# ---------------------------
def run_chat(retriever, llm):
    app = build_graph(retriever, llm)
    chat_history: List[Tuple[str, str]] = []

    print("\n🧠 OCR RAG Agent — ask anything about your document. Type :quit to exit.\n")
    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in {":q", ":quit", ":exit"}:
            break

        init_state: GraphState = {
            "question": user_q,
            "chat_history": chat_history,
            "context": "",
            "answer": "",
            "summary": "",
        }
        final_state = app.invoke(init_state)

        answer = final_state["answer"].strip()
        summary = final_state["summary"].strip()

        print("\nAssistant (answer):\n" + answer + "\n")
        print("Assistant (summary):\n" + summary + "\n")

        chat_history.append((user_q, answer))

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="OCR RAG Chat Agent with LangChain + LangGraph")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ocr-file", type=str, help="Path to a plain-text file produced by OCR")
    source.add_argument("--ocr-text", type=str, help="OCR'd text passed directly on the command line")
    source.add_argument("--ocr-png", type=str, help="Path to a .png file to OCR and convert to text")
    source.add_argument("--pdf", type=str, help="Path to a PDF file (native text extraction)")
    source.add_argument("--ocr-pdf", type=str, help="Path to a scanned PDF to OCR (requires pdf2image + Poppler + Tesseract)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of retrieved chunks")
    parser.add_argument("--model", type=str, default=CHAT_MODEL, help="Chat model (e.g., gpt-4o-mini)")
    parser.add_argument("--embedding", type=str, default=EMBEDDING_MODEL, help="Embedding model")
    parser.add_argument("--temp", type=float, default=TEMPERATURE, help="LLM temperature")
    args = parser.parse_args()

    # Load text
    if args.ocr_file:
        with open(args.ocr_file, "r", encoding="utf-8", errors="ignore") as f:
            ocr_text = f.read()
    elif args.ocr_text:
        ocr_text = args.ocr_text
    elif args.ocr_png:
        print(f"Running OCR on image: {args.ocr_png}")
        ocr_text = ocr_png_to_text(args.ocr_png, txt_out="ocr_output.txt")
        print("OCR result saved to ocr_output.txt")
    elif args.pdf:
        print(f"Extracting text from PDF: {args.pdf}")
        ocr_text = extract_text_from_pdf(args.pdf, txt_out="pdf_output.txt")
        print("PDF text saved to pdf_output.txt")
        # If the PDF has no extractable text, suggest OCR fallback
        if not ocr_text.strip():
            print("No text found via native extraction. If this is a scanned PDF, try --ocr-pdf instead.")
    elif args.ocr_pdf:
        print(f"OCR'ing scanned PDF: {args.ocr_pdf}")
        ocr_text = ocr_pdf_to_text(args.ocr_pdf, txt_out="ocr_pdf_output.txt")
        print("OCR PDF text saved to ocr_pdf_output.txt")
    else:
        raise ValueError("No OCR input provided.")

    if not ocr_text.strip():
        raise ValueError("No text content available after extraction/OCR.")

    # Build vector store & retriever
    vectorstore = build_vectorstore_from_text(ocr_text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.top_k})

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    # Run chat
    run_chat(retriever, llm)


if __name__ == "__main__":
    main()