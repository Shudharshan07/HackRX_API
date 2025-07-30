from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import json
import requests
import pdfplumber
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

# ----------- Models -----------

class InputPayload(BaseModel):
    documents: HttpUrl
    questions: List[str]

class OutputPayload(BaseModel):
    answer: List[str]

# ----------- Core Classes -----------

class DocumentLoader:
    def __init__(self, url):
        self.url = url

    def download_pdf(self, save_path="temp.pdf"):
        response = requests.get(self.url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path

    def extract_text(self, path):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text


class SemanticQAEngine:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
        self.text_chunks = []
        self.index = None

    def chunk_text(self, text, chunk_size=500):
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        self.text_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                for i in range(0, len(chunk), chunk_size):
                    self.text_chunks.append(chunk[i:i+chunk_size])
            else:
                self.text_chunks.append(chunk)

    def build_index(self):
        embeddings = self.model.encode(self.text_chunks, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, question, top_k=3):
        q_vec = self.model.encode([question], convert_to_numpy=True)
        distances, indices = self.index.search(q_vec, top_k)
        return [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]

    def answer_question(self, question):
        top_chunks = self.search(question)
        context = " ".join(top_chunks)
        prompt = f"Based on the context below, answer the question:\nContext: {context}\nQuestion: {question}"
        result = self.qa_model(prompt, max_new_tokens=100, do_sample=False)
        return result[0]["generated_text"]


# ----------- Route -----------
engine = SemanticQAEngine()

@app.post("/hackrx/run", response_model=OutputPayload)
async def run_pipeline(payload: InputPayload, Authorization: Optional[str] = Header(None)):
    try:
        # Step 1: Load and parse document
        loader = DocumentLoader(payload.documents)
        pdf_path = loader.download_pdf()
        raw_text = loader.extract_text(pdf_path)

        # Step 2: Build semantic QA engine
        engine.chunk_text(raw_text)
        engine.build_index()

        # Step 3: Answer questions
        answers = [engine.answer_question(q) for q in payload.questions]

        return {"answer": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
