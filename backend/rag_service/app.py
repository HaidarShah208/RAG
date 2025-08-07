"""
FastAPI RAG microservice with Local LLM:
✅ Pinecone integration for vector storage
✅ Local LLM (llama-cpp) for text generation
✅ PDF text extraction and chunking
✅ Vector embedding and storage
✅ Query processing with RAG
✅ CORS enabled for frontend integration
"""

from fastapi import FastAPI, UploadFile, File, Form, Request, status, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uvicorn
import PyPDF2
import numpy as np
import uuid
import os
import io
import re
from dotenv import load_dotenv
import docx
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

 
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
    print("✅ Llama-cpp available - using local LLM")
except ImportError:
    print("❌ Llama-cpp not available. Please install: pip install llama-cpp-python")
    exit(1)

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("Validation error:", exc.errors())
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": exc.errors()})

 
PINECONE_API_KEY = 'pcsk_6SakMP_PLbnZkok8sr4fLoHpMViYzNubor5z4oMSXa9jHtgM6HzfXJxYJjZW5UAgQ82D3J'
INDEX_NAME = "chatbase-chunks"

pc = Pinecone(api_key=PINECONE_API_KEY)

 
embedder = SentenceTransformer("all-MiniLM-L6-v2")

 
llm = None
try:
         
    model_paths = [
        "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        os.getenv("LLAMA_MODEL_PATH", "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,  # Increased context window
                    n_gpu_layers=0,  
                    verbose=False
                )
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠️  Failed to load {model_path}: {e}")
                continue
    
    if not model_loaded:
        print("❌ No models found. Please download a model first:")
        exit(1)
        
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")
    exit(1)

 
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_bytes):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = " ".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text(file_bytes, filename=None):
    """Extract text from various file types"""
    if filename:
        ext = filename.lower().split('.')[-1]
        if ext == 'pdf':
            return extract_text_from_pdf(file_bytes)
        elif ext == 'docx':
            return extract_text_from_docx(file_bytes)
        elif ext == 'txt':
            return file_bytes.decode("utf-8")
    return file_bytes.decode("utf-8")

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
         
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > start + chunk_size * 0.7:  
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [chunk for chunk in chunks if chunk.strip()]

def generate_local_response(context, query):
    """Generate response using local LLM"""
    if not context.strip():
        return "I don't have enough information to answer your question."
    
     
    prompt = f"""Answer using only this context: {context}
    
Question: {query}
Answer:"""
    
    try:
        response = llm(prompt, max_tokens=500, temperature=0.7, stop=["Question:", "Context:"])
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return f"Error generating response: {str(e)}"



class QueryRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    data_source_id: str = None

@app.post("/ingest")
async def ingest(
    data_source_id: str = Form(None),
    file: UploadFile = File(...)
):
    """Ingest PDF file and create embeddings"""
    try:
        file_bytes = await file.read()
        
         
        text = extract_text(file_bytes, file.filename)
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "No text could be extracted from the file"}
            )
        
         
        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks from {file.filename}")
        
         
        embeddings = embedder.encode(chunks, convert_to_numpy=True)
        
         
        vectors = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            vector_id = str(uuid.uuid4())
            metadata = {
                "chunk": chunk,
                "chunk_index": i,
                "filename": file.filename
            }
            
            if data_source_id:
                metadata["data_source_id"] = data_source_id
                
            vectors.append({
                "id": vector_id,
                "values": vector.tolist(),
                "metadata": metadata
            })
        

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        print(f"[INGEST] Successfully ingested {len(vectors)} chunks")
        return {
            "status": "success",
            "chunks_added": len(vectors),
            "filename": file.filename
        }
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to ingest file: {str(e)}"}
        )

@app.post("/query")
async def query(req: QueryRequest):
    """Query the RAG system"""
    try:
         
        query_embedding = embedder.encode([req.query], convert_to_numpy=True)[0]
        
        print(f"[QUERY] Processing query: {req.query}")
        
         
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=3,  # Reduced from 5 to 3
            include_metadata=True
        )
        
        if not results.matches:
            return JSONResponse(
                status_code=404,
                content={"error": "No relevant documents found"}
            )
        
         
        retrieved_chunks = [match.metadata["chunk"] for match in results.matches]
        # Limit context to prevent token overflow
        context = "\n\n".join(retrieved_chunks)
        if len(context) > 2000:  # Limit to ~2000 characters
            context = context[:2000] + "..."
        
        print(f"[QUERY] Retrieved {len(retrieved_chunks)} chunks")
        print(f"[QUERY] Context length: {len(context)} characters")
        

        answer = generate_local_response(context, req.query)
        
        print(f"[QUERY] Generated answer: {answer[:200]}...")
        
        return {
            "answer": answer,
            "context_chunks": len(retrieved_chunks),
            "query": req.query,
            "llm_type": "local-llama"
        }
        
    except Exception as e:
        print(f"Error during query: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process query: {str(e)}"}
        )

@app.post("/debug-retrieve")
async def debug_retrieve(req: QueryRequest):
    """Debug endpoint to see what chunks are retrieved"""
    try:
        query_embedding = embedder.encode([req.query], convert_to_numpy=True)[0]
        
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=3,  # Reduced from 5 to 3
            include_metadata=True
        )
        
        retrieved_chunks = [match.metadata["chunk"] for match in results.matches]
        
        return {
            "query": req.query,
            "retrieved_chunks": retrieved_chunks,
            "num_chunks": len(retrieved_chunks)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Debug retrieve failed: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002) 