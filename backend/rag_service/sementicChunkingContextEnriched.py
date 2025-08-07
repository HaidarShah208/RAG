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

def semantic_chunk_text(text, chunk_size=1000, overlap=200):
    """Semantic chunking based on sentence similarity"""
    if len(text) <= chunk_size:
        return [text]
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 1:
        return chunk_text(text, chunk_size, overlap)  
    
    try:
        sentence_embeddings = embedder.encode(sentences, convert_to_numpy=True)
        
         
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            vec1 = sentence_embeddings[i]
            vec2 = sentence_embeddings[i + 1]
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(similarity)
        
         
        if similarities:
            threshold = np.percentile(similarities, 75)  
            breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold]
        else:
            breakpoints = []
        
         
        chunks = []
        start = 0
        
        for bp in breakpoints:
            chunk_text = ". ".join(sentences[start:bp + 1])
            if len(chunk_text) > 50:  
                chunks.append(chunk_text)
            start = bp + 1
        
         
        if start < len(sentences):
            chunk_text = ". ".join(sentences[start:])
            if len(chunk_text) > 50:
                chunks.append(chunk_text)
        
        # If semantic chunking didn't work well, fallback to regular chunking
        if len(chunks) < 2:
            return chunk_text(text, chunk_size, overlap)
        
        return chunks
        
    except Exception as e:
        print(f"Semantic chunking failed, using regular chunking: {e}")
        return chunk_text(text, chunk_size, overlap)

def context_enriched_retrieve(query_embedding, top_k=3, context_size=1):
    """Retrieve chunks with neighboring context"""
    try:
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        if not results.matches:
            return []
        
        # Get all unique chunk indices
        chunk_indices = set()
        for match in results.matches:
            chunk_index = match.metadata.get("chunk_index", 0)
            # Add neighboring chunks
            for i in range(max(0, int(chunk_index - context_size)), 
                          int(chunk_index + context_size + 1)):
                chunk_indices.add(i)
        
        # Retrieve all chunks with context
        enriched_chunks = []
        for match in results.matches:
            chunk_index = match.metadata.get("chunk_index", 0)
            
            # Get neighboring chunks
            start_idx = max(0, int(chunk_index - context_size))
            end_idx = int(chunk_index + context_size + 1)
            
            # Find all chunks in this range
            for other_match in results.matches:
                other_index = other_match.metadata.get("chunk_index", 0)
                if start_idx <= int(other_index) < end_idx:
                    enriched_chunks.append(other_match.metadata["chunk"])
            
             
            break
        
         
        seen = set()
        unique_chunks = []
        for chunk in enriched_chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)
        
        return unique_chunks[:top_k + context_size * 2]  
        
    except Exception as e:
        print(f"Context-enriched retrieval failed: {e}")
         
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return [match.metadata["chunk"] for match in results.matches]

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
    use_context_enriched: bool = True  

class IngestRequest(BaseModel):
    data_source_id: str = None
    use_semantic_chunking: bool = True  

@app.post("/ingest")
async def ingest(
    data_source_id: str = Form(None),
    file: UploadFile = File(...),
    use_semantic_chunking: bool = Form(True)
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
        
         
         
        if use_semantic_chunking:
            chunks = semantic_chunk_text(text)
            print(f"Created {len(chunks)} semantic chunks from {file.filename}")
        else:
            chunks = chunk_text(text)
            print(f"Created {len(chunks)} regular chunks from {file.filename}")
        
         
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
        
         
         
        if req.use_context_enriched:
            retrieved_chunks = context_enriched_retrieve(query_embedding, top_k=3, context_size=1)
            print(f"[QUERY] Using context-enriched retrieval")
        else:
             
            results = index.query(
                vector=query_embedding.tolist(),
                top_k=3,
                include_metadata=True
            )
            retrieved_chunks = [match.metadata["chunk"] for match in results.matches]
            print(f"[QUERY] Using regular retrieval")
        
        if not retrieved_chunks:
            return JSONResponse(
                status_code=404,
                content={"error": "No relevant documents found"}
            )
        
         
        context = "\n\n".join(retrieved_chunks)
        if len(context) > 2000:  
            context = context[:2000] + "..."
        
        print(f"[QUERY] Retrieved {len(retrieved_chunks)} chunks")
        print(f"[QUERY] Context length: {len(context)} characters")
        print(f"[QUERY] Retrieval method: {'Context-Enriched' if req.use_context_enriched else 'Regular'}")
        

        answer = generate_local_response(context, req.query)
        
        print(f"[QUERY] Generated answer: {answer}")
        
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
        
         
        retrieved_chunks = context_enriched_retrieve(query_embedding, top_k=3, context_size=1)
        
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "rag-service-enhanced",
        "features": {
            "semantic_chunking": True,
            "context_enriched_retrieval": True,
            "local_llm": True,
            "pinecone_integration": True
        },
        "llm_type": "local-llama"
    }

@app.post("/test-chunking")
async def test_chunking(text: str = Body(...)):
    """Test different chunking methods on sample text"""
    try:
         
        regular_chunks = chunk_text(text)
        
         
        semantic_chunks = semantic_chunk_text(text)
        
        return {
            "regular_chunks": {
                "count": len(regular_chunks),
                    "chunks": regular_chunks[:3]  
            },
            "semantic_chunks": {
                "count": len(semantic_chunks),
                "chunks": semantic_chunks[:3]  
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Chunking test failed: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002) 