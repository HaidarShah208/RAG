"""
FastAPI RAG microservice with Local LLM and Contextual Chunk Headers (CCH):
✅ Pinecone integration for vector storage
✅ Local LLM (llama-cpp) for text generation
✅ PDF text extraction and chunking
✅ Contextual Chunk Headers for better retrieval
✅ Dual embeddings (header + content)
✅ Enhanced semantic search with combined similarity
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
                    verbose=False,
                    chat_format="chatml"  # Better chat format
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

def generate_chunk_header(chunk_text):
    """Generate a descriptive header for a text chunk using local LLM"""
    try:
        # Create a prompt to generate a concise header
        prompt = f"""Generate a concise and informative title (max 10 words) for this text chunk. 
        Return only the title, nothing else.
        
        Text chunk: {chunk_text[:500]}...
        
        Title:"""
        
        response = llm(prompt, max_tokens=20, temperature=0.3, stop=["\n", "Text chunk:", "Title:"])
        header = response['choices'][0]['text'].strip()
        
        # Clean up the header
        header = re.sub(r'^[^a-zA-Z]*', '', header)  # Remove leading non-letters
        header = re.sub(r'[^a-zA-Z0-9\s\-_\.]', '', header)  # Keep only alphanumeric, spaces, hyphens, underscores, dots
        header = header.strip()
        
        # If header generation failed, create a simple one
        if not header or len(header) < 3:
            # Extract first meaningful sentence or create a simple header
            sentences = re.split(r'[.!?]', chunk_text)
            first_sentence = sentences[0].strip() if sentences else chunk_text[:50]
            header = f"Content: {first_sentence[:30]}..."
        
        return header
        
    except Exception as e:
        print(f"Header generation failed: {e}")
        # Fallback to simple header
        return f"Content: {chunk_text[:30]}..."

def generate_simple_header(chunk_text):
    """Generate a simple header without LLM for faster processing"""
    try:
        # Extract first meaningful sentence
        sentences = re.split(r'[.!?]', chunk_text)
        first_sentence = sentences[0].strip() if sentences else chunk_text[:50]
        
        # Clean up and limit length
        header = re.sub(r'[^a-zA-Z0-9\s\-_\.]', '', first_sentence)
        header = header.strip()[:50]  # Limit to 50 characters
        
        if len(header) < 10:
            # If too short, use a simple pattern
            words = chunk_text.split()[:5]
            header = " ".join(words) + "..."
        
        return f"Content: {header}"
        
    except Exception as e:
        return f"Content: {chunk_text[:30]}..."

def chunk_text_with_headers(text, chunk_size=1000, overlap=200, use_llm_headers=False):
    """Split text into overlapping chunks with generated headers"""
    if len(text) <= chunk_size:
        header = generate_chunk_header(text) if use_llm_headers else generate_simple_header(text)
        return [{"header": header, "text": text}]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundaries
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > start + chunk_size * 0.7:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunk_text = chunk.strip()
        if chunk_text:
            # Use simple headers for faster processing (can be changed to LLM headers if needed)
            header = generate_chunk_header(chunk_text) if use_llm_headers else generate_simple_header(chunk_text)
            chunks.append({"header": header, "text": chunk_text})
        
        start = end - overlap
    
    return chunks

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def enhanced_semantic_search(query, query_embedding, top_k=3):
    """Enhanced search using both header and content embeddings"""
    try:
        # Get all vectors from Pinecone
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k * 2,  # Get more results for re-ranking
            include_metadata=True
        )
        
        if not results.matches:
            return []
        
        # Re-rank using combined similarity (header + content)
        enhanced_results = []
        for match in results.matches:
            chunk_text = match.metadata.get("chunk", "")
            chunk_header = match.metadata.get("header", "")
            
            # Create embeddings for header and content on-the-fly
            if chunk_header and chunk_text:
                try:
                    header_embedding = embedder.encode([chunk_header], convert_to_numpy=True)[0]
                    
                    # Calculate similarities
                    header_sim = cosine_similarity(query_embedding, header_embedding)
                    content_sim = match.score  # Use Pinecone's content similarity
                    
                    # Combined similarity (weighted average)
                    combined_sim = (header_sim * 0.3) + (content_sim * 0.7)
                    
                    enhanced_results.append({
                        "chunk": chunk_text,
                        "header": chunk_header,
                        "similarity": combined_sim,
                        "header_sim": header_sim,
                        "content_sim": content_sim
                    })
                except Exception as e:
                    print(f"Error calculating enhanced similarity: {e}")
                    # Fallback to original similarity
                    enhanced_results.append({
                        "chunk": chunk_text,
                        "header": chunk_header,
                        "similarity": match.score,
                        "header_sim": match.score,
                        "content_sim": match.score
                    })
            else:
                # Fallback for chunks without headers
                enhanced_results.append({
                    "chunk": chunk_text,
                    "header": "Content",
                    "similarity": match.score,
                    "header_sim": match.score,
                    "content_sim": match.score
                })
        
        # Sort by combined similarity and return top_k
        enhanced_results.sort(key=lambda x: x["similarity"], reverse=True)
        return enhanced_results[:top_k]
        
    except Exception as e:
        print(f"Enhanced search failed, falling back to regular search: {e}")
        # Fallback to regular search
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return [{"chunk": match.metadata["chunk"], "header": "Content", "similarity": match.score} 
                for match in results.matches]

def generate_local_response(context, query):
    """Generate response using local LLM"""
    if not context.strip():
        return "I don't have enough information to answer your question."
    
    # Clean and structure the context
    context_parts = []
    for line in context.split('\n'):
        if line.strip():
            # Remove header formatting and clean up
            if line.startswith('[') and line.endswith(']'):
                continue  # Skip header lines
            context_parts.append(line.strip())
    
    clean_context = '\n'.join(context_parts)
    
    # Better prompt for direct answers
    prompt = f"""You are a helpful assistant. Answer the user's question using only the information provided in the context. 
    Give a direct, concise answer. Do not mention headers, sections, or formatting. Focus on the specific information requested.
    
    If asked about names, provide the person's name clearly.
    If asked about contact information, provide any available contact details.
    If asked about skills, list the specific skills mentioned.
    If asked about achievements, list the key accomplishments.
    
    Context: {clean_context}
    
    Question: {query}
    
    Answer:"""
    
    try:
        response = llm(prompt, max_tokens=300, temperature=0.3, stop=["Question:", "Context:", "\n\n"])
        answer = response['choices'][0]['text'].strip()
        
        # Clean up any remaining tokens
        answer = answer.replace('<|system|>', '').replace('<|user|>', '').replace('<|assistant|>', '')
        answer = answer.replace('Section:', '').replace('Header:', '').replace('Content:', '')
        
        # Remove extra formatting
        answer = re.sub(r'\[.*?\]', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer
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
    """Ingest PDF file and create embeddings with contextual headers"""
    try:
        file_bytes = await file.read()
        
        # Extract text
        text = extract_text(file_bytes, file.filename)
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "No text could be extracted from the file"}
            )
        
        # Create chunks with headers (using simple headers for faster processing)
        chunks_with_headers = chunk_text_with_headers(text, use_llm_headers=False)
        print(f"Created {len(chunks_with_headers)} chunks with headers from {file.filename}")
        
        # Create embeddings for both headers and content
        vectors = []
        for i, chunk_data in enumerate(chunks_with_headers):
            vector_id = str(uuid.uuid4())
            
            # Create embeddings for header and content
            header_embedding = embedder.encode([chunk_data["header"]], convert_to_numpy=True)[0]
            content_embedding = embedder.encode([chunk_data["text"]], convert_to_numpy=True)[0]
            
            metadata = {
                "chunk": chunk_data["text"],
                "header": chunk_data["header"],
                "chunk_index": i,
                "filename": file.filename
            }
            
            if data_source_id:
                metadata["data_source_id"] = data_source_id
            
            # Store content embedding as primary, header text in metadata
            vectors.append({
                "id": vector_id,
                "values": content_embedding.tolist(),  # Primary embedding
                "metadata": metadata  # Only store text metadata, not embeddings
            })
        
        # Upload to Pinecone
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        print(f"[INGEST] Successfully ingested {len(vectors)} chunks with headers")
        return {
            "status": "success",
            "chunks_added": len(vectors),
            "filename": file.filename,
            "with_headers": True
        }
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to ingest file: {str(e)}"}
        )

@app.post("/query")
async def query(req: QueryRequest):
    """Query the RAG system with enhanced retrieval"""
    try:
        # Create query embedding
        query_embedding = embedder.encode([req.query], convert_to_numpy=True)[0]
        
        print(f"[QUERY] Processing query: {req.query}")
        
        # Use enhanced semantic search
        enhanced_results = enhanced_semantic_search(query_embedding, query_embedding, top_k=3)
        
        if not enhanced_results:
            return JSONResponse(
                status_code=404,
                content={"error": "No relevant documents found"}
            )
        
        # Create clean context without header formatting
        context_parts = []
        for result in enhanced_results:
            chunk = result["chunk"]
            # Clean up the chunk text
            clean_chunk = re.sub(r'<\|.*?\|>', '', chunk)  # Remove any remaining tokens
            clean_chunk = re.sub(r'\s+', ' ', clean_chunk).strip()  # Clean whitespace
            if clean_chunk:
                context_parts.append(clean_chunk)
        
        context = "\n\n".join(context_parts)
        if len(context) > 1500:  # Reduced for better responses
            context = context[:1500] + "..."
        
        print(f"[QUERY] Retrieved {len(enhanced_results)} chunks with headers")
        print(f"[QUERY] Context length: {len(context)} characters")
        print(f"[QUERY] Using enhanced retrieval with headers")
        
        # Generate response
        answer = generate_local_response(context, req.query)
        
        print(f"[QUERY] Generated answer: {answer}")
        
        return {
            "answer": answer,
            "context_chunks": len(enhanced_results),
            "query": req.query,
            "llm_type": "local-llama",
            "retrieval_method": "enhanced-with-headers"
        }
        
    except Exception as e:
        print(f"Error during query: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process query: {str(e)}"}
        )

@app.post("/debug-retrieve")
async def debug_retrieve(req: QueryRequest):
    """Debug endpoint to see what chunks are retrieved with headers"""
    try:
        query_embedding = embedder.encode([req.query], convert_to_numpy=True)[0]
        
        enhanced_results = enhanced_semantic_search(query_embedding, query_embedding, top_k=3)
        
        return {
            "query": req.query,
            "retrieved_chunks": [
                {
                    "header": result.get("header", "Content"),
                    "chunk": result["chunk"],
                    "similarity": result["similarity"]
                }
                for result in enhanced_results
            ],
            "num_chunks": len(enhanced_results),
            "retrieval_method": "enhanced-with-headers"
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
        "service": "rag-service-with-cch",
        "features": {
            "contextual_chunk_headers": True,
            "dual_embeddings": True,
            "enhanced_retrieval": True,
            "local_llm": True,
            "pinecone_integration": True
        },
        "llm_type": "local-llama"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002) 