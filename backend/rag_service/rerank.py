"""
FastAPI RAG microservice with Local LLM and Reranking:
✅ Pinecone integration for vector storage
✅ Local LLM (llama-cpp) for text generation
✅ PDF text extraction and chunking
✅ Vector embedding and storage
✅ Two-step retrieval with reranking
✅ LLM-based and keyword-based reranking
✅ Query processing with enhanced RAG
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

def rerank_with_llm(query, results, top_n=3):
    """
    Reranks search results using LLM relevance scoring.
    
    Args:
        query (str): User query
        results (List[Dict]): Initial search results
        top_n (int): Number of results to return after reranking
        
    Returns:
        List[Dict]: Reranked results
    """
    print(f"[RERANK] LLM reranking {len(results)} documents...")
    
    scored_results = []
    
    for i, result in enumerate(results):
        if i % 3 == 0:
            print(f"[RERANK] Scoring document {i+1}/{len(results)}...")
        
        try:
            # Create prompt for LLM scoring
            prompt = f"""Rate how well this document answers the query on a scale from 0 to 10.
            Return only a single number (0-10).
            
            Query: {query}
            
            Document: {result['chunk'][:500]}...
            
            Score:"""
            
            response = llm(prompt, max_tokens=5, temperature=0.1, stop=["\n", "Query:", "Document:"])
            score_text = response['choices'][0]['text'].strip()
            
            # Extract numerical score
            score_match = re.search(r'\b(10|[0-9])\b', score_text)
            if score_match:
                score = float(score_match.group(1))
            else:
                # Fallback to similarity score
                score = result.get("similarity", 0.5) * 10
            
            scored_results.append({
                "chunk": result["chunk"],
                "similarity": result.get("similarity", 0.5),
                "relevance_score": score,
                "metadata": result.get("metadata", {})
            })
            
        except Exception as e:
            print(f"[RERANK] Error scoring document {i}: {e}")
            # Fallback to original similarity
            scored_results.append({
                "chunk": result["chunk"],
                "similarity": result.get("similarity", 0.5),
                "relevance_score": result.get("similarity", 0.5) * 10,
                "metadata": result.get("metadata", {})
            })
    
    # Sort by relevance score and return top_n
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)
    return reranked_results[:top_n]

def rerank_with_keywords(query, results, top_n=3):
    """
    Reranks search results using keyword matching and position scoring.
    
    Args:
        query (str): User query
        results (List[Dict]): Initial search results
        top_n (int): Number of results to return after reranking
        
    Returns:
        List[Dict]: Reranked results
    """
    print(f"[RERANK] Keyword reranking {len(results)} documents...")
    
    # Extract important keywords from query (more intelligent extraction)
    query_lower = query.lower()
    keywords = []
    
    # Professional approach: Use all query words for semantic matching
    for word in query.split():
        cleaned_word = word.lower().strip('.,!?;:')
        if len(cleaned_word) > 1:  # Include all meaningful words
            keywords.append(cleaned_word)
    
    print(f"[RERANK] Using keywords: {keywords}")
    
    scored_results = []
    
    for result in results:
        document_text = result["chunk"].lower()
        
        # Base score from vector similarity (reduced weight)
        base_score = result.get("similarity", 0.5) * 0.3
        
        # Calculate keyword score with better logic
        keyword_score = 0
        matched_keywords = 0
        
        for keyword in keywords:
            if keyword in document_text:
                matched_keywords += 1
                # More points for keyword presence
                keyword_score += 0.2
                
                # Bonus for early position (first 25% of text)
                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:
                    keyword_score += 0.15
                
                # Bonus for frequency but cap it
                frequency = document_text.count(keyword)
                keyword_score += min(0.1 * frequency, 0.3)
        
        # Bonus for matching multiple keywords
        if matched_keywords > 1:
            keyword_score += 0.3 * (matched_keywords - 1)
        
        # Penalty for very long chunks that might be less specific
        if len(result["chunk"]) > 800:
            keyword_score *= 0.8
        
        final_score = base_score + keyword_score
        
        scored_results.append({
            "chunk": result["chunk"],
            "similarity": result.get("similarity", 0.5),
            "relevance_score": final_score,
            "metadata": result.get("metadata", {}),
            "matched_keywords": matched_keywords
        })
    
    # Sort by relevance score and return top_n
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)
    
    # Debug output
    for i, result in enumerate(reranked_results[:top_n]):
        print(f"[RERANK] Rank {i+1}: score={result['relevance_score']:.3f}, matched_keywords={result.get('matched_keywords', 0)}, chunk_preview={result['chunk'][:100]}...")
    
    return reranked_results[:top_n]

def generate_local_response(context, query):
    """Generate response using local LLM with improved prompting"""
    if not context.strip():
        return "I don't have enough information to answer your question."
    
    # Clean up context formatting
    clean_context = re.sub(r'(Header:|Content:|Section:)', '', context)
    clean_context = re.sub(r'\n+', '\n', clean_context).strip()
    
    try:
        # Completely neutral prompt
        prompt = f"""Answer the question using the provided context.

Context: {clean_context}

Question: {query}

Answer:"""
        
        # Simple, adaptive token limit
        max_tokens = 120
        
        response = llm(prompt, max_tokens=max_tokens, temperature=0.1, stop=["Question:", "Context:", "\n\n"])
        answer = response['choices'][0]['text'].strip()
        
        # Minimal cleanup
        answer = re.sub(r'<\|.*?\|>', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer if answer else "I don't have enough information to answer your question."
        
    except Exception as e:
        print(f"LLM error: {e}")
        return f"Error generating response: {str(e)}"



class QueryRequest(BaseModel):
    query: str
    reranking_method: str = "keyword"  # Options: "none", "llm", "keyword"

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
    """Query the RAG system with two-step retrieval and reranking"""
    try:
        # Create query embedding
        query_embedding = embedder.encode([req.query], convert_to_numpy=True)[0]
        
        print(f"[QUERY] Processing query: {req.query}")
        print(f"[QUERY] Reranking method: {req.reranking_method}")
        
        # Step 1: Initial retrieval - professional approach
        if req.reranking_method == "none":
            top_k_initial = 3 
        else:
            top_k_initial = 12  # Standard amount for all queries
        
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k_initial,
            include_metadata=True
        )
        
        if not results.matches:
            return JSONResponse(
                status_code=404,
                content={"error": "No relevant documents found"}
            )
        
        # Convert Pinecone results to our format
        initial_results = []
        for match in results.matches:
            initial_results.append({
                "chunk": match.metadata["chunk"],
                "similarity": match.score,
                "metadata": match.metadata
            })
        
        print(f"[QUERY] Initial retrieval: {len(initial_results)} chunks")
        
        # Step 2: Reranking (if enabled)
        if req.reranking_method == "llm":
            final_results = rerank_with_llm(req.query, initial_results, top_n=3)
            retrieval_method = "llm-reranked"
        elif req.reranking_method == "keyword":
            final_results = rerank_with_keywords(req.query, initial_results, top_n=3)
            retrieval_method = "keyword-reranked"
        else:
            # No reranking
            final_results = initial_results[:3]
            retrieval_method = "standard"
        
        print(f"[QUERY] Final results after reranking: {len(final_results)} chunks")
        
        # Create context from final results
        retrieved_chunks = [result["chunk"] for result in final_results]
        context = "\n\n".join(retrieved_chunks)
        
        # Simple context management
        max_context = 1000
        
        if len(context) > max_context:
            context = context[:max_context] + "..."
        
        print(f"[QUERY] Context length: {len(context)} characters")
        print(f"[QUERY] Retrieval method: {retrieval_method}")
        
        # Generate response
        answer = generate_local_response(context, req.query)
        
        print(f"[QUERY] Generated answer: {answer}")
        
        # Prepare relevance scores for response
        relevance_scores = [result.get("relevance_score", result.get("similarity", 0)) for result in final_results]
        
        return {
            "answer": answer,
            "context_chunks": len(final_results),
            "query": req.query,
            "llm_type": "local-llama",
            "retrieval_method": retrieval_method,
            "reranking_method": req.reranking_method,
            "relevance_scores": relevance_scores
        }
        
    except Exception as e:
        print(f"Error during query: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process query: {str(e)}"}
        )

@app.post("/debug-retrieve")
async def debug_retrieve(req: QueryRequest):
    """Debug endpoint to see what chunks are retrieved with reranking details"""
    try:
        query_embedding = embedder.encode([req.query], convert_to_numpy=True)[0]
        
        # Get initial results
        top_k_initial = 10 if req.reranking_method != "none" else 3
        
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k_initial,
            include_metadata=True
        )
        
        # Convert to our format
        initial_results = []
        for match in results.matches:
            initial_results.append({
                "chunk": match.metadata["chunk"][:200] + "..." if len(match.metadata["chunk"]) > 200 else match.metadata["chunk"],
                "similarity": match.score,
                "metadata": match.metadata
            })
        
        # Apply reranking if requested
        if req.reranking_method == "llm":
            final_results = rerank_with_llm(req.query, initial_results, top_n=3)
        elif req.reranking_method == "keyword":
            final_results = rerank_with_keywords(req.query, initial_results, top_n=3)
        else:
            final_results = initial_results[:3]
        
        return {
            "query": req.query,
            "reranking_method": req.reranking_method,
            "initial_results": [
                {
                    "chunk": result["chunk"],
                    "similarity": result["similarity"]
                }
                for result in initial_results
            ],
            "final_results": [
                {
                    "chunk": result["chunk"],
                    "similarity": result.get("similarity", 0),
                    "relevance_score": result.get("relevance_score", result.get("similarity", 0))
                }
                for result in final_results
            ],
            "num_initial": len(initial_results),
            "num_final": len(final_results)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Debug retrieve failed: {str(e)}"}
        )
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002) 