"""
FastAPI RAG microservice with Local LLM and Document Augmentation:
✅ Pinecone integration for vector storage
✅ Local LLM (llama-cpp) for text generation
✅ PDF text extraction and chunking
✅ Question Generation for Document Augmentation
✅ Enhanced retrieval with chunk + question similarity
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
        
        # Try to break at sentence boundaries
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

def generate_questions_for_chunk(chunk_text, num_questions=3):
    """Generate relevant questions for a text chunk using local LLM"""
    try:
        # Create a prompt to generate questions
        prompt = f"""Generate {num_questions} specific questions that can be answered using only this text chunk. 
        Focus on key information like technologies, skills, projects, and specific details. 
        Make questions specific and avoid generic questions. Return only the questions, one per line.
        
        Text chunk: {chunk_text[:500]}...
        
        Questions:"""
        
        response = llm(prompt, max_tokens=150, temperature=0.7, stop=["Text chunk:", "Context:"])
        questions_text = response['choices'][0]['text'].strip()
        
        # Extract questions from response
        questions = []
        for line in questions_text.split('\n'):
            line = line.strip()
            # Remove numbering and clean up
            line = re.sub(r'^\d+\.\s*', '', line)
            if line and line.endswith('?'):
                questions.append(line)
        
        # If LLM generation failed, create simple questions
        if not questions:
            sentences = re.split(r'[.!?]', chunk_text)
            first_sentence = sentences[0].strip() if sentences else chunk_text[:100]
            questions = [f"What information is mentioned about {first_sentence[:30]}...?"]
        
        return questions[:num_questions]  # Limit to requested number
        
    except Exception as e:
        print(f"Question generation failed: {e}")
        # Fallback to simple questions
        return [f"What information is mentioned in this text?"]

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def enhanced_semantic_search_with_questions(query, query_embedding, top_k=3):
    """Enhanced search using both chunks and generated questions"""
    try:
        # Adjust top_k based on query type for more focused results
        if "technolog" in query.lower() or "tech" in query.lower():
            search_top_k = top_k * 3  # More results for technology questions
        else:
            search_top_k = top_k * 2  # Default multiplier
        
        # Get results from Pinecone
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=search_top_k,
            include_metadata=True
        )
        
        if not results.matches:
            return []
        
        # Re-rank using combined similarity (chunks + questions)
        enhanced_results = []
        for match in results.matches:
            chunk_text = match.metadata.get("chunk", "")
            question_text = match.metadata.get("question", "")
            item_type = match.metadata.get("type", "chunk")
            
            try:
                if item_type == "question" and question_text:
                    # For questions, calculate similarity to query
                    question_embedding = embedder.encode([question_text], convert_to_numpy=True)[0]
                    question_sim = cosine_similarity(query_embedding, question_embedding)
                    
                    enhanced_results.append({
                        "chunk": chunk_text,
                        "question": question_text,
                        "type": "question",
                        "similarity": question_sim,
                        "original_similarity": match.score
                    })
                else:
                    # For chunks, use original similarity
                    enhanced_results.append({
                        "chunk": chunk_text,
                        "question": "",
                        "type": "chunk",
                        "similarity": match.score,
                        "original_similarity": match.score
                    })
                    
            except Exception as e:
                print(f"Error calculating enhanced similarity: {e}")
                # Fallback to original similarity
                enhanced_results.append({
                    "chunk": chunk_text,
                    "question": question_text,
                    "type": item_type,
                    "similarity": match.score,
                    "original_similarity": match.score
                })
        
        # Sort by similarity and return top_k
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
        return [{"chunk": match.metadata["chunk"], "question": "", "type": "chunk", "similarity": match.score} 
                for match in results.matches]

def generate_local_response(context, query):
    """Generate response using local LLM"""
    if not context.strip():
        return "I don't have enough information to answer your question."
    
    # Clean and structure the context
    context_parts = []
    for line in context.split('\n'):
        if line.strip():
            # Remove any formatting artifacts
            clean_line = re.sub(r'<\|.*?\|>', '', line.strip())
            if clean_line:
                context_parts.append(clean_line)
    
    clean_context = '\n'.join(context_parts)
    
    # Better prompt for direct answers
    prompt = f"""You are a helpful assistant. Answer the user's question using only the information provided in the context. 
    Give a direct, concise answer. Do not mention headers, sections, or formatting. Focus on the specific information requested.
    
    IMPORTANT RULES:
    - If asked about technologies, list ONLY the technologies mentioned, no repetition
    - If asked about specific projects, focus only on that project
    - If asked about structure/architecture, provide a clear, organized response
    - Do not repeat the same information multiple times
    - Keep responses focused and relevant to the question
    
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
        
        # Remove duplicate technologies if it's a technology question
        if "technolog" in query.lower() or "tech" in query.lower():
            # Split by common separators and remove duplicates
            tech_list = re.split(r'[,.\s]+', answer)
            unique_tech = []
            seen = set()
            for tech in tech_list:
                tech = tech.strip()
                if tech and tech not in seen and len(tech) > 2:
                    seen.add(tech)
                    unique_tech.append(tech)
            answer = ', '.join(unique_tech)
        
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
    """Ingest PDF file and create embeddings with question augmentation"""
    try:
        file_bytes = await file.read()
        
        # Extract text
        text = extract_text(file_bytes, file.filename)
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "No text could be extracted from the file"}
            )
        
        # Create chunks
        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks from {file.filename}")
        
        # Create embeddings for chunks and questions
        vectors = []
        for i, chunk in enumerate(chunks):
            vector_id = str(uuid.uuid4())
            
            # Create embedding for chunk
            chunk_embedding = embedder.encode([chunk], convert_to_numpy=True)[0]
            
            # Store chunk
            metadata = {
                "chunk": chunk,
                "chunk_index": i,
                "filename": file.filename,
                "type": "chunk"
            }
            
            if data_source_id:
                metadata["data_source_id"] = data_source_id
            
            vectors.append({
                "id": vector_id,
                "values": chunk_embedding.tolist(),
                "metadata": metadata
            })
            
            # Generate questions for this chunk
            questions = generate_questions_for_chunk(chunk, num_questions=3)
            
            # Create embeddings for questions
            for j, question in enumerate(questions):
                question_vector_id = str(uuid.uuid4())
                question_embedding = embedder.encode([question], convert_to_numpy=True)[0]
                
                question_metadata = {
                    "chunk": chunk,  # Store original chunk
                    "question": question,
                    "chunk_index": i,
                    "question_index": j,
                    "filename": file.filename,
                    "type": "question"
                }
                
                if data_source_id:
                    question_metadata["data_source_id"] = data_source_id
                
                vectors.append({
                    "id": question_vector_id,
                    "values": question_embedding.tolist(),
                    "metadata": question_metadata
                })
        
        # Upload to Pinecone
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        print(f"[INGEST] Successfully ingested {len(chunks)} chunks and {len(vectors) - len(chunks)} questions")
        return {
            "status": "success",
            "chunks_added": len(chunks),
            "questions_added": len(vectors) - len(chunks),
            "total_vectors": len(vectors),
            "filename": file.filename,
            "with_question_augmentation": True
        }
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to ingest file: {str(e)}"}
        )

@app.post("/query")
async def query(req: QueryRequest):
    """Query the RAG system with enhanced retrieval using question augmentation"""
    try:
        # Create query embedding
        query_embedding = embedder.encode([req.query], convert_to_numpy=True)[0]
        
        print(f"[QUERY] Processing query: {req.query}")
        
        # Use enhanced semantic search with questions
        enhanced_results = enhanced_semantic_search_with_questions(query_embedding, query_embedding, top_k=3)
        
        if not enhanced_results:
            return JSONResponse(
                status_code=404,
                content={"error": "No relevant documents found"}
            )
        
        # Create clean context from chunks and questions
        context_parts = []
        for result in enhanced_results:
            chunk = result["chunk"]
            question = result.get("question", "")
            
            # Clean up the chunk text
            clean_chunk = re.sub(r'<\|.*?\|>', '', chunk)
            clean_chunk = re.sub(r'\s+', ' ', clean_chunk).strip()
            
            if clean_chunk:
                if question:
                    context_parts.append(f"Question: {question}\nAnswer: {clean_chunk}")
                else:
                    context_parts.append(clean_chunk)
        
        context = "\n\n".join(context_parts)
        
        # Dynamic context length based on query type
        if "technologies" in req.query.lower() or "tech" in req.query.lower():
            max_context = 800  # Shorter for technology questions
        elif "structure" in req.query.lower() or "architecture" in req.query.lower():
            max_context = 1200  # Medium for structure questions
        else:
            max_context = 1000  # Default length
        
        if len(context) > max_context:
            context = context[:max_context] + "..."
        
        print(f"[QUERY] Retrieved {len(enhanced_results)} items (chunks + questions)")
        print(f"[QUERY] Context length: {len(context)} characters")
        print(f"[QUERY] Using question-augmented retrieval")
        
        # Generate response
        answer = generate_local_response(context, req.query)
        
        print(f"[QUERY] Generated answer: {answer}")
        
        return {
            "answer": answer,
            "context_items": len(enhanced_results),
            "query": req.query,
            "llm_type": "local-llama",
            "retrieval_method": "question-augmented"
        }
        
    except Exception as e:
        print(f"Error during query: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process query: {str(e)}"}
        )

@app.post("/debug-retrieve")
async def debug_retrieve(req: QueryRequest):
    """Debug endpoint to see what chunks and questions are retrieved"""
    try:
        query_embedding = embedder.encode([req.query], convert_to_numpy=True)[0]
        
        enhanced_results = enhanced_semantic_search_with_questions(query_embedding, query_embedding, top_k=3)
        
        return {
            "query": req.query,
            "retrieved_items": [
                {
                    "type": result["type"],
                    "chunk": result["chunk"][:200] + "..." if len(result["chunk"]) > 200 else result["chunk"],
                    "question": result.get("question", ""),
                    "similarity": result["similarity"]
                }
                for result in enhanced_results
            ],
            "num_items": len(enhanced_results),
            "retrieval_method": "question-augmented"
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
        "service": "rag-service-with-question-augmentation",
        "features": {
            "question_generation": True,
            "enhanced_retrieval": True,
            "local_llm": True,
            "pinecone_integration": True
        },
        "llm_type": "local-llama"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002) 