import axios from 'axios';
import { AllFiles } from '../entities/AllFiles';

export class RAGIntegrationService {
  private readonly RAG_SERVICE_URL = 'http://localhost:8002';

  
  async embedFileAfterUpload(file: AllFiles, fileBuffer: Buffer): Promise<void> {
    try {
      console.log(`[RAG] Starting automatic embedding for file: ${file.originalName}`);

      const FormData = require('form-data');
      const formData = new FormData();
      
      formData.append('file', fileBuffer, {
        filename: file.originalName,
        contentType: 'application/pdf'
      });
      
      formData.append('agent_id', 'default');
      formData.append('data_source_id', file.id.toString());

      const response = await axios.post(`${this.RAG_SERVICE_URL}/ingest`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 120000  // Increased to 2 minutes for CCH processing
      });

      if (response.data.status === 'success') {
        console.log(`[RAG] Successfully embedded file ${file.originalName} with ${response.data.chunks_added} chunks`);
      } else {
        console.error(`[RAG] Failed to embed file ${file.originalName}: ${response.data.error}`);
      }

    } catch (error) {
      console.error(`[RAG] Error embedding file ${file.originalName}:`, error);
    }
  }
 
  async deleteEmbeddings(fileId: string): Promise<void> {
    try {
      console.log(`[RAG] Deleting embeddings for file ID: ${fileId}`);

      const response = await axios.post(`${this.RAG_SERVICE_URL}/delete`, {
        agent_id: 'default',
        data_source_id: fileId
      });

      if (response.data.status === 'success') {
        console.log(`[RAG] Successfully deleted embeddings for file ID: ${fileId}`);
      }

    } catch (error) {
      console.error(`[RAG] Error deleting embeddings for file ID ${fileId}:`, error);
    }
  }

 
} 