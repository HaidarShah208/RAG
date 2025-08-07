import { AppDataSource } from '../config/database';
import { AllFiles } from '../entities/AllFiles';
import * as fs from 'fs';
import * as path from 'path';
import { RAGIntegrationService } from './ragIntegrationService';

export class FileService {
  private fileRepository = AppDataSource.getRepository(AllFiles);
  private ragService = new RAGIntegrationService();

  async uploadFile(file: Express.Multer.File, description?: string): Promise<AllFiles> {
    try {
      const uploadsDir = path.join(__dirname, '../uploads');
      if (!fs.existsSync(uploadsDir)) {
        fs.mkdirSync(uploadsDir, { recursive: true });
      }

      const timestamp = Date.now();
      const fileExtension = path.extname(file.originalname);
      const filename = `${timestamp}_${file.originalname}`;
      const filePath = path.join(uploadsDir, filename);

      fs.writeFileSync(filePath, file.buffer);

      const fileRecord = this.fileRepository.create({
        filename,
        originalName: file.originalname,
        mimeType: file.mimetype,
        size: file.size,
        filePath: filePath,
        description: description || '',
        fileType: 'pdf'
      });

      const savedFile = await this.fileRepository.save(fileRecord);
      
      try {
        await this.ragService.embedFileAfterUpload(savedFile, file.buffer);
      } catch (ragError) {
        console.error('RAG embedding failed, but file upload succeeded:', ragError);
      }
      
      return savedFile;
    } catch (error) {
      console.error('Error uploading file:', error);
      throw new Error('Failed to upload file');
    }
  }

  async getAllFiles(): Promise<AllFiles[]> {
    try {
      return await this.fileRepository.find({
        order: { createdAt: 'DESC' }
      });
    } catch (error) {
      console.error('Error fetching files:', error);
      throw new Error('Failed to fetch files');
    }
  }

  async getFileById(id: string): Promise<AllFiles | null> {
    try {
      return await this.fileRepository.findOne({ where: { id } });
    } catch (error) {
      console.error('Error fetching file by ID:', error);
      throw new Error('Failed to fetch file');
    }
  }

  async deleteFile(id: string): Promise<boolean> {
    try {
      const file = await this.fileRepository.findOne({ where: { id } });
      if (!file) {
        return false;
      }

      if (fs.existsSync(file.filePath)) {
        fs.unlinkSync(file.filePath);
      }

      try {
        await this.ragService.deleteEmbeddings(id);
      } catch (ragError) {
        console.error('RAG deletion failed, but file deletion succeeded:', ragError);
      }
      
      await this.fileRepository.remove(file);
      return true;
    } catch (error) {
      console.error('Error deleting file:', error);
      throw new Error('Failed to delete file');
    }
  }

  async updateFileDescription(id: string, description: string): Promise<AllFiles | null> {
    try {
      const file = await this.fileRepository.findOne({ where: { id } });
      if (!file) {
        return null;
      }

      file.description = description;
      return await this.fileRepository.save(file);
    } catch (error) {
      console.error('Error updating file description:', error);
      throw new Error('Failed to update file description');
    }
  }
} 