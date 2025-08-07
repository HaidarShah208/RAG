import { Request, Response } from 'express';
import { FileService } from '../services/fileService';
import * as fs from 'fs';
import * as path from 'path';

export class FileController {
  private fileService = new FileService();

  // Upload PDF file
  async uploadFile(req: Request, res: Response) {
    try {
      if (!req.file) {
        return res.status(400).json({ 
          success: false, 
          message: 'No file uploaded' 
        });
      }

      // Check if file is PDF
      if (req.file.mimetype !== 'application/pdf') {
        return res.status(400).json({ 
          success: false, 
          message: 'Only PDF files are allowed' 
        });
      }

      const description = req.body.description || '';
      const uploadedFile = await this.fileService.uploadFile(req.file, description);

      res.status(201).json({
        success: true,
        message: 'File uploaded successfully',
        data: {
          id: uploadedFile.id,
          filename: uploadedFile.filename,
          originalName: uploadedFile.originalName,
          size: uploadedFile.size,
          description: uploadedFile.description,
          createdAt: uploadedFile.createdAt
        }
      });
    } catch (error) {
      console.error('Upload error:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to upload file',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Get all files
  async getAllFiles(req: Request, res: Response) {
    try {
      const files = await this.fileService.getAllFiles();
      
      res.json({
        success: true,
        data: files.map(file => ({
          id: file.id,
          filename: file.filename,
          originalName: file.originalName,
          size: file.size,
          description: file.description,
          fileType: file.fileType,
          createdAt: file.createdAt,
          updatedAt: file.updatedAt
        }))
      });
    } catch (error) {
      console.error('Get files error:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to fetch files',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Get file by ID
  async getFileById(req: Request, res: Response) {
    try {
      const { id } = req.params;
      const file = await this.fileService.getFileById(id);

      if (!file) {
        return res.status(404).json({
          success: false,
          message: 'File not found'
        });
      }

      res.json({
        success: true,
        data: {
          id: file.id,
          filename: file.filename,
          originalName: file.originalName,
          size: file.size,
          description: file.description,
          fileType: file.fileType,
          createdAt: file.createdAt,
          updatedAt: file.updatedAt
        }
      });
    } catch (error) {
      console.error('Get file error:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to fetch file',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Download file
  async downloadFile(req: Request, res: Response) {
    try {
      const { id } = req.params;
      const file = await this.fileService.getFileById(id);

      if (!file) {
        return res.status(404).json({
          success: false,
          message: 'File not found'
        });
      }

      if (!fs.existsSync(file.filePath)) {
        return res.status(404).json({
          success: false,
          message: 'File not found on disk'
        });
      }

      res.download(file.filePath, file.originalName);
    } catch (error) {
      console.error('Download error:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to download file',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Delete file
  async deleteFile(req: Request, res: Response) {
    try {
      const { id } = req.params;
      const deleted = await this.fileService.deleteFile(id);

      if (!deleted) {
        return res.status(404).json({
          success: false,
          message: 'File not found'
        });
      }

      res.json({
        success: true,
        message: 'File deleted successfully'
      });
    } catch (error) {
      console.error('Delete error:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to delete file',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Update file description
  async updateFileDescription(req: Request, res: Response) {
    try {
      const { id } = req.params;
      const { description } = req.body;

      if (!description) {
        return res.status(400).json({
          success: false,
          message: 'Description is required'
        });
      }

      const updatedFile = await this.fileService.updateFileDescription(id, description);

      if (!updatedFile) {
        return res.status(404).json({
          success: false,
          message: 'File not found'
        });
      }

      res.json({
        success: true,
        message: 'File description updated successfully',
        data: {
          id: updatedFile.id,
          description: updatedFile.description,
          updatedAt: updatedFile.updatedAt
        }
      });
    } catch (error) {
      console.error('Update description error:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to update file description',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }
} 