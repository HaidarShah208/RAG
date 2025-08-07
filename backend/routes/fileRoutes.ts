import { Router } from 'express';
import { FileController } from '../controllers/fileController';
import multer from 'multer';

const router = Router();
const fileController = new FileController();

const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
 
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed'));
    }
  }
});

router.post('/upload', upload.single('file'), fileController.uploadFile.bind(fileController));

router.get('/', fileController.getAllFiles.bind(fileController));

router.get('/:id', fileController.getFileById.bind(fileController));

router.get('/:id/download', fileController.downloadFile.bind(fileController));

router.patch('/:id/description', fileController.updateFileDescription.bind(fileController));

router.delete('/:id', fileController.deleteFile.bind(fileController));

export default router; 