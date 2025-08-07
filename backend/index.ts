import 'reflect-metadata';
import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { initializeDatabase } from './config/database';
import fileRoutes from './routes/fileRoutes';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use((req: Request, res: Response, next: NextFunction) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

 

app.get('/', (req: Request, res: Response) => {
  res.json({ 
    message: 'Welcome to the File Management API!',
    timestamp: new Date().toISOString(),
    status: 'running',
    endpoints: {
      health: '/health',
      files: '/api/files',
      upload: '/api/files/upload'
    }
  });
});


app.use('/api/files', fileRoutes);

app.use('*', (req: Request, res: Response) => {
  res.status(404).json({ 
    error: 'Route not found',
    path: req.originalUrl
  });
});

 


const startServer = async () => {
  try {

    await initializeDatabase();
    
    
    app.listen(PORT, () => {
      console.log(`🚀 Server is running on port ${PORT}`);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
};

startServer();

export default app;
