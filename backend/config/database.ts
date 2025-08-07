import { DataSource } from 'typeorm';
import { AllFiles } from '../entities/AllFiles';
import dotenv from 'dotenv';

dotenv.config();

export const AppDataSource = new DataSource({
  type: 'postgres',
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '5432'),
  username: process.env.DB_USERNAME || 'postgres',
  password: process.env.DB_PASSWORD || 'root',
  database: process.env.DB_NAME || 'promptEngineering',
  synchronize: false,
  logging: process.env.NODE_ENV === 'development',
  entities: [AllFiles],
  subscribers: [],
  migrations: ['migrations/*.ts'],
  migrationsTableName: 'migrations'
});

export const initializeDatabase = async () => {
  try {
    await AppDataSource.initialize();
    console.log('✅ Database connection established');
    
    await AppDataSource.runMigrations();
  } catch (error) {
    console.error('❌ Database connection failed:', error);
    throw error;
  }
}; 