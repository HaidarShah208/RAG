import { AppDataSource } from '../config/database';

const runMigrations = async () => {
  try {
    await AppDataSource.initialize();
    console.log('✅ Database connection established');
    
    await AppDataSource.runMigrations();
    console.log('✅ All migrations completed successfully');
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Migration failed:', error);
    process.exit(1);
  }
};

runMigrations(); 