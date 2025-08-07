# File Management API

A robust Express.js server built with TypeScript, TypeORM, and PostgreSQL for PDF file management.

## 🏗️ Architecture

```
backend/
├── config/
│   └── database.ts          # TypeORM database configuration
├── controllers/
│   └── fileController.ts    # HTTP request handlers
├── services/
│   └── fileService.ts       # Business logic layer
├── routes/
│   └── fileRoutes.ts        # API route definitions
├── entities/
│   └── AllFiles.ts          # TypeORM entity for database
└── index.ts                 # Main server file
```

## ✨ Features

- ✅ **TypeORM** with PostgreSQL integration
- ✅ **PDF file upload** with validation
- ✅ **File storage** on disk with database metadata
- ✅ **RESTful API** for file management
- ✅ **TypeScript** with decorators support
- ✅ **CORS** enabled for frontend integration
- ✅ **Error handling** and logging
- ✅ **File download** functionality
- ✅ **File description** management

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Database Setup
Create a PostgreSQL database and update your `.env` file:

```env
# Server Configuration
PORT=3000
NODE_ENV=development

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=postgres
DB_PASSWORD=your_password
DB_NAME=promptEngineering
```

### 3. Run Migrations
```bash
# Run database migrations
npm run migrate

# Or use TypeORM CLI
npm run migration:run
```

### 4. Run the Server
```bash
# Development mode
npm run dev

# Production mode
npm run build
npm start
```

## 📁 API Endpoints

### File Management

#### `POST /api/files/upload`
Upload a PDF file with optional description.

**Request:**
```bash
curl -X POST http://localhost:3000/api/files/upload \
  -F "file=@document.pdf" \
  -F "description=My important document"
```

**Response:**
```json
{
  "success": true,
  "message": "File uploaded successfully",
  "data": {
    "id": "uuid",
    "filename": "1234567890_document.pdf",
    "originalName": "document.pdf",
    "size": 1024000,
    "description": "My important document",
    "createdAt": "2024-01-01T12:00:00.000Z"
  }
}
```

#### `GET /api/files`
Get all uploaded files.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "uuid",
      "filename": "1234567890_document.pdf",
      "originalName": "document.pdf",
      "size": 1024000,
      "description": "My important document",
      "fileType": "pdf",
      "createdAt": "2024-01-01T12:00:00.000Z",
      "updatedAt": "2024-01-01T12:00:00.000Z"
    }
  ]
}
```

#### `GET /api/files/:id`
Get file details by ID.

#### `GET /api/files/:id/download`
Download a file by ID.

#### `PATCH /api/files/:id/description`
Update file description.

**Request:**
```bash
curl -X PATCH http://localhost:3000/api/files/uuid/description \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated description"}'
```

#### `DELETE /api/files/:id`
Delete a file by ID.

### Health Check

#### `GET /health`
Check server and database status.

## 🗄️ Database Schema

The `all_files` table structure:

```sql
CREATE TABLE all_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename VARCHAR(255) NOT NULL,
  original_name VARCHAR(100) NOT NULL,
  mime_type VARCHAR(50) NOT NULL,
  size INTEGER NOT NULL,
  file_path TEXT NOT NULL,
  description VARCHAR(255),
  file_type VARCHAR(100) DEFAULT 'pdf',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `3000` |
| `NODE_ENV` | Environment | `development` |
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `5432` |
| `DB_USERNAME` | Database username | `postgres` |
| `DB_PASSWORD` | Database password | `password` |
| `DB_NAME` | Database name | `file_management` |

### File Upload Limits

- **File size**: 10MB maximum
- **File type**: PDF only
- **Storage**: Local disk with database metadata

## 🛠️ Development

### Project Structure
- **Controllers**: Handle HTTP requests/responses
- **Services**: Business logic and database operations
- **Routes**: API endpoint definitions
- **Entities**: TypeORM database models
- **Config**: Database and server configuration

### Running Tests
```bash
npm test
```

### Building for Production
```bash
npm run build
```

## 🗄️ Migration Management

### Available Migration Commands

```bash
# Run all pending migrations
npm run migrate

# Generate a new migration (after entity changes)
npm run migration:generate -- src/migrations/MigrationName

# Create an empty migration
npm run migration:create -- src/migrations/MigrationName

# Revert the last migration
npm run migration:revert

# Run migrations using TypeORM CLI
npm run migration:run
```

### Migration Files
- **Location**: `migrations/` folder
- **Naming**: `timestamp-MigrationName.ts`
- **Example**: `1703123456789-CreateAllFilesTable.ts`

## 🔒 Security Features

- ✅ File type validation (PDF only)
- ✅ File size limits (10MB)
- ✅ CORS configuration
- ✅ Input validation
- ✅ Error handling

## 📝 Frontend Integration

Example frontend code for file upload:

```javascript
const uploadFile = async (file, description) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('description', description);

  const response = await fetch('http://localhost:3000/api/files/upload', {
    method: 'POST',
    body: formData
  });

  return response.json();
};
```

The server will start on port 3000 by default and automatically create the database table on first run. 