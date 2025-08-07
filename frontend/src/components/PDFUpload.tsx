import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const PDFUpload = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [description, setDescription] = useState<string>('');
  const navigate = useNavigate();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type === 'application/pdf') {
        setSelectedFile(file);
        setUploadStatus('File selected successfully!');
      } else {
        setUploadStatus('Please select a PDF file only!');
        setSelectedFile(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('Please select a file first!');
      return;
    }

    setIsUploading(true);
    setUploadStatus('Uploading to database...');
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      if (description.trim()) {
        formData.append('description', description.trim());
      }

      const response = await fetch('http://localhost:5000/api/files/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadStatus(`File uploaded successfully! File ID: ${result.data.id}`);
        setSelectedFile(null);
        setDescription('');
        const fileInput = document.getElementById('pdf-file-input') as HTMLInputElement;
        if (fileInput) {
          fileInput.value = '';
        }
      } else {
        setUploadStatus(`Upload failed: ${result.message}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus('Upload failed: Network error. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="pdf-upload-container">
      <div className="upload-card">
        <h2>PDF Upload</h2>
        <p>Select a PDF file to upload to the database</p>
        
        <div className="file-input-container">
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileSelect}
            id="pdf-file-input"
            className="file-input"
            disabled={isUploading}
          />
          <label htmlFor="pdf-file-input" className="file-input-label">
            Choose PDF File
          </label>
        </div>

        {selectedFile && (
          <div className="file-info">
            <p><strong>Selected File:</strong> {selectedFile.name}</p>
            <p><strong>Size:</strong> {(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        )}

        <div className="description-container">
          <label htmlFor="description-input">Description (optional):</label>
          <textarea
            id="description-input"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Enter a description for this PDF..."
            className="description-input"
            disabled={isUploading}
          />
        </div>

        <div className="upload-actions">
          <button 
            onClick={handleUpload}
            disabled={!selectedFile || isUploading}
            className="upload-button"
          >
            {isUploading ? 'Uploading...' : 'Upload to Database'}
          </button>
          
          <button 
            onClick={() => navigate('/')}
            className="back-button"
            disabled={isUploading}
          >
            Back to Home
          </button>
        </div>

        {uploadStatus && (
          <div className={`status-message ${uploadStatus.includes('successfully') ? 'success' : uploadStatus.includes('Please') || uploadStatus.includes('failed') ? 'error' : 'info'}`}>
            {uploadStatus}
          </div>
        )}
      </div>
    </div>
  );
};

export default PDFUpload; 