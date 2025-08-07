import { useState, useEffect } from 'react';

interface FileData {
  id: string;
  filename: string;
  originalName: string;
  size: number;
  description: string;
  fileType: string;
  createdAt: string;
  updatedAt: string;
}

const FileList = () => {
  const [files, setFiles] = useState<FileData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/api/files');
      const result = await response.json();

      if (response.ok) {
        setFiles(result.data);
      } else {
        setError('Failed to fetch files');
      }
    } catch (error) {
      console.error('Error fetching files:', error);
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (fileId: string, originalName: string) => {
    try {
      const response = await fetch(`http://localhost:5000/api/files/${fileId}/download`);
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = originalName;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert('Failed to download file');
      }
    } catch (error) {
      console.error('Download error:', error);
      alert('Download failed. Please try again.');
    }
  };

  const handleDelete = async (fileId: string) => {
    if (!confirm('Are you sure you want to delete this file?')) {
      return;
    }

    try {
      const response = await fetch(`http://localhost:5000/api/files/${fileId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setFiles(files.filter(file => file.id !== fileId));
      } else {
        alert('Failed to delete file');
      }
    } catch (error) {
      console.error('Delete error:', error);
      alert('Delete failed. Please try again.');
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="file-list-container">
        <div className="loading-message">
          <div className="loading-spinner"></div>
          <p>Loading files...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="file-list-container">
        <div className="error-message">
          <p>{error}</p>
          <button onClick={fetchFiles} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="file-list-container">
      <div className="file-list-header">
        <h3>Uploaded Files ({files.length})</h3>
        <button onClick={fetchFiles} className="refresh-button">
          Refresh
        </button>
      </div>

      {files.length === 0 ? (
        <div className="no-files-message">
          <p>No files uploaded yet.</p>
        </div>
      ) : (
        <div className="file-grid">
          {files.map((file) => (
            <div key={file.id} className="file-card">
              <div className="file-icon">
                📄
              </div>
              <div className="file-info">
                <h4 className="file-name">{file.originalName}</h4>
                <p className="file-size">{formatFileSize(file.size)}</p>
                {file.description && (
                  <p className="file-description">{file.description}</p>
                )}
                <p className="file-date">Uploaded: {formatDate(file.createdAt)}</p>
              </div>
              <div className="file-actions">
                <button
                  onClick={() => handleDownload(file.id, file.originalName)}
                  className="download-button"
                  title="Download file"
                >
                  📥
                </button>
                <button
                  onClick={() => handleDelete(file.id)}
                  className="delete-button"
                  title="Delete file"
                >
                  🗑️
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default FileList; 