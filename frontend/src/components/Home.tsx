import { useNavigate } from 'react-router-dom';
import FileList from './FileList';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <div className="home-card">
        <h1>Welcome to PDF Upload App</h1>
        <p>Upload and manage your PDF files easily</p>
        
        <div className="button-group">
          <button 
            onClick={() => navigate('/upload')}
            className="upload-nav-button"
          >
            Upload PDF
          </button>
          
          <button 
            onClick={() => navigate('/chat')}
            className="chat-nav-button"
          >
            RAG Chat
          </button>
        </div>
      </div>

      <div className="files-section">
        <FileList />
      </div>
    </div>
  );
};

export default Home; 