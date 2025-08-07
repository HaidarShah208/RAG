import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios'; 
import { Send, Upload, FileText, Loader2 } from 'lucide-react'; 
import './Chat.css';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (text: string, isUser: boolean) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      text,
      isUser,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage = inputText.trim();
    setInputText('');
    addMessage(userMessage, true);
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8002/query', {
        query: userMessage,
      });

      const botMessage = response.data.answer;
      addMessage(botMessage, false);
    } catch (error) {
      console.error('Error sending message:', error);
      addMessage('Sorry, I encountered an error while processing your request. Please try again.', false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);


    try {
      const response = await axios.post('http://localhost:8002/ingest', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.status === 'success') {
        setUploadedFiles(prev => [...prev, file.name]);
        addMessage(`✅ Successfully uploaded and processed "${file.name}" (${response.data.chunks_added} chunks created)`, false);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      addMessage(`❌ Failed to upload "${file.name}". Please try again.`, false);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>RAG Chat Interface</h2>
        <div className="upload-section">
          <label htmlFor="file-upload" className="upload-button">
            {isUploading ? (
              <Loader2 className="upload-icon spinning" />
            ) : (
              <Upload className="upload-icon" />
            )}
            Upload PDF
          </label>
          <input
            id="file-upload"
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <FileText className="welcome-icon" />
            <h3>Welcome to RAG Chat!</h3>
            <p>Upload a PDF document and start asking questions about its content.</p>
            <p>I'll use the document to provide accurate answers to your queries.</p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.isUser ? 'user-message' : 'bot-message'}`}
          >
            <div className="message-content">
              {message.text}
            </div>
            <div className="message-timestamp">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="message bot-message">
            <div className="message-content">
              <Loader2 className="loading-icon spinning" />
              Thinking...
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your uploaded documents..."
            disabled={isLoading}
            rows={1}
            className="message-input"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputText.trim() || isLoading}
            className="send-button"
          >
            <Send className="send-icon" />
          </button>
        </div>
      </div>

      {uploadedFiles.length > 0 && (
        <div className="uploaded-files">
          <h4>Uploaded Files:</h4>
          <div className="file-list">
            {uploadedFiles.map((filename, index) => (
              <div key={index} className="file-item">
                <FileText className="file-icon" />
                {filename}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Chat; 