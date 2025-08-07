import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './components/Home';
import PDFUpload from './components/PDFUpload';
import Chat from './components/Chat';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<PDFUpload />} />
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
