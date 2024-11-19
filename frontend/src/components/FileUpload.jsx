
import React, { useState } from 'react';
import axios from '../services/api'; // Ensure this path is correct

const FileUpload = ({ onResult }) => {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileUpload = async () => {
    // Validate file inputs
    if (!file1 || !file2) {
      setError('Please upload two files.');
      return;
    }
    setError('');
    setLoading(true);
    onResult(null);
    // Create FormData object for file uploads
    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);
    console.log([...formData.entries()]);
    try {
      const response = await axios.post('/api/compare_documents', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      // Handle the response by passing data to the parent component
      onResult(response.data);
    } catch (err) {
      console.error('File upload error:', err);
      setError('Error processing files. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Upload Files for Comparison</h2>
      <div>
        <label>
          File 1:
          <input type="file" onChange={(e) => setFile1(e.target.files[0])} />
        </label>
      </div>
      <div>
        <label>
          File 2:
          <input type="file" onChange={(e) => setFile2(e.target.files[0])} />
        </label>
      </div>
      <button onClick={handleFileUpload} disabled={loading}>
        {loading ? 'Processing...' : 'Compare'}
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
};

export default FileUpload;

