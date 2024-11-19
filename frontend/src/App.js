

import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import CompareResults from './components/CompareResults';
import Summary from './components/Summary';
import 'bootstrap/dist/css/bootstrap.min.css'; // Bootstrap CSS import

function App() {
  const [results, setResults] = useState(null);

  return (
    <div className="container">
      <h1 className="text-center my-4">AI-Powered M&A Analysis</h1>
      <FileUpload onResult={setResults} />
      {results && (
        <>
          <Summary summaryData={results} />
          <CompareResults results={results} />
        </>
      )}
    </div>
  );
}

export default App;

