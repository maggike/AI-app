import React from 'react';

const Summary = ({ summaryData }) => {
  return (
    <div className="mt-4">
      <h2>Summary</h2>
      <div className="alert alert-info">
        <h4>Semantic Similarity</h4>
        <p>{summaryData.semantic_similarity}</p>
      </div>
      <div>
        <h4>Risks and Opportunities</h4>
        <ul>
          {summaryData.risks.map((risk, index) => (
            <li key={index}>{risk}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Summary;
