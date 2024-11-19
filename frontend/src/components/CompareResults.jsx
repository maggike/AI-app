
import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register required components for Chart.js
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const CompareResults = ({ results }) => {
  // Handle missing or empty results gracefully
  if (!results) {
    return <p>No results to display. Please upload files and try again.</p>;
  }

  // Prepare data for the bar chart
  const chartData = {
    labels: ['Semantic Similarity', 'Added Sections', 'Removed Sections'],
    datasets: [
      {
        label: 'Analysis Metrics',
        data: [
          results.semantic_similarity || 0,
          results.added_sections.length || 0,
          results.removed_sections.length || 0,
        ],
        backgroundColor: [
          'rgba(75, 192, 192, 0.6)',
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 99, 132, 0.6)',
        ],
      },
    ],
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false, // Ensures it scales properly
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Comparison Metrics Chart',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Counts / Similarity',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Metrics',
        },
      },
    },
  };

  return (
    <div className="mt-4">
      <h2>Comparison Results</h2>

      {/* Bar Chart */}
      <div style={{ height: '400px' }} className="mb-4">
        <Bar data={chartData} options={chartOptions} />
      </div>

      {/* Detailed Tables */}
      <div>
        <h3>Added Sections</h3>
        {results.added_sections.length > 0 ? (
          <table className="table table-striped">
            <thead>
              <tr>
                <th>#</th>
                <th>Section</th>
              </tr>
            </thead>
            <tbody>
              {results.added_sections.map((section, index) => (
                <tr key={index}>
                  <td>{index + 1}</td>
                  <td>{section}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>No added sections.</p>
        )}

        <h3>Removed Sections</h3>
        {results.removed_sections.length > 0 ? (
          <table className="table table-striped">
            <thead>
              <tr>
                <th>#</th>
                <th>Section</th>
              </tr>
            </thead>
            <tbody>
              {results.removed_sections.map((section, index) => (
                <tr key={index}>
                  <td>{index + 1}</td>
                  <td>{section}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>No removed sections.</p>
        )}
      </div>
    </div>
  );
};

export default CompareResults;
