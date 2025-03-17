import React, { useState } from "react";
import axios from "axios";
import Plot from "react-plotly.js";
import "./App.css";
// If using npm-installed Animate.css, uncomment the line below:
// import 'animate.css/animate.min.css';

function App() {
  const [formData, setFormData] = useState({
    temperatureMin: "",
    temperatureMax: "",
    conductivityMin: "",
    conductivityMax: "",
    year: "",
  });
  const [prediction, setPrediction] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    // Combine features into an array (your model expects 5 values)
    const featureArray = [
      parseFloat(formData.temperatureMin),
      parseFloat(formData.temperatureMax),
      parseFloat(formData.conductivityMin),
      parseFloat(formData.conductivityMax),
      parseFloat(formData.year),
    ];

    try {
      // Make the API call to your Flask backend
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        features: featureArray,
      });
      setPrediction(response.data);

      // Append new prediction to predictionHistory
      setPredictionHistory((prevHistory) => [
        ...prevHistory,
        {
          input: featureArray,
          predictedPh: response.data.predicted_pH,
          waterQuality: response.data.water_quality,
        },
      ]);
    } catch (error) {
      console.error("Error fetching predictions:", error);
      alert("Error fetching predictions. Check the console for details.");
    } finally {
      setLoading(false);
    }
  };

  // Dynamic chart data for "Predicted pH Over Time"
  const xValues = predictionHistory.map((_, index) => index + 1);
  const yValues = predictionHistory.map((item) => item.predictedPh);

  // Build hover text for each data point
  const hoverText = predictionHistory.map((item, index) => {
    const [tempMin, tempMax, condMin, condMax, year] = item.input;
    return (
      `Prediction #${index + 1}` +
      `<br>Inputs: [${tempMin}, ${tempMax}, ${condMin}, ${condMax}, ${year}]` +
      `<br>pH: ${item.predictedPh}` +
      `<br>WQ: ${item.waterQuality}`
    );
  });

  // Additional Chart: Water Quality Distribution (Bar Chart)
  const safeCount = predictionHistory.filter(
    (item) => item.waterQuality === "Safe"
  ).length;
  const unsafeCount = predictionHistory.filter(
    (item) => item.waterQuality === "Unsafe"
  ).length;

  return (
    <div className="app-container animate__animated animate__fadeIn">
      <h1>Groundwater Quality Prediction</h1>
      
      {/* Feature Ranges Section */}
      <div className="ranges">
        <h2>Feature Ranges</h2>
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Min</th>
              <th>Max</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Temperature Min</td>
              <td>-5.603681</td>
              <td>24.999860</td>
            </tr>
            <tr>
              <td>Temperature Max</td>
              <td>-6.270739</td>
              <td>39.999805</td>
            </tr>
            <tr>
              <td>Conductivity (µmhos/cm) Min</td>
              <td>-0.426889</td>
              <td>499.999230</td>
            </tr>
            <tr>
              <td>Conductivity (µmhos/cm) Max</td>
              <td>-0.471073</td>
              <td>799.356026</td>
            </tr>
          </tbody>
        </table>
      </div>

      <form onSubmit={handleSubmit} className="prediction-form">
        <input
          type="number"
          name="temperatureMin"
          placeholder="Min Temperature"
          onChange={handleChange}
          required
        />
        <input
          type="number"
          name="temperatureMax"
          placeholder="Max Temperature"
          onChange={handleChange}
          required
        />
        <input
          type="number"
          name="conductivityMin"
          placeholder="Min Conductivity"
          onChange={handleChange}
          required
        />
        <input
          type="number"
          name="conductivityMax"
          placeholder="Max Conductivity"
          onChange={handleChange}
          required
        />
        <input
          type="number"
          name="year"
          placeholder="Year"
          onChange={handleChange}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {prediction && (
        <div className="result animate__animated animate__slideInUp">
          <h2>Prediction Results</h2>
          <p>
            <strong>Predicted pH:</strong> {prediction.predicted_pH}
          </p>
          <p>
            <strong>Water Quality:</strong> {prediction.water_quality}
          </p>
          {/* Display accuracy if provided (e.g., 0.92 will show as 92%) */}
          {prediction.accuracy && (
            <p className="accuracy" style={{ fontSize: "0.9rem", color: "#555" }}>
              Model Accuracy: {Math.round(prediction.accuracy * 100)}%
            </p>
          )}
        </div>
      )}

      {/* Dynamic Chart 1: Predicted pH Over Time */}
      <div className="chart animate__animated animate__fadeInUp">
        <h2>Predicted pH Over Time</h2>
        <Plot
          data={[
            {
              x: xValues,
              y: yValues,
              type: "scatter",
              mode: "lines+markers",
              marker: { color: "blue" },
              text: hoverText,
              hoverinfo: "text+y",
            },
          ]}
          layout={{
            title: "",
            xaxis: { title: "Prediction Index" },
            yaxis: { title: "Predicted pH" },
            font: { family: "Arial, sans-serif", color: "#333" },
          }}
          style={{ width: "100%", height: "400px" }}
        />
      </div>

      {/* Dynamic Chart 2: Water Quality Distribution */}
      <div className="chart animate__animated animate__fadeInUp">
        <h2>Water Quality Distribution</h2>
        <Plot
          data={[
            {
              x: ["Safe", "Unsafe"],
              y: [safeCount, unsafeCount],
              type: "bar",
              marker: { color: ["green", "red"] },
            },
          ]}
          layout={{
            title: "",
            xaxis: { title: "Water Quality" },
            yaxis: { title: "Count" },
            font: { family: "Arial, sans-serif", color: "#333" },
          }}
          style={{ width: "100%", height: "400px" }}
        />
      </div>
    </div>
  );
}

export default App;
