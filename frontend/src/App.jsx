import React, { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8001/api/predict";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const analyzeMammogram = async () => {
    if (!file) {
      setError("Please upload a mammogram image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to connect to the AI backend. Make sure the QuantamDX FastAPI server is running on port 8001."
      );
    } finally {
      setLoading(false);
    }
  };

  const clearAnalysis = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
  };

  const getPercent = (value) =>
    value != null ? (value * 100).toFixed(2) : "0.00";

  const getPredictionClass = (prediction) =>
    prediction?.toLowerCase() === "malignant"
      ? "malignant-result"
      : "benign-result";

  const getPredictionSymbol = (prediction) =>
    prediction?.toLowerCase() === "malignant" ? "!" : "✓";

  return (
    <div className="app">

      {/* ================= HEADER ================= */}

      <header className="topbar">

        <div className="brand">

          <div className="brand-icon">
            Q
          </div>

          <div>
            <h1>QuantumMed-Q</h1>

            <p>
              AI-Powered Mammography Research Workstation
            </p>
          </div>

        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          AI Backend Online
        </div>

      </header>


      {/* ================= MAIN ================= */}

      <main className="main-container">

        {/* ================= HERO ================= */}

        <section className="hero">

          <div>

            <div className="eyebrow">
              HYBRID QUANTUM-CLASSICAL AI
            </div>

            <h2>
              Mammography Analysis
              <br />
              <span>with Multi-Model AI Intelligence</span>
            </h2>

            <p>
              Upload a mammogram image for research-oriented analysis
              using EfficientNet-B0, ResNet-50, and an EfficientNet-B3
              feature extractor integrated with an 8-qubit variational
              quantum circuit.
            </p>

          </div>

          <div className="hero-quantum">

            <div className="orbit orbit-one"></div>
            <div className="orbit orbit-two"></div>

            <div className="quantum-core">
              ⚛
            </div>

          </div>

        </section>


        {/* ================= WORKSPACE ================= */}

        <section className="workspace">

          {/* ================= UPLOAD CARD ================= */}

          <div className="panel upload-panel">

            <div className="panel-heading">

              <div>
                <h3>Upload Mammogram</h3>

                <p>
                  Select a mammography image for analysis.
                </p>
              </div>

              <span className="step-number">
                01
              </span>

            </div>

            <input
              id="mammogram-input"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              hidden
            />

            {!preview ? (

              <label
                htmlFor="mammogram-input"
                className="drop-zone"
              >

                <div className="upload-icon">
                  ↑
                </div>

                <strong>
                  Choose mammogram image
                </strong>

                <span>
                  PNG, JPG or JPEG
                </span>

              </label>

            ) : (

              <div className="preview-wrapper">

                <img
                  src={preview}
                  alt="Uploaded mammogram"
                  className="mammogram-preview"
                />

                <div className="preview-overlay">
                  Image loaded
                </div>

              </div>

            )}

            {file && (

              <div className="file-info">

                <div>

                  <span className="file-label">
                    Selected image
                  </span>

                  <strong>
                    {file.name}
                  </strong>

                </div>

                <button
                  className="change-button"
                  onClick={() =>
                    document
                      .getElementById("mammogram-input")
                      .click()
                  }
                >
                  Change
                </button>

              </div>

            )}

            <button
              className="analyze-button"
              onClick={analyzeMammogram}
              disabled={!file || loading}
            >

              {loading ? (
                <>
                  <span className="spinner"></span>
                  Running 3 AI Models...
                </>
              ) : (
                <>
                  Analyze Mammogram
                  <span>→</span>
                </>
              )}

            </button>

            {file && !loading && (
              <button
                className="clear-button"
                onClick={clearAnalysis}
              >
                Clear analysis
              </button>
            )}

            {error && (

              <div className="error-box">

                <strong>
                  Analysis Error
                </strong>

                <span>
                  {error}
                </span>

              </div>

            )}

          </div>


          {/* ================= MODEL ARCHITECTURE ================= */}

          <div className="panel architecture-panel">

            <div className="panel-heading">

              <div>
                <h3>Model Architecture</h3>

                <p>
                  Multi-model research inference pipeline
                </p>
              </div>

              <span className="step-number">
                02
              </span>

            </div>

            <div className="pipeline">

              <div className="pipeline-node">

                <div className="node-icon">
                  B0
                </div>

                <div>
                  <strong>
                    EfficientNet-B0
                  </strong>

                  <span>
                    Classical CNN classifier
                  </span>
                </div>

              </div>

              <div className="pipeline-arrow">
                +
              </div>

              <div className="pipeline-node">

                <div className="node-icon">
                  R50
                </div>

                <div>
                  <strong>
                    ResNet-50
                  </strong>

                  <span>
                    Classical CNN classifier
                  </span>
                </div>

              </div>

              <div className="pipeline-arrow">
                +
              </div>

              <div className="pipeline-node quantum-node">

                <div className="node-icon quantum">
                  B3
                </div>

                <div>
                  <strong>
                    EfficientNet-B3
                  </strong>

                  <span>
                    1536D feature extractor
                  </span>
                </div>

              </div>

              <div className="pipeline-arrow">
                ↓
              </div>

              <div className="pipeline-node">

                <div className="node-icon">
                  8D
                </div>

                <div>
                  <strong>
                    Supervised Projection
                  </strong>

                  <span>
                    1536 → 128 → 32 → 8
                  </span>
                </div>

              </div>

              <div className="pipeline-arrow">
                ↓
              </div>

              <div className="pipeline-node quantum-node">

                <div className="node-icon quantum">
                  ⚛
                </div>

                <div>
                  <strong>
                    8-Qubit VQC
                  </strong>

                  <span>
                    3 variational layers
                  </span>
                </div>

              </div>

              <div className="pipeline-arrow">
                ↓
              </div>

              <div className="pipeline-node">

                <div className="node-icon">
                  AI
                </div>

                <div>
                  <strong>
                    Hybrid Fusion
                  </strong>

                  <span>
                    Benign / Malignant
                  </span>
                </div>

              </div>

            </div>

          </div>

        </section>


        {/* ================= RESULTS ================= */}

        {result && (

          <section className="results-section">

            <div className="results-heading">

              <div>

                <div className="eyebrow">
                  ANALYSIS COMPLETE
                </div>

                <h2>
                  Prediction Results
                </h2>

              </div>

              <div className="model-tag">
                3 MODEL INFERENCE
              </div>

            </div>


            {/* ================= MODEL COMPARISON ================= */}

            <div className="comparison-section">

              <div className="quantum-section-header">

                <div>
                  <span className="result-label">
                    MODEL COMPARISON
                  </span>

                  <h3>
                    Independent Model Predictions
                  </h3>
                </div>

                <div className="qubit-badge">
                  3 MODELS
                </div>

              </div>


              <div className="model-comparison-grid">

                {/* EfficientNet-B0 */}

                <div className="model-result-card">

                  <div className="model-card-header">

                    <div className="model-icon">
                      B0
                    </div>

                    <div>
                      <strong>
                        EfficientNet-B0
                      </strong>

                      <span>
                        Classical CNN
                      </span>
                    </div>

                  </div>

                  <div
                    className={`model-prediction ${getPredictionClass(
                      result.models.efficientnet_b0.prediction
                    )}`}
                  >

                    <span>
                      {getPredictionSymbol(
                        result.models.efficientnet_b0.prediction
                      )}
                    </span>

                    <strong>
                      {result.models.efficientnet_b0.prediction}
                    </strong>

                  </div>

                  <div className="mini-probabilities">

                    <div>
                      <span>Benign</span>
                      <strong>
                        {getPercent(
                          result.models.efficientnet_b0.probabilities.benign
                        )}%
                      </strong>
                    </div>

                    <div>
                      <span>Malignant</span>
                      <strong>
                        {getPercent(
                          result.models.efficientnet_b0.probabilities.malignant
                        )}%
                      </strong>
                    </div>

                  </div>

                </div>


                {/* ResNet-50 */}

                <div className="model-result-card">

                  <div className="model-card-header">

                    <div className="model-icon">
                      R50
                    </div>

                    <div>
                      <strong>
                        ResNet-50
                      </strong>

                      <span>
                        Classical CNN
                      </span>
                    </div>

                  </div>

                  <div
                    className={`model-prediction ${getPredictionClass(
                      result.models.resnet50.prediction
                    )}`}
                  >

                    <span>
                      {getPredictionSymbol(
                        result.models.resnet50.prediction
                      )}
                    </span>

                    <strong>
                      {result.models.resnet50.prediction}
                    </strong>

                  </div>

                  <div className="mini-probabilities">

                    <div>
                      <span>Benign</span>
                      <strong>
                        {getPercent(
                          result.models.resnet50.probabilities.benign
                        )}%
                      </strong>
                    </div>

                    <div>
                      <span>Malignant</span>
                      <strong>
                        {getPercent(
                          result.models.resnet50.probabilities.malignant
                        )}%
                      </strong>
                    </div>

                  </div>

                </div>


                {/* Hybrid VQC */}

                <div className="model-result-card hybrid-card">

                  <div className="model-card-header">

                    <div className="model-icon quantum">
                      ⚛
                    </div>

                    <div>
                      <strong>
                        EfficientNet-B3 + 8-Qubit VQC
                      </strong>

                      <span>
                        Hybrid Quantum-Classical Model
                      </span>
                    </div>

                  </div>

                  <div
                    className={`model-prediction ${getPredictionClass(
                      result.models.hybrid_vqc.prediction
                    )}`}
                  >

                    <span>
                      {getPredictionSymbol(
                        result.models.hybrid_vqc.prediction
                      )}
                    </span>

                    <strong>
                      {result.models.hybrid_vqc.prediction}
                    </strong>

                  </div>

                  <div className="mini-probabilities">

                    <div>
                      <span>Benign</span>
                      <strong>
                        {getPercent(
                          result.models.hybrid_vqc.probabilities.benign
                        )}%
                      </strong>
                    </div>

                    <div>
                      <span>Malignant</span>
                      <strong>
                        {getPercent(
                          result.models.hybrid_vqc.probabilities.malignant
                        )}%
                      </strong>
                    </div>

                  </div>

                </div>

              </div>


              {/* Model agreement */}

              <div className="agreement-box">

                <div className="agreement-icon">
                  i
                </div>

                <div>

                  <strong>
                    Model Prediction Summary
                  </strong>

                  <p>
                    EfficientNet-B0 predicted{" "}
                    <b>
                      {result.models.efficientnet_b0.prediction}
                    </b>
                    , while ResNet-50 and the Hybrid B3 + VQC
                    model predicted{" "}
                    <b>
                      {result.models.resnet50.prediction}
                    </b>
                    .
                  </p>

                </div>

              </div>

            </div>


            {/* ================= PRIMARY HYBRID RESULT ================= */}

            <div className="hybrid-result-section">

              <div className="results-heading">

                <div>

                  <div className="eyebrow">
                    HYBRID MODEL
                  </div>

                  <h2>
                    EfficientNet-B3 + 8-Qubit VQC
                  </h2>

                </div>

                <div className="model-tag">
                  PRIMARY RESEARCH MODEL
                </div>

              </div>


              <div className="result-grid">

                {/* Prediction */}

                <div className="prediction-card">

                  <span className="result-label">
                    MODEL PREDICTION
                  </span>

                  <div
                    className={`prediction-result ${getPredictionClass(
                      result.hybrid_result.prediction
                    )}`}
                  >

                    <div className="result-symbol">
                      {getPredictionSymbol(
                        result.hybrid_result.prediction
                      )}
                    </div>

                    <div>

                      <span>
                        Classification
                      </span>

                      <strong>
                        {result.hybrid_result.prediction}
                      </strong>

                    </div>

                  </div>

                  <div className="risk-display">

                    <span>
                      Risk Level
                    </span>

                    <strong className="risk-low">
                      {result.hybrid_result.risk_level}
                    </strong>

                  </div>

                </div>


                {/* Probabilities */}

                <div className="probability-card">

                  <span className="result-label">
                    HYBRID CLASS PROBABILITIES
                  </span>

                  <div className="probability-item">

                    <div className="probability-header">

                      <span>
                        Benign
                      </span>

                      <strong>
                        {getPercent(
                          result.hybrid_result.probabilities.benign
                        )}%
                      </strong>

                    </div>

                    <div className="probability-bar">

                      <div
                        className="benign-bar"
                        style={{
                          width: `${getPercent(
                            result.hybrid_result.probabilities.benign
                          )}%`,
                        }}
                      ></div>

                    </div>

                  </div>


                  <div className="probability-item">

                    <div className="probability-header">

                      <span>
                        Malignant
                      </span>

                      <strong>
                        {getPercent(
                          result.hybrid_result.probabilities.malignant
                        )}%
                      </strong>

                    </div>

                    <div className="probability-bar">

                      <div
                        className="malignant-bar"
                        style={{
                          width: `${getPercent(
                            result.hybrid_result.probabilities.malignant
                          )}%`,
                        }}
                      ></div>

                    </div>

                  </div>

                </div>

              </div>

            </div>


            {/* ================= QUANTUM FEATURES ================= */}

            <div className="quantum-section">

              <div className="quantum-section-header">

                <div>

                  <span className="result-label">
                    QUANTUM FEATURE SPACE
                  </span>

                  <h3>
                    8-Qubit Feature Representation
                  </h3>

                </div>

                <div className="qubit-badge">
                  8 QUBITS
                </div>

              </div>


              <div className="feature-grid">

                {result.quantum_features?.map(
                  (value, index) => (

                    <div
                      className="feature-box"
                      key={index}
                    >

                      <span>
                        Q{index + 1}
                      </span>

                      <strong>
                        {Number(value).toFixed(4)}
                      </strong>

                    </div>

                  )
                )}

              </div>

            </div>


            {/* ================= MODEL DETAILS ================= */}

            <div className="details-grid">

              <div className="detail-card">

                <span>
                  Feature Extractor
                </span>

                <strong>
                  EfficientNet-B3
                </strong>

              </div>


              <div className="detail-card">

                <span>
                  CNN Feature Dimension
                </span>

                <strong>
                  1536
                </strong>

              </div>


              <div className="detail-card">

                <span>
                  Quantum Qubits
                </span>

                <strong>
                  8
                </strong>

              </div>


              <div className="detail-card">

                <span>
                  VQC Layers
                </span>

                <strong>
                  3
                </strong>

              </div>

            </div>


            {/* ================= DISCLAIMER ================= */}

            <div className="disclaimer">

              <div className="disclaimer-icon">
                i
              </div>

              <div>

                <strong>
                  Research Prototype
                </strong>

                <p>
                  {result.disclaimer}
                </p>

              </div>

            </div>

          </section>

        )}

      </main>


      {/* ================= FOOTER ================= */}

      <footer>

        <div>

          <strong>
            QuantumMed-Q
          </strong>

          <span>
            Hybrid Quantum-Classical Mammography Research Platform
          </span>

        </div>

        <span>
          SIH26139 • Research Prototype
        </span>

      </footer>

    </div>
  );
}

export default App;