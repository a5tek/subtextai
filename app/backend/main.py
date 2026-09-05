"""
Subtext FastAPI Backend Application (Milestone 10)

Exposes:
  - GET  /health           : Service health & model metadata
  - POST /api/v1/predict   : Severity prediction, uncertainty, and safety alerts
  - POST /api/v1/interpret : Detailed token attribution analysis
  - GET  /                 : Interactive Demonstration Dashboard
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.backend.service import SubtextInferenceService, DISCLAIMER_TEXT

app = FastAPI(
    title="Subtext AI - Context-Aware Distress Severity Detection API",
    description="Research prototype for 4-class distress severity classification, uncertainty quantification, and ethical sanitization.",
    version="1.0.0",
)

# Enable CORS for external frontend or local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton service instance
_service: Optional[SubtextInferenceService] = None


def get_service() -> SubtextInferenceService:
    global _service
    if _service is None:
        # Check available experiment checkpoints in order of priority
        ckpt_candidates = [
            PROJECT_ROOT / "experiments" / "loss_functions" / "E4_weighted_focal_loss" / "checkpoints",
            PROJECT_ROOT / "experiments" / "loss_functions" / "E3_focal_loss" / "checkpoints",
            PROJECT_ROOT / "experiments" / "loss_functions" / "E2_weighted_cross_entropy" / "checkpoints",
            PROJECT_ROOT / "experiments" / "loss_functions" / "E1_cross_entropy" / "checkpoints",
            PROJECT_ROOT / "experiments" / "roberta" / "checkpoints",
        ]
        chosen_ckpt = None
        for cand in ckpt_candidates:
            if (cand / "best_model.pt").exists():
                chosen_ckpt = cand
                break

        baseline_ckpt = PROJECT_ROOT / "experiments" / "baseline" / "baseline_model.joblib"

        _service = SubtextInferenceService(
            model_dir=chosen_ckpt,
            baseline_path=baseline_ckpt,
            model_name="roberta-base",
            num_classes=4,
        )
    return _service


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text sample to evaluate", json_schema_extra={"example": "this midterm is literally killing me"})
    calibrate: bool = Field(True, description="Whether to apply temperature scaling calibration")
    explain: bool = Field(False, description="Whether to include token feature attributions")
    model_type: str = Field("classical", description="Model engine to use ('classical' for 96.7% baseline or 'roberta')")


class PredictResponse(BaseModel):
    original_text: str
    sanitized_text: str
    model_used: Optional[str] = "Classical Baseline (TF-IDF + Logistic Regression)"
    pii_redacted: bool
    redaction_details: Dict[str, int]
    predicted_class: int
    severity_label: str
    confidence: float
    is_calibrated: bool
    probabilities: Dict[str, float]
    uncertainty_score: float
    uncertainty_rating: str
    severe_crisis_flag: bool
    safety_intervention: Optional[Dict[str, Any]] = None
    linguistic_nuance: Optional[Dict[str, Any]] = None
    disclaimer: str
    top_attributions: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    status: str
    model_backbone: str
    num_classes: int
    classes: List[str]
    disclaimer: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns API health, loaded architecture, and non-clinical disclaimer."""
    service = get_service()
    return HealthResponse(
        status="healthy",
        model_backbone=service.model.pretrained_model_name or "roberta-base",
        num_classes=service.num_classes,
        classes=service.classes,
        disclaimer=DISCLAIMER_TEXT,
    )


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict_severity(req: PredictRequest):
    """
    Analyzes input text, applies PII sanitization, and predicts severity level,
    calibrated probabilities, entropy-based uncertainty, and crisis triage alerts.
    """
    try:
        service = get_service()
        result = service.predict(
            raw_text=req.text,
            calibrate=req.calibrate,
            explain=req.explain,
            model_type=req.model_type,
        )
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")


@app.post("/api/v1/interpret")
def interpret_prediction(req: PredictRequest):
    """
    Computes token-level feature attributions to introspect model evidence.
    """
    try:
        service = get_service()
        result = service.predict(
            raw_text=req.text,
            calibrate=req.calibrate,
            explain=True,
            top_k_attributions=10,
            model_type=req.model_type,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpretation error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def demonstration_ui():
    """
    Serves interactive demonstration frontend dashboard.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Subtext AI — Context-Aware Online Distress Severity Detection</title>
      <style>
        :root {
          --bg: #0F172A;
          --card-bg: #1E293B;
          --border: #334155;
          --text: #F8FAFC;
          --text-muted: #94A3B8;
          --primary: #3B82F6;
          --primary-hover: #2563EB;
          --c-control: #10B981;
          --c-low: #F59E0B;
          --c-mod: #F97316;
          --c-sev: #EF4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: var(--bg);
          color: var(--text);
          padding: 2rem 1rem;
          line-height: 1.5;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .disclaimer-banner {
          background: rgba(239, 68, 68, 0.15);
          border: 1px solid var(--c-sev);
          color: #FCA5A5;
          padding: 0.85rem 1.25rem;
          border-radius: 8px;
          margin-bottom: 1.5rem;
          font-size: 0.85rem;
        }
        header { margin-bottom: 2rem; }
        h1 { font-size: 1.85rem; font-weight: 700; margin-bottom: 0.5rem; }
        .subtitle { color: var(--text-muted); font-size: 0.95rem; }
        .card {
          background: var(--card-bg);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 1.5rem;
          margin-bottom: 1.5rem;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        label { display: block; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem; }
        textarea {
          width: 100%;
          min-height: 100px;
          background: #0F172A;
          border: 1px solid var(--border);
          border-radius: 8px;
          color: var(--text);
          padding: 0.75rem;
          font-size: 0.95rem;
          font-family: inherit;
          resize: vertical;
        }
        textarea:focus { outline: none; border-color: var(--primary); }
        .preset-buttons { margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .preset-btn {
          background: #334155;
          color: #E2E8F0;
          border: none;
          padding: 0.35rem 0.75rem;
          border-radius: 6px;
          font-size: 0.8rem;
          cursor: pointer;
        }
        .preset-btn:hover { background: #475569; }
        .action-row { margin-top: 1rem; display: flex; justify-content: space-between; align-items: center; }
        .btn-analyze {
          background: var(--primary);
          color: white;
          border: none;
          padding: 0.65rem 1.5rem;
          border-radius: 8px;
          font-size: 0.95rem;
          font-weight: 600;
          cursor: pointer;
        }
        .btn-analyze:hover { background: var(--primary-hover); }
        .results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
        @media (max-width: 650px) { .results-grid { grid-template-columns: 1fr; } }
        .stat-box {
          background: #0F172A;
          padding: 1rem;
          border-radius: 8px;
          border: 1px solid var(--border);
        }
        .stat-label { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }
        .stat-value { font-size: 1.3rem; font-weight: 700; margin-top: 0.25rem; }
        .badge {
          display: inline-block;
          padding: 0.25rem 0.6rem;
          border-radius: 9999px;
          font-size: 0.8rem;
          font-weight: 700;
        }
        .prob-bar-row { margin-top: 0.6rem; }
        .prob-bar-label { display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem; }
        .bar-bg { background: #334155; height: 10px; border-radius: 5px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 5px; transition: width 0.4s ease; }
        .attribution-tag {
          display: inline-block;
          padding: 0.2rem 0.5rem;
          margin: 0.2rem;
          border-radius: 4px;
          font-size: 0.8rem;
          font-family: monospace;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="disclaimer-banner">
          <strong>Non-Clinical Research System:</strong> This interface demonstrates contextual severity modeling, uncertainty quantification, and ethical sanitization for NLP research. It does NOT offer medical or psychiatric diagnostic determinations.
        </div>
        <header>
          <h1>Subtext AI</h1>
          <p class="subtitle">Context-Aware Online Distress Severity Detection & Uncertainty Quantification</p>
        </header>

        <div class="card">
          <label for="inputText">Input Text (Social Media / Online Post)</label>
          <textarea id="inputText" placeholder="Enter text to analyze..."></textarea>
          <div class="preset-buttons">
            <span style="font-size: 0.8rem; color: var(--text-muted); align-self: center;">Try examples:</span>
            <button class="preset-btn" onclick="setPreset('midterm is literally killing me right now, so stressed out!')">Colloquial Hyperbole</button>
            <button class="preset-btn" onclick="setPreset('Lol wishing my car would swerve off the bridge on the way to work, classic me haha')">Dark Humor Mask</button>
            <button class="preset-btn" onclick="setPreset('I am so done and cannot take this anymore, lost 5 ranked matches in a row, rage quitting!')">Ambiguous (Gaming - Mitigated)</button>
            <button class="preset-btn" onclick="setPreset('I am done and cannot take this pain anymore, I gave away my dog and said goodbye to my family.')">Ambiguous (Crisis - Escalation)</button>
            <button class="preset-btn" onclick="setPreset('Working 14 hour shifts this week, exhausted and burnt out.')">Low Stress</button>
            <button class="preset-btn" onclick="setPreset('I feel empty, completely hopeless, and crying every night.')">Moderate Distress</button>
            <button class="preset-btn" onclick="setPreset('I have decided to end my life tonight, goodbye world.')">Severe Crisis (Hotlines)</button>
            <button class="preset-btn" onclick="setPreset('Contact user@example.com or visit https://hotline.org for help')">PII Scrubbing</button>
          </div>
          <div class="action-row">
            <div style="display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;">
              <div>
                <label style="display: inline; font-size: 0.85rem; font-weight: 600; margin-right: 0.35rem;">Model Engine:</label>
                <select id="modelSelector" style="background: #0F172A; color: #E2E8F0; border: 1px solid var(--border); border-radius: 6px; padding: 0.35rem 0.6rem; font-size: 0.85rem;">
                  <option value="classical" selected>Classical Baseline (TF-IDF + LogReg, 96.7% Test Acc)</option>
                  <option value="roberta">Deep Transformer (RoBERTa-base)</option>
                </select>
              </div>
              <div>
                <label style="display: inline; font-size: 0.85rem; font-weight: normal;">
                  <input type="checkbox" id="checkExplain" checked> Include Token Attributions
                </label>
              </div>
            </div>
            <button class="btn-analyze" id="btnAnalyze" onclick="analyzeText()">Analyze Severity</button>
          </div>
        </div>

        <div id="resultsCard" class="card" style="display: none;">
          <!-- Safety Protocol: Crisis Support Intervention Card -->
          <div id="crisisInterventionCard" style="display:none; background: rgba(239, 68, 68, 0.18); border: 2px solid var(--c-sev); border-radius: 10px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
              <span style="font-size: 1.3rem;">⚠️</span>
              <h3 style="color: #FCA5A5; font-size: 1.15rem; font-weight: 700; margin: 0;" id="crisisHeading">Crisis Support Protocol Triggered</h3>
            </div>
            <p style="color: #F8FAFC; font-size: 0.9rem; margin-bottom: 1rem;" id="crisisMessage">
              Immediate, free, and confidential crisis support is available 24/7. You do not have to go through this alone.
            </p>
            <div id="hotlineList" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem;"></div>
          </div>

          <!-- Linguistic Nuance & Context Window Advisory Card -->
          <div id="nuanceCard" style="display:none; background: rgba(59, 130, 246, 0.15); border: 1px solid var(--primary); border-radius: 8px; padding: 1rem; margin-bottom: 1.25rem;">
            <div style="font-weight: 700; color: #93C5FD; font-size: 0.95rem; margin-bottom: 0.35rem;" id="nuanceHeading">Linguistic Nuance & Context Advisory</div>
            <div style="font-size: 0.85rem; color: #E2E8F0; line-height: 1.45;" id="nuanceBody"></div>
          </div>

          <h3>Inference Results</h3>
          <div class="results-grid">
            <div class="stat-box">
              <div class="stat-label">Predicted Severity Tier</div>
              <div class="stat-value" id="resSeverity">-</div>
            </div>
            <div class="stat-box">
              <div class="stat-label">Confidence & Model</div>
              <div class="stat-value" id="resConfidence">-</div>
            </div>
            <div class="stat-box">
              <div class="stat-label">Uncertainty (Entropy)</div>
              <div class="stat-value" id="resUncertainty">-</div>
            </div>
            <div class="stat-box">
              <div class="stat-label">Sanitization Audit</div>
              <div class="stat-value" id="resSanitization" style="font-size: 1rem;">-</div>
            </div>
          </div>

          <div style="margin-top: 1.5rem;">
            <h4>Class Probability Distribution</h4>
            <div id="probBars"></div>
          </div>

          <div id="attributionsSection" style="margin-top: 1.5rem; display: none;">
            <h4>Token Feature Attributions (Evidence Introspection)</h4>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;">
              Tokens highlighted by feature importance (contribution towards predicted class):
            </p>
            <div id="attributionTags"></div>
          </div>
        </div>
      </div>

      <script>
        function setPreset(txt) {
          document.getElementById('inputText').value = txt;
        }

        async function analyzeText() {
          const text = document.getElementById('inputText').value.trim();
          if (!text) return;
          const explain = document.getElementById('checkExplain').checked;
          const model_type = document.getElementById('modelSelector').value;
          const btn = document.getElementById('btnAnalyze');
          btn.textContent = 'Evaluating...';
          btn.disabled = true;

          try {
            const resp = await fetch('/api/v1/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text, calibrate: true, explain, model_type })
            });
            const data = await resp.json();
            renderResults(data);
          } catch (err) {
            alert('Error calling inference service: ' + err.message);
          } finally {
            btn.textContent = 'Analyze Severity';
            btn.disabled = false;
          }
        }

        function renderResults(data) {
          document.getElementById('resultsCard').style.display = 'block';

          // 1. Safety Protocol: Crisis Support Card Wireup
          const crisisCard = document.getElementById('crisisInterventionCard');
          if (data.severe_crisis_flag && data.safety_intervention) {
            crisisCard.style.display = 'block';
            const intervention = data.safety_intervention;
            document.getElementById('crisisHeading').textContent = intervention.heading || 'Crisis Support Protocol Triggered';
            document.getElementById('crisisMessage').textContent = intervention.message || 'Immediate, free, and confidential crisis support is available 24/7.';
            const hotlineContainer = document.getElementById('hotlineList');
            hotlineContainer.innerHTML = '';
            (intervention.hotlines || []).forEach(hl => {
              hotlineContainer.innerHTML += `
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 0.85rem;">
                  <div style="font-weight: 700; color: #FCA5A5; font-size: 0.95rem;">${hl.name}</div>
                  <div style="font-weight: 600; color: #FFFFFF; font-size: 1.05rem; margin: 0.25rem 0;">${hl.contact}</div>
                  <div style="font-size: 0.8rem; color: #CBD5E1; margin-bottom: 0.5rem;">${hl.desc}</div>
                  <a href="${hl.url}" target="_blank" rel="noopener noreferrer" style="color: #60A5FA; font-size: 0.8rem; text-decoration: underline;">Visit Resource &rarr;</a>
                </div>
              `;
            });
          } else {
            crisisCard.style.display = 'none';
          }

          // 2. Linguistic Nuance & Context Window Advisory
          const nuanceCard = document.getElementById('nuanceCard');
          const nuance = data.linguistic_nuance;
          if (nuance && (nuance.masking_detected || nuance.ambiguity_detected)) {
            nuanceCard.style.display = 'block';
            let html = '';
            if (nuance.masking_detected) {
              html += `<div style="margin-bottom: 0.35rem;"><strong>🎭 Masked Distress Alert (Dark Humor):</strong> ${nuance.masking_explanation}</div>`;
            }
            if (nuance.ambiguity_detected) {
              html += `<div><strong>🔍 Context Window Disambiguation (${nuance.context_domain}):</strong> ${nuance.context_explanation}</div>`;
            }
            document.getElementById('nuanceBody').innerHTML = html;
          } else {
            nuanceCard.style.display = 'none';
          }

          const colors = {
            'Control': 'var(--c-control)',
            'Low Stress': 'var(--c-low)',
            'Moderate Distress': 'var(--c-mod)',
            'Severe Crisis': 'var(--c-sev)'
          };

          const sevColor = colors[data.severity_label] || 'white';
          document.getElementById('resSeverity').innerHTML =
            `<span class="badge" style="background: ${sevColor}; color: #0F172A;">${data.severity_label} (${data.predicted_class})</span>`;

          document.getElementById('resConfidence').textContent =
            `${(data.confidence * 100).toFixed(1)}% (${data.model_used || 'Calibrated'})`;

          document.getElementById('resUncertainty').textContent =
            `${data.uncertainty_rating} (H = ${data.uncertainty_score.toFixed(3)})`;

          document.getElementById('resSanitization').textContent =
            data.pii_redacted ? `Redactions: ${JSON.stringify(data.redaction_details)}` : 'Clean (No PII detected)';

          // Probability bars
          const probContainer = document.getElementById('probBars');
          probContainer.innerHTML = '';
          for (const [cls, p] of Object.entries(data.probabilities)) {
            const c = colors[cls] || 'var(--primary)';
            const pct = (p * 100).toFixed(1);
            probContainer.innerHTML += `
              <div class="prob-bar-row">
                <div class="prob-bar-label"><span>${cls}</span><span>${pct}%</span></div>
                <div class="bar-bg">
                  <div class="bar-fill" style="width: ${pct}%; background: ${c};"></div>
                </div>
              </div>
            `;
          }

          // Attributions
          const attrSection = document.getElementById('attributionsSection');
          const attrContainer = document.getElementById('attributionTags');
          if (data.top_attributions && data.top_attributions.length > 0) {
            attrSection.style.display = 'block';
            attrContainer.innerHTML = '';
            data.top_attributions.forEach(item => {
              const bg = item.importance_score > 0 ? 'rgba(239, 68, 68, 0.25)' : 'rgba(59, 130, 246, 0.25)';
              const border = item.importance_score > 0 ? 'var(--c-sev)' : 'var(--primary)';
              attrContainer.innerHTML += `
                <span class="attribution-tag" style="background: ${bg}; border: 1px solid ${border};">
                  ${item.token} (${item.importance_score > 0 ? '+' : ''}${item.importance_score.toFixed(3)})
                </span>
              `;
            });
          } else {
            attrSection.style.display = 'none';
          }
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
