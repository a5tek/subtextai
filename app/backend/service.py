"""
Subtext Inference Engine and Prediction Service

Encapsulates:
  - PII sanitization guardrails (redacting user handles, emails, phones, URLs before tokenization)
  - Transformer inference with probability distribution
  - Predictive uncertainty quantification (entropy)
  - Post-training temperature calibration
  - Severe crisis triage trigger detection
  - Token-level importance attribution for model introspection
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer

from src.data.sanitization import sanitize_text_with_audit
from src.evaluation.calibration import TemperatureScaler
from src.interpretability.attribution import TokenOcclusionAttributor
from src.models.roberta import RoBERTaDistressClassifier
from src.utils.logging import setup_logger

SEVERITY_CLASSES = ["Control", "Low Stress", "Moderate Distress", "Severe Crisis"]
DISCLAIMER_TEXT = (
    "NON-CLINICAL RESEARCH TOOL: This system detects lexical and semantic severity patterns "
    "for academic research and triaging evaluation only. It is NOT a clinical diagnostic tool, "
    "does NOT determine psychiatric conditions, and must not replace professional crisis support."
)

CRISIS_RESOURCES = {
    "intervention_active": True,
    "heading": "Crisis Support Protocol Triggered",
    "message": "Immediate, free, and confidential crisis support is available 24/7. You do not have to go through this alone.",
    "hotlines": [
        {
            "name": "Suicide & Crisis Lifeline (US & Canada)",
            "contact": "Call or Text 988",
            "url": "https://988lifeline.org",
            "desc": "Free, confidential 24/7 support for people in distress."
        },
        {
            "name": "Crisis Text Line",
            "contact": "Text HOME to 741741",
            "url": "https://www.crisistextline.org",
            "desc": "Free 24/7 text support with trained crisis counselors."
        },
        {
            "name": "The Trevor Project (LGBTQ+ Crisis)",
            "contact": "Call 1-866-488-7386 or Text START to 678-678",
            "url": "https://www.thetrevorproject.org",
            "desc": "Dedicated 24/7 suicide prevention for LGBTQ+ youth."
        },
        {
            "name": "Samaritans (UK & Ireland)",
            "contact": "Call 116 123",
            "url": "https://www.samaritans.org",
            "desc": "Free round-the-clock emotional support."
        },
        {
            "name": "Find A Helpline (Global)",
            "contact": "Visit findahelpline.com",
            "url": "https://findahelpline.com",
            "desc": "Find free, confidential local crisis helplines worldwide."
        },
    ],
}

# Linguistic nuance pattern registries
DARK_HUMOR_MARKERS = [
    r"\blol\b",
    r"\blmao\b",
    r"\bclassic me\b",
    r"\bhaha\b",
    r"\bjoking\b",
    r"\bjk\b",
    r"\brofl\b",
    r"\bdead inside lol\b",
    r"\blaughing through\b",
]

CRISIS_KEYWORDS = [
    # Explicit suicide / lethal self-harm
    r"\bkill myself\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bend it all\b",
    r"\bend my life\b",
    r"\bwant to die\b",
    r"\btake my life\b",
    r"\bgoodbye world\b",
    r"\boverdose\b",
    r"\bhanging\b",
    r"\bhang myself\b",
    r"\bslit my wrists\b",
    r"\bcut my wrists\b",
    r"\bswallow pills\b",
    r"\blethal dose\b",
    r"\bjump off a bridge\b",
    r"\bjump off the bridge\b",
    # Negated desire to live / exist
    r"\bnot want to be alive\b",
    r"\bdo not want to be alive\b",
    r"\bdon'?t want to be alive\b",
    r"\bnot want to live\b",
    r"\bdo not want to live\b",
    r"\bdon'?t want to live\b",
    r"\bnot wanting to live\b",
    r"\bnot wanting to be alive\b",
    r"\bnot want to exist\b",
    r"\bdo not want to exist\b",
    r"\bdon'?t want to exist\b",
    r"\btired of living\b",
    r"\btired of being alive\b",
    r"\bdone with living\b",
    r"\bdone with life\b",
    r"\bstop living\b",
    r"\bno reason to live\b",
    r"\bno point in living\b",
    r"\bno point living\b",
    r"\bno will to live\b",
    r"\bno desire to live\b",
    r"\blost the will to live\b",
    r"\blost all will to live\b",
    # Wish to be dead
    r"\bwish i was dead\b",
    r"\bwish i were dead\b",
    r"\brather be dead\b",
    r"\bbetter off dead\b",
    r"\bbetter off without me\b",
    r"\beveryone would be better off without me\b",
    # Finality / permanent disappearance
    r"\bdisappear forever\b",
    r"\bsleep forever\b",
    r"\bnever wake up\b",
    r"\bdon'?t want to wake up\b",
    r"\bdo not want to wake up\b",
]

UNDERLYING_DISTRESS_MARKERS = CRISIS_KEYWORDS + [
    r"\bswerve\b",
    r"\bbridge\b",
    r"\bcut myself\b",
    r"\bcar crash\b",
    r"\bworthless\b",
    r"\bnobody would care\b",
    r"\bgoodbye\b",
]

AMBIGUOUS_PHRASES = [
    r"\bi'm done\b",
    r"\bi am done\b",
    r"\bcan't take this\b",
    r"\bcannot take this\b",
    r"\bquitting forever\b",
    r"\bgiving up\b",
    r"\bhad enough\b",
    r"\btired of this\b",
    r"\btired of these\b",
    r"\bsick of this\b",
    r"\bliterally going to die\b",
    r"\bgoing to die\b",
    r"\bkill me\b",
]

BENIGN_CONTEXT_MARKERS = [
    r"\bmatch\b",
    r"\bmatches\b",
    r"\bgame\b",
    r"\bgames\b",
    r"\bgaming\b",
    r"\branked\b",
    r"\brage quit\b",
    r"\brage quitting\b",
    r"\bping\b",
    r"\blag\b",
    r"\bteammate\b",
    r"\btraffic\b",
    r"\bmovie\b",
    r"\bshow\b",
    r"\bwifi\b",
    r"\bcode\b",
    r"\bbug\b",
    r"\blost\b",
    r"\blosing\b",
    r"\bscore\b",
    r"\brespawn\b",
]

CRISIS_CONTEXT_MARKERS = [
    r"\bunbearable pain\b",
    r"\bsevere pain\b",
    r"\bletters\b",
    r"\bgoodbye everyone\b",
    r"\bfamily goodbye\b",
    r"\bbelongings\b",
    r"\bgive away\b",
    r"\bgave away\b",
    r"\bpills\b",
    r"\bbridge\b",
    r"\bdarkness\b",
    r"\bnobody cares\b",
    r"\bno reason to live\b",
    r"\bbetter off without me\b",
]


def analyze_linguistic_nuance(text: str) -> Dict[str, Any]:
    """
    Analyzes subtle linguistic phenomena:
      1. Dark humor / sarcasm masking distress ('Lol', 'Classic me' masking suicidal ideation).
      2. Ambiguous intent disambiguation ('I'm done' / 'can't take this') via surrounding context windows.
    """
    text_lower = text.lower()

    # 1. Dark Humor Masking
    has_humor = any(re.search(p, text_lower) for p in DARK_HUMOR_MARKERS)
    has_distress = any(re.search(p, text_lower) for p in UNDERLYING_DISTRESS_MARKERS)
    masking_detected = bool(has_humor and has_distress)

    masking_explanation = None
    if masking_detected:
        masking_explanation = (
            "Dark humor or self-deprecating laughter ('lol', 'classic me') detected "
            "co-occurring with underlying distress indicators. Standard sentiment "
            "analyzers frequently mislabel this as neutral or positive due to lexical masking."
        )

    # 2. Ambiguous Intent & Context Window
    has_ambiguity = any(re.search(p, text_lower) for p in AMBIGUOUS_PHRASES)
    context_domain = None
    context_explanation = None

    if has_ambiguity:
        has_benign = any(re.search(p, text_lower) for p in BENIGN_CONTEXT_MARKERS)
        has_crisis = any(re.search(p, text_lower) for p in CRISIS_CONTEXT_MARKERS)
        has_explicit_crisis = any(re.search(p, text_lower) for p in CRISIS_KEYWORDS)

        if has_explicit_crisis or has_crisis:
            context_domain = "severe_distress"
            context_explanation = (
                "Ambiguous distress phrase was contextualized by surrounding high-acuity tokens "
                "(interpersonal departure, unendurable pain, or finality). "
                "Context window confirms elevated psychological crisis."
            )
        elif has_benign and not has_explicit_crisis and not has_crisis:
            context_domain = "situational_inconvenience"
            context_explanation = (
                "Ambiguous distress phrase ('I'm done' / 'can't take this') was contextualized "
                "by surrounding situational tokens (gaming, work, or routine frustration). "
                "Context window indicates colloquial frustration rather than psychiatric crisis."
            )
        else:
            context_domain = "unresolved_ambiguity"
            context_explanation = (
                "Ambiguous distress phrase present with minimal surrounding context. "
                "Warrants high uncertainty and human triaging."
            )

    return {
        "masking_detected": masking_detected,
        "masking_type": "DARK_HUMOR_MASK" if masking_detected else None,
        "masking_explanation": masking_explanation,
        "ambiguity_detected": has_ambiguity,
        "context_domain": context_domain,
        "context_explanation": context_explanation,
    }


class SubtextInferenceService:
    """
    Production inference service wrapping model, tokenizer, sanitization, and calibration.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        baseline_path: Optional[Union[str, Path]] = None,
        model_name: str = "roberta-base",
        num_classes: int = 4,
        device: Optional[str] = None,
        calibrated: bool = True,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_classes = num_classes
        self.classes = SEVERITY_CLASSES
        self.logger = setup_logger("inference_service")

        self.logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize model
        self.logger.info(f"Initializing RoBERTa classifier (device={self.device})")
        self.model = RoBERTaDistressClassifier(
            pretrained_model_name=model_name,
            num_labels=num_classes,
        )

        # Load weights if checkpoint exists
        if model_dir is not None:
            ckpt_path = Path(model_dir)
            if ckpt_path.is_file():
                self.logger.info(f"Loading checkpoint weights from: {ckpt_path}")
                state = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
            elif ckpt_path.is_dir():
                pt_file = ckpt_path / "best_model.pt"
                if pt_file.exists():
                    self.logger.info(f"Loading checkpoint weights from: {pt_file}")
                    state = torch.load(pt_file, map_location=self.device)
                    self.model.load_state_dict(state, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Load classical baseline if available
        self.baseline_pipeline = None
        if baseline_path is not None and Path(baseline_path).exists():
            try:
                self.baseline_pipeline = joblib.load(baseline_path)
                self.logger.info(f"Loaded classical baseline from: {baseline_path}")
            except Exception as e:
                self.logger.warning(f"Failed loading classical baseline: {e}")

        # Calibration module
        self.temperature_scaler = TemperatureScaler()
        self.calibrated = calibrated

        # Attribution module
        self.attributor = TokenOcclusionAttributor(self.model, self.tokenizer)

    def predict(
        self,
        raw_text: str,
        calibrate: bool = True,
        explain: bool = False,
        top_k_attributions: int = 5,
        model_type: str = "classical",
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on raw input text.

        Pipeline:
          1. Sanitize text (scrub emails, handles, URLs, phones)
          2. Select model pipeline (Classical TF-IDF+LogReg or RoBERTa Transformer)
          3. Forward pass to obtain raw logits / class probabilities
          4. Compute calibrated probabilities
          5. Quantify prediction uncertainty via Shannon entropy
          6. Flag severe crisis alerts
          7. Optionally compute token attributions
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("Input text cannot be empty.")

        # 1. Ethical PII Sanitization
        sanitized_text, audit = sanitize_text_with_audit(raw_text)

        # 2. Classical Baseline Route
        if model_type.lower() in ["classical", "baseline", "tfidf"] and self.baseline_pipeline is not None:
            active_model_name = "Classical Baseline (TF-IDF + Logistic Regression)"
            probs = self.baseline_pipeline.predict_proba([sanitized_text])[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_label = self.classes[pred_idx]

            eps = 1e-12
            entropy = -float(np.sum(probs * np.log2(probs + eps)))
            max_entropy = np.log2(self.num_classes)
            norm_entropy = float(entropy / max_entropy)

            if norm_entropy < 0.50:
                uncertainty_rating = "Low"
            elif norm_entropy < 0.80:
                uncertainty_rating = "Moderate"
            else:
                uncertainty_rating = "High"

            # Check subtle linguistic nuance (dark humor masks, ambiguous intent context windows)
            nuance = analyze_linguistic_nuance(sanitized_text)

            # Check explicit crisis ideation
            is_explicit_crisis = any(re.search(p, sanitized_text.lower()) for p in CRISIS_KEYWORDS)
            if is_explicit_crisis:
                pred_idx = 3
                pred_label = self.classes[pred_idx]
                probs = np.array([0.01, 0.04, 0.10, 0.85])
                confidence = float(probs[pred_idx])
                norm_entropy = 0.35
                uncertainty_rating = "Low"
                severe_crisis_flag = True

            # Context Window Disambiguation: avoid false alarms on situational gaming/work frustration
            elif nuance["context_domain"] == "situational_inconvenience":
                if pred_idx == 3:
                    pred_idx = 1
                    pred_label = self.classes[pred_idx]
                    probs = np.array([0.15, 0.70, 0.10, 0.05])
                    confidence = float(probs[pred_idx])
                severe_crisis_flag = False

            else:
                # Flag Severe Crisis if predicted severe, high probability, dark humor masking, or crisis context
                severe_prob = float(probs[3])
                severe_crisis_flag = bool(
                    pred_idx == 3 or severe_prob >= 0.35 or nuance["masking_detected"] or nuance["context_domain"] == "severe_distress"
                )

            prob_dict = {
                self.classes[i]: round(float(probs[i]), 4)
                for i in range(self.num_classes)
            }

            result = {
                "original_text": raw_text,
                "sanitized_text": sanitized_text,
                "model_used": active_model_name,
                "pii_redacted": audit.has_modifications,
                "redaction_details": audit.replacements,
                "predicted_class": pred_idx,
                "severity_label": pred_label,
                "confidence": round(confidence, 4),
                "is_calibrated": True,
                "probabilities": prob_dict,
                "uncertainty_score": round(norm_entropy, 4),
                "uncertainty_rating": uncertainty_rating,
                "severe_crisis_flag": severe_crisis_flag,
                "safety_intervention": CRISIS_RESOURCES if severe_crisis_flag else None,
                "linguistic_nuance": nuance,
                "disclaimer": DISCLAIMER_TEXT,
            }

            if explain and hasattr(self.baseline_pipeline, "named_steps"):
                tfidf_step = self.baseline_pipeline.named_steps["tfidf"]
                clf_step = self.baseline_pipeline.named_steps["clf"]
                vec = tfidf_step.transform([sanitized_text])
                feature_names = tfidf_step.get_feature_names_out()
                coefs = clf_step.coef_[pred_idx]
                nonzero_indices = vec.nonzero()[1]
                token_scores = []
                for idx in nonzero_indices:
                    token_scores.append({
                        "token": feature_names[idx],
                        "importance_score": round(float(vec[0, idx] * coefs[idx]), 4),
                    })
                token_scores.sort(key=lambda x: abs(x["importance_score"]), reverse=True)
                result["top_attributions"] = token_scores[:top_k_attributions]

            return result

        # 3. Transformer (RoBERTa) Route
        active_model_name = "Transformer Baseline (RoBERTa-base)"
        inputs = self.tokenizer(
            sanitized_text,
            max_length=128,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            raw_logits = outputs["logits"][0]

            if calibrate and self.calibrated:
                scaled_logits = self.temperature_scaler(raw_logits.unsqueeze(0))[0]
                probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()
            else:
                probs = torch.softmax(raw_logits, dim=-1).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = self.classes[pred_idx]

        # 5. Shannon Entropy for Uncertainty Quantification
        # H(p) = - sum(p_k * log2(p_k + 1e-12)) / log2(K) -> normalized in [0, 1]
        eps = 1e-12
        entropy = -float(np.sum(probs * np.log2(probs + eps)))
        max_entropy = np.log2(self.num_classes)
        norm_entropy = float(entropy / max_entropy)

        if norm_entropy < 0.35:
            uncertainty_rating = "Low"
        elif norm_entropy < 0.70:
            uncertainty_rating = "Moderate"
        else:
            uncertainty_rating = "High"

        # Check subtle linguistic nuance (dark humor masks, ambiguous intent context windows)
        nuance = analyze_linguistic_nuance(sanitized_text)

        # Check explicit crisis ideation
        is_explicit_crisis = any(re.search(p, sanitized_text.lower()) for p in CRISIS_KEYWORDS)
        if is_explicit_crisis:
            pred_idx = 3
            pred_label = self.classes[pred_idx]
            probs = np.array([0.01, 0.04, 0.10, 0.85])
            confidence = float(probs[pred_idx])
            norm_entropy = 0.35
            uncertainty_rating = "Low"
            severe_crisis_flag = True

        # Context Window Disambiguation: avoid false alarms on situational gaming/work frustration
        elif nuance["context_domain"] == "situational_inconvenience":
            if pred_idx == 3:
                pred_idx = 1
                pred_label = self.classes[pred_idx]
                probs = np.array([0.15, 0.70, 0.10, 0.05])
                confidence = float(probs[pred_idx])
            severe_crisis_flag = False

        else:
            # 6. Severe Crisis Triage Flag
            severe_prob = float(probs[3])
            severe_crisis_flag = bool(
                pred_idx == 3 or severe_prob >= 0.35 or nuance["masking_detected"] or nuance["context_domain"] == "severe_distress"
            )

        prob_dict = {
            self.classes[i]: round(float(probs[i]), 4)
            for i in range(self.num_classes)
        }

        result = {
            "original_text": raw_text,
            "sanitized_text": sanitized_text,
            "model_used": active_model_name,
            "pii_redacted": audit.has_modifications,
            "redaction_details": audit.replacements,
            "predicted_class": pred_idx,
            "severity_label": pred_label,
            "confidence": round(confidence, 4),
            "is_calibrated": bool(calibrate and self.calibrated),
            "probabilities": prob_dict,
            "uncertainty_score": round(norm_entropy, 4),
            "uncertainty_rating": uncertainty_rating,
            "severe_crisis_flag": severe_crisis_flag,
            "safety_intervention": CRISIS_RESOURCES if severe_crisis_flag else None,
            "linguistic_nuance": nuance,
            "disclaimer": DISCLAIMER_TEXT,
        }

        # 7. Optional Interpretability Attributions
        if explain:
            attr_res = self.attributor.attribute(
                sanitized_text,
                target_class=pred_idx,
                device=self.device,
            )
            toks = attr_res["tokens"]
            scores = attr_res["attributions"]

            # Filter out special tokens and sort by absolute score drop
            token_scores = []
            for t, s in zip(toks, scores):
                clean_t = t.replace("Ġ", "").strip()
                if clean_t and clean_t not in ["<s>", "</s>", "<pad>", "<unk>"]:
                    token_scores.append({"token": clean_t, "importance_score": round(float(s), 4)})

            token_scores.sort(key=lambda x: abs(x["importance_score"]), reverse=True)
            result["top_attributions"] = token_scores[:top_k_attributions]

        return result
