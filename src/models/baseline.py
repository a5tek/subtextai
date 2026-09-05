"""
Classical NLP Baseline Module

Chains Scikit-Learn TfidfVectorizer with Logistic Regression or Linear SVM.
Maintains strict data hygiene: vectorizer vocabulary and IDF weights are fitted
exclusively on the training split.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


class ClassicalBaseline:
    """
    Classical NLP baseline model combining TF-IDF representation learning
    with a regularized linear classifier.
    """

    def __init__(
        self,
        classifier_type: str = "logistic_regression",
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 2,
        C: float = 1.0,
        class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
        random_state: int = 42,
    ):
        """
        Initializes the classical baseline pipeline.
        
        Args:
            classifier_type: 'logistic_regression' or 'linear_svm'.
            max_features: Maximum vocabulary size for TF-IDF.
            ngram_range: Range of n-gram boundaries (min_n, max_n).
            sublinear_tf: If True, applies sublinear term frequency scaling 1 + log(tf).
            min_df: Minimum document frequency threshold.
            C: Inverse regularization strength.
            class_weight: 'balanced' or custom dict to counteract class imbalance.
            random_state: Random seed for reproducibility.
        """
        self.classifier_type = classifier_type.lower()
        self.max_features = max_features
        self.ngram_range = tuple(ngram_range)
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            strip_accents="unicode",
        )

        if self.classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=1000,
                solver="lbfgs",
                random_state=self.random_state,
            )
        elif self.classifier_type == "linear_svm":
            base_svm = LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                random_state=self.random_state,
                dual="auto",
            )
            # Calibrate SVM using isotonic/sigmoid to produce proper probabilities
            self.classifier = CalibratedClassifierCV(estimator=base_svm, cv=3)
        else:
            raise ValueError(f"Unsupported classifier type: '{classifier_type}'. Use 'logistic_regression' or 'linear_svm'.")

        self.pipeline = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.classifier),
        ])
        self.is_fitted = False

    def fit(self, texts: Union[List[str], np.ndarray], labels: Union[List[int], np.ndarray]) -> "ClassicalBaseline":
        """
        Fits the TF-IDF vectorizer and classifier strictly on the provided training texts.
        
        Args:
            texts: List or array of sanitized strings.
            labels: List or array of discrete integer severity labels.
            
        Returns:
            Fitted instance of self.
        """
        texts_clean = [str(t) for t in texts]
        labels_arr = np.asarray(labels, dtype=int)

        self.pipeline.fit(texts_clean, labels_arr)
        self.is_fitted = True
        return self

    def predict(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Predicts discrete severity labels for new texts.
        
        Args:
            texts: List or array of input texts.
            
        Returns:
            1D NumPy array of predicted labels {0, 1, 2, 3}.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() before .predict().")
        texts_clean = [str(t) for t in texts]
        return self.pipeline.predict(texts_clean)

    def predict_proba(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Predicts class posterior probabilities for new texts.
        
        Args:
            texts: List or array of input texts.
            
        Returns:
            2D NumPy array of shape (N, num_classes) with normalized probabilities summing to 1.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() before .predict_proba().")
        texts_clean = [str(t) for t in texts]
        return self.pipeline.predict_proba(texts_clean)

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serializes the fitted pipeline to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ClassicalBaseline":
        """Loads a serialized model checkpoint from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")
        return joblib.load(path)
