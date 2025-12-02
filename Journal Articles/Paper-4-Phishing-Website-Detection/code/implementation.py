"""
Explainable Phishing Website Detection for Secure and Sustainable Cyber Infrastructure
Authors: Tanzila Kehkashan, Maha Abdelhaq, Ahmad Sami Al-Shamayleh, et al.
Journal: Scientific Reports (2025)
DOI: 10.1038/s41598-025-27984-w

This implementation provides phishing detection using machine learning models
integrated with SHAP for explainability, achieving 97% accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class PhishingDetector:
    """
    Explainable phishing detection system using ML models with SHAP integration.
    """

    def __init__(self, model_type: str = 'rf', use_shap: bool = True):
        """
        Initialize phishing detector.

        Args:
            model_type: Type of model ('rf', 'svm', 'dt', 'lr', 'knn')
            use_shap: Whether to use SHAP for feature explainability
        """
        self.model_type = model_type
        self.use_shap = use_shap
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.shap_explainer = None
        self.shap_values = None

        # Initialize model based on type
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the specified model type."""
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            )
        elif self.model_type == 'dt':
            self.model = DecisionTreeClassifier(
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        elif self.model_type == 'lr':
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'knn':
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                metric='euclidean',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def preprocess_data(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """
        Preprocess feature data.

        Args:
            X: Input features
            fit_scaler: Whether to fit the scaler (True for training data)

        Returns:
            Scaled feature array
        """
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on training data.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.feature_names = X_train.columns.tolist()

        # Preprocess data
        X_train_scaled = self.preprocess_data(X_train, fit_scaler=True)

        # Train model
        print(f"Training {self.model_type.upper()} model...")
        self.model.fit(X_train_scaled, y_train)

        # Initialize SHAP explainer if requested
        if self.use_shap:
            print("Initializing SHAP explainer...")
            if self.model_type in ['rf', 'dt']:
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                # For SVM, LR, KNN use KernelExplainer (slower but works)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(X_train_scaled, 100)
                )

        print("Training completed!")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test: Test features

        Returns:
            Predicted labels
        """
        X_test_scaled = self.preprocess_data(X_test)
        return self.model.predict(X_test_scaled)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X_test: Test features

        Returns:
            Prediction probabilities
        """
        X_test_scaled = self.preprocess_data(X_test)
        return self.model.predict_proba(X_test_scaled)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)

        results = {
            'accuracy': accuracy_score(y_test, y_pred) * 100,
            'precision': precision_score(y_test, y_pred, average='weighted') * 100,
            'recall': recall_score(y_test, y_pred, average='weighted') * 100,
            'f1_score': f1_score(y_test, y_pred, average='weighted') * 100,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return results

    def explain_predictions(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate SHAP explanations for predictions.

        Args:
            X_test: Test features

        Returns:
            SHAP values
        """
        if not self.use_shap or self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Set use_shap=True during initialization.")

        X_test_scaled = self.preprocess_data(X_test)

        print("Computing SHAP values...")
        if self.model_type in ['rf', 'dt']:
            self.shap_values = self.shap_explainer.shap_values(X_test_scaled)
        else:
            self.shap_values = self.shap_explainer.shap_values(X_test_scaled)

        return self.shap_values

    def plot_shap_summary(self, X_test: pd.DataFrame, max_display: int = 15) -> None:
        """
        Plot SHAP summary visualization.

        Args:
            X_test: Test features
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            self.explain_predictions(X_test)

        X_test_scaled = self.preprocess_data(X_test)

        plt.figure(figsize=(10, 8))

        # For binary classification, use positive class SHAP values
        shap_vals = self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values

        shap.summary_plot(
            shap_vals,
            X_test_scaled,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(f'SHAP Feature Importance - {self.model_type.upper()} Model')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot confusion matrix.

        Args:
            X_test: Test features
            y_test: Test labels
        """
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.title(f'Confusion Matrix - {self.model_type.upper()} Model')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get top N most important features.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance scores
        """
        if self.model_type == 'rf':
            importances = self.model.feature_importances_
        elif self.model_type == 'dt':
            importances = self.model.feature_importances_
        else:
            # For other models, use SHAP values
            if self.shap_values is None:
                raise ValueError("SHAP values not computed. Call explain_predictions() first.")

            shap_vals = self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values
            importances = np.abs(shap_vals).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df


def load_phishing_dataset(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load phishing dataset from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        Tuple of (features, labels)
    """
    df = pd.read_csv(file_path)

    # Separate features and target
    # Assuming last column is the target ('Result' or similar)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert labels: -1 (legitimate) to 0, 1 (phishing) stays 1
    y = y.apply(lambda x: 0 if x == -1 else 1)

    return X, y


def compare_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: pd.Series, y_test: pd.Series,
                  use_shap: bool = True) -> pd.DataFrame:
    """
    Compare performance of all models.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        use_shap: Whether to use SHAP

    Returns:
        DataFrame with comparison results
    """
    models = ['rf', 'dt', 'knn', 'svm', 'lr']
    results = []

    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Model")
        print('='*60)

        detector = PhishingDetector(model_type=model_type, use_shap=use_shap)
        detector.train(X_train, y_train)

        metrics = detector.evaluate(X_test, y_test)

        results.append({
            'Model': model_type.upper(),
            'Accuracy (%)': f"{metrics['accuracy']:.2f}",
            'Precision (%)': f"{metrics['precision']:.2f}",
            'Recall (%)': f"{metrics['recall']:.2f}",
            'F1-Score (%)': f"{metrics['f1_score']:.2f}"
        })

        # Plot confusion matrix
        detector.plot_confusion_matrix(X_test, y_test)

        # Generate SHAP explanations if enabled
        if use_shap:
            try:
                detector.plot_shap_summary(X_test)
            except Exception as e:
                print(f"Warning: Could not generate SHAP plot for {model_type}: {e}")

    return pd.DataFrame(results)


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("PHISHING DETECTION RESULTS")
    print("=" * 60)

    print(f"\n{'Accuracy:':<30} {results['accuracy']:.2f}%")
    print(f"{'Precision:':<30} {results['precision']:.2f}%")
    print(f"{'Recall:':<30} {results['recall']:.2f}%")
    print(f"{'F1-Score:':<30} {results['f1_score']:.2f}%")

    print(f"\n{'CONFUSION MATRIX':^60}")
    print("-" * 60)
    cm = results['confusion_matrix']
    print(f"True Negatives:  {cm[0][0]:>5}  |  False Positives: {cm[0][1]:>5}")
    print(f"False Negatives: {cm[1][0]:>5}  |  True Positives:  {cm[1][1]:>5}")

    print("\n" + "=" * 60)

    # Interpretation
    accuracy = results['accuracy']
    if accuracy >= 95:
        print("✅ EXCELLENT - High-accuracy phishing detection!")
    elif accuracy >= 90:
        print("✓ GOOD - Reliable phishing detection")
    else:
        print("⚠️  MODERATE - Consider model improvement")

    print("=" * 60 + "\n")


def main():
    """
    Example usage demonstration.
    """
    print("Explainable Phishing Website Detection System")
    print("=" * 60)

    # Example: Generate synthetic data for demonstration
    # In practice, load real data using load_phishing_dataset()

    from sklearn.datasets import make_classification

    print("\nGenerating example dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=3,
        random_state=42
    )

    # Convert to DataFrame with feature names
    feature_names = [
        'UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'PrefixSuffix',
        'SubDomains', 'HTTPS', 'DomainRegLen', 'Favicon', 'NonStdPort',
        'RequestURL', 'AnchorURL', 'LinksInScriptTags', 'ServerFormHandler',
        'InfoEmail', 'AbnormalURL', 'WebsiteForwarding', 'StatusBarCust',
        'DisableRightClick', 'UsingPopupWindow', 'IframeRedirection',
        'AgeofDomain', 'DNSRecording', 'WebsiteTraffic', 'PageRank',
        'GoogleIndex', 'LinksPointingToPage', 'StatsReport', 'HTTPSDomainURL',
        'RightClickDisabled'
    ]

    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train and evaluate best model (Random Forest with SHAP)
    print("\n" + "=" * 60)
    print("Training Random Forest Model with SHAP")
    print("=" * 60)

    detector = PhishingDetector(model_type='rf', use_shap=True)
    detector.train(X_train, y_train)

    # Evaluate
    results = detector.evaluate(X_test, y_test)
    print_results(results)

    # Plot confusion matrix
    detector.plot_confusion_matrix(X_test, y_test)

    # Generate SHAP explanations
    detector.explain_predictions(X_test)
    detector.plot_shap_summary(X_test, max_display=15)

    # Get top features
    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    top_features = detector.get_feature_importance(top_n=10)
    print(top_features.to_string(index=False))

    # Compare all models
    print("\n" + "=" * 60)
    print("COMPARING ALL MODELS")
    print("=" * 60)
    comparison_df = compare_models(X_train, X_test, y_train, y_test, use_shap=True)
    print("\n" + comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
