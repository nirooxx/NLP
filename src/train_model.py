"""
train_model.py – Modelltraining & Bewertung mit LightGBM.

Dieses Skript enthält:
- Training eines LightGBM-Modells mit Standard-Hyperparametern
- Modellbewertung mit Accuracy, Confusion Matrix & Classification Report
- Visualisierung der Confusion Matrix mit Seaborn

Das Skript kann als Modul genutzt oder direkt ausgeführt werden.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_lightgbm(X_train, y_train, n_estimators=500, max_depth=20, random_state=42):
    """
    Trainiert ein LightGBM-Modell mit Standardparametern oder übergebenen Hyperparametern.

    Args:
        X_train (array-like): Trainingsdaten.
        y_train (array-like): Labels für das Training.
        n_estimators (int): Anzahl der Bäume im Modell.
        max_depth (int): Maximale Tiefe der Entscheidungsbäume.
        random_state (int): Zufallsstate für Reproduzierbarkeit.

    Returns:
        LGBMClassifier: Trainiertes Modell.
    """
    logging.info("Starte Training des LightGBM-Modells...")
    
    model = LGBMClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    model.fit(X_train, y_train)
    logging.info("LightGBM-Training abgeschlossen.")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Bewertet das Modell anhand der Testdaten und gibt die Metriken zurück.

    Args:
        model (LGBMClassifier): Trainiertes Modell.
        X_test (array-like): Testdaten.
        y_test (array-like): Wahre Labels.

    Returns:
        tuple: (accuracy, confusion_matrix, classification_report)
    """
    logging.info("Starte Modellbewertung...")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)

    logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy, conf_matrix, report

def plot_confusion_matrix(conf_matrix):
    """
    Visualisiert die Confusion-Matrix als Heatmap.

    Args:
        conf_matrix (array-like): Confusion Matrix.
    """
    logging.info("Erstelle Confusion Matrix...")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive']
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
