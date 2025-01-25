from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier

def train_lightgbm(X_train, y_train):
    """
    Trainiert ein LightGBM-Modell ohne Hyperparameter-Tuning.
    """
    model = LGBMClassifier(random_state=42, class_weight="balanced", n_estimators=500, max_depth=20)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Bewertet das Modell basierend auf den Testdaten.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)
    return accuracy, conf_matrix, report

def plot_confusion_matrix(conf_matrix):
    """
    Visualisiert die Confusion-Matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive']
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
