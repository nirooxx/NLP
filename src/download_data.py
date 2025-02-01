"""
download_data.py – Lädt benötigte NLTK-Daten für das NLP-Projekt herunter.

Dieses Skript lädt die für die Sentiment-Analyse notwendigen 
NLTK-Ressourcen, darunter:
- `twitter_samples` – Enthält positive & negative Tweets.
- `punkt` – Notwendig für Tokenization.
- `stopwords` – Stoppwörter-Filter für Textverarbeitung.

Wird dieses Skript direkt ausgeführt, startet der Download automatisch.
"""

import nltk

def download_nltk_data():
    """Überprüft und lädt die benötigten NLTK-Daten, falls nicht vorhanden."""
    resources = [
        ("corpora/twitter_samples", "twitter_samples"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords")
    ]
    
    for resource, name in resources:
        if not nltk.data.find(resource, quiet=True):
            print(f"Lade NLTK-Ressource: {name} ...")
            nltk.download(name, quiet=True)
        else:
            print(f"NLTK-Ressource bereits vorhanden: {name}")

    print("Alle erforderlichen NLTK-Daten sind verfügbar.")

# Falls das Skript direkt ausgeführt wird
if __name__ == "__main__":
    download_nltk_data()
