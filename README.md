# README – Sentiment-Analyse von Tweets

## Projektübersicht
Dieses Projekt führt eine Sentiment-Analyse von Tweets durch und klassifiziert diese als positiv oder negativ.

Es kombiniert verschiedene moderne NLP-Techniken wie:

- **TF-IDF**: Text-Vektorisierung auf Basis von Wort- und Bi-Gram-Häufigkeiten.
- **GloVe**: Vortrainierte Wort-Embeddings zur semantischen Repräsentation von Wörtern.
- **Emoji2Vec**: Spezialisierte Vektoren für Emojis.
- **BERT**: Kontextuelle Embeddings für tiefere semantische Analysen.
- **LightGBM**: Ein leistungsstarker Algorithmus zur Klassifikation der Daten.

## Projektstruktur
Das Projekt ist wie folgt organisiert:

```plaintext
parent-folder/
├── data/                       # Enthält Embeddings oder Daten
│   ├── glove.twitter.27B.200d.txt # GloVe-Embeddings (nicht enthalten)
│   ├── emoji2vec.txt              # Emoji2Vec-Embeddings (nicht enthalten)
├── notebooks/                     # Enthält die PDF für Aufgabe 4 und das Jupyter-Notebook
│   ├── nlp_assignment.ipynb       # Jupyter-Notebook für Aufgaben 1-3
│   ├── Assignment_Task4.pdf       # PDF-Dokumentation für Aufgabe 4
├── src/                        # Enthält alle Python-Skripte
│   ├── dataset_preparation.py     # Funktionen zum Laden und Aufteilen der Daten
│   ├── download_data.py           # Lädt notwendige NLTK-Daten
│   ├── main.py                    # Hauptskript zur Ausführung der Pipeline
│   ├── preprocess.py              # Funktionen für Textbereinigung und Feature-Engineering
│   ├── train_model.py             # Training, Bewertung und Visualisierung des Modells
├── requirements.txt               # Liste der Python-Bibliotheken
├── README.md                      # Projektbeschreibung und Anleitung
```
## Voraussetzungen
Für die Ausführung des Projekts werden folgende Voraussetzungen benötigt:

- **Python-Version**: 3.8 oder höher
- **Internetverbindung**: Zum Herunterladen von NLTK-Daten, falls sie nicht vorhanden sind
- **Speicherplatz**: Für große Embeddings wie GloVe (ca. 1 GB) und Emoji2Vec

## Schritt-für-Schritt-Anleitung

### 1. Installiere die Abhängigkeiten
Navigiere in das Projektverzeichnis:
```bash
cd parent-folder
```
Installiere die benötigten Python-Bibliotheken:

```bash
pip install -r requirements.txt
```

### Wichtige Python-Bibliotheken
- **nltk**: Für Textverarbeitung.
- **numpy**: Verarbeitung von numerischen Daten.
- **scikit-learn**: Datenaufteilung, Metriken.
- **transformers**: BERT-Modelle und Tokenizer.
- **lightgbm**: Training des Modells.

### 2. Stelle sicher, dass die notwendigen Daten verfügbar sind
Das Projekt benötigt die folgenden vortrainierten Embeddings, die **nicht in der Abgabe enthalten** sind:

- **GloVe**:  
  Lade die Datei `glove.twitter.27B.200d.txt` von der offiziellen Seite herunter:  
  [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)  
  Verschiebe die Datei in den Ordner `data/`.

- **Emoji2Vec**:  
  Lade die Datei `emoji2vec.txt` von GitHub herunter:  
  [Emoji2Vec](https://github.com/uclnlp/emoji2vec)  
  Verschiebe die Datei ebenfalls in den Ordner `data/`.

### 3. Führe das Projekt aus
Das gesamte Projekt kann durch Ausführen der `main.py` gestartet werden:
```bash
python src/main.py
```
Während der Ausführung:

- Lädt das Skript automatisch die notwendigen NLTK-Daten herunter (falls sie nicht vorhanden sind).
- Vektorisiert die Tweets mit TF-IDF, GloVe, Emoji2Vec und BERT.
- Trainiert ein LightGBM-Modell und bewertet dessen Leistung.

### 4. Ergebnisse
Nach der erfolgreichen Ausführung des Projekts sehen Sie:

1. **Testgenauigkeit** des Modells:  
   - Die Gesamtgenauigkeit, beispielsweise `96.40%`.

2. **Classification Report**:  
   - Ein detaillierter Bericht mit folgenden Metriken:  
     - **Precision**: Anteil korrekter positiver Vorhersagen.  
     - **Recall**: Anteil korrekter Vorhersagen unter allen tatsächlichen positiven Instanzen.  
     - **F1-Score**: Harmonisches Mittel aus Precision und Recall.

3. **Confusion Matrix**:  
   - Eine grafische Darstellung der True Positives, True Negatives, False Positives und False Negatives.  

Diese Ergebnisse bieten Einblicke in die Leistungsfähigkeit des Modells und helfen, potenzielle Verbesserungsbereiche zu identifizieren.

## Datenquellen

- **NLTK Twitter-Datensatz**:  
  - Positiv und negativ klassifizierte Tweets.  
  - Wird automatisch beim ersten Ausführen heruntergeladen.

- **GloVe Embeddings**:  
  - Zur semantischen Repräsentation von Wörtern.  
  - Download-Link: [GloVe](https://nlp.stanford.edu/projects/glove/)

- **Emoji2Vec Embeddings**:  
  - Zur Verarbeitung von Emojis.  
  - Download-Link: [Emoji2Vec](https://github.com/uclnlp/emoji2vec)

## Problemlösung

Falls Fehler auftreten:

- **Abhängigkeiten nicht installiert**:  
  Stelle sicher, dass Sie folgenden Befehl ausgeführt haben:  
```bash
  pip install -r requirements.txt
```

- **Fehlende Daten**:
Überprüfe, ob die Dateien `glove.twitter.27B.200d.txt` und `emoji2vec.txt` im Ordner `data/` vorhanden sind.

- **Speicherprobleme**:
Falls GloVe und BERT zu viel Speicher beanspruchen, können Sie kleinere Embeddings verwenden oder das Projekt mit nur TF-IDF und Emoji2Vec ausführen.

## Erweiterungsmöglichkeiten
- **Hyperparameter-Tuning**: Verbesserung der Genauigkeit durch Optimierung von LightGBM.
- **Datenaugmentation**: Erhöhen der Trainingsdaten durch Synonymersetzung oder Übersetzungen.
- **Andere Embeddings**: Testen von Alternativen wie FastText oder RoBERTa.

> **Hinweis**: Die Zusammenfassung für Aufgabe 4 befindet sich im Ordner `notebooks/` und ist als `NLP_Assignment_Nangiale_Shir_8151475_Aufgabe_4.pdf` verfügbar.

## GitHub-Link

Das Projekt ist auch auf GitHub verfügbar:  
[GitHub-Repository](https://github.com/nirooxx/NLP)







