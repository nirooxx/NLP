"""
preprocess.py – NLP-Preprocessing und Feature-Vektorisierung für Tweets.

Dieses Skript enthält:
- Textbereinigung (Tokenization, Stemming, Stopword-Entfernung, Lemmatization)
- Feature-Engineering mit TF-IDF, GloVe, Emoji2Vec und BERT
- Lade- und Vektorisierungsfunktionen für verschiedene Embedding-Methoden

Wird als Modul genutzt oder direkt ausgeführt.
"""

import nltk
import logging
import re
import numpy as np
import emoji
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ressourcen-Setup
def setup_nltk_resources():
    """Überprüft, ob NLTK-Ressourcen vorhanden sind, und lädt sie falls notwendig."""
    resources = ["tokenizers/punkt", "corpora/stopwords", "corpora/wordnet"]
    resource_names = ["punkt", "stopwords", "wordnet"]

    for resource, name in zip(resources, resource_names):
        try:
            nltk.data.find(resource)
            logging.info(f"NLTK-Ressource bereits vorhanden: {name}")
        except LookupError:
            logging.info(f"Lade NLTK-Ressource: {name} ...")
            nltk.download(name, quiet=True)

setup_nltk_resources()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Textbereinigung & Tokenization
def clean_and_tokenize(tweet):
    """
    Bereinigt einen Tweet und tokenisiert ihn in Wörter.
    Entfernt URLs, Mentions und Sonderzeichen.
    """
    tweet = re.sub(r"http\S+", "", tweet)  # URLs entfernen
    tweet = re.sub(r"@\w+", "", tweet)  # Mentions entfernen
    tweet = re.sub(r"[^a-zA-Z0-9\s#]", "", tweet)  # Sonderzeichen entfernen
    return word_tokenize(tweet)

def remove_stopwords(tokens):
    """Entfernt Stopwörter aus einer Liste von Tokens."""
    return [word for word in tokens if word.lower() not in stop_words]

def lemmatize_tokens(tokens):
    """Wendet Lemmatization auf eine Liste von Tokens an."""
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_tweet(tweet):
    """Bereinigt einen Tweet vollständig: Tokenization, Stopword-Entfernung, Lemmatization."""
    tokens = clean_and_tokenize(tweet)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)

# GloVe Embeddings
def load_glove_embeddings(filepath):
    """
    Lädt GloVe-Embeddings aus einer Datei und gibt das Wörterbuch mit Embeddings zurück.

    Args:
        filepath (str): Pfad zur GloVe-Embedding-Datei.

    Returns:
        dict: Wörterbuch mit Wort-Embeddings.
        np.array: Durchschnittsvektor für unbekannte Wörter.
    """
    embeddings = {}
    vector_size = None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)

                if vector_size is None:
                    vector_size = len(vector)
                elif len(vector) != vector_size:
                    continue

                embeddings[word] = vector

        glove_mean = np.mean(list(embeddings.values()), axis=0)
        logging.info(f"{len(embeddings)} GloVe-Embeddings erfolgreich geladen.")
        return embeddings, glove_mean
    except FileNotFoundError:
        logging.error(f"GloVe-Embeddings nicht gefunden: {filepath}")
        return {}, np.zeros(200)

def vectorize_with_glove(tweets, glove_embeddings, glove_mean, vector_size=200):
    """
    Erstellt GloVe-Vektoren für eine Liste von Tweets.

    Args:
        tweets (list): Liste von Tweets.
        glove_embeddings (dict): Wörterbuch mit Embeddings.
        glove_mean (np.array): Durchschnittsvektor für unbekannte Wörter.
        vector_size (int): Größe der Embeddings.

    Returns:
        np.array: Vektorisierte Tweets.
    """
    tweet_vectors = []
    for tweet in tweets:
        tokens = tweet.split()
        vectors = [glove_embeddings.get(word, glove_mean) for word in tokens]
        tweet_vectors.append(np.mean(vectors, axis=0) if vectors else np.zeros(vector_size))
    
    return np.array(tweet_vectors)

# TF-IDF Vektorisierung
def vectorize_with_tfidf(tweets, max_features=20000, ngram_range=(1, 2)):
    """Erstellt TF-IDF-Vektoren für eine Liste von Tweets."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(tweets)
    return tfidf_matrix, vectorizer

# Emoji2Vec
def load_emoji2vec(filepath):
    """Lädt Emoji2Vec-Embeddings aus einer Datei."""
    emoji_vectors = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                emoji_char = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                emoji_vectors[emoji_char] = vector
        logging.info(f"{len(emoji_vectors)} Emoji-Vektoren geladen.")
        return emoji_vectors
    except FileNotFoundError:
        logging.error(f"Emoji2Vec-Embeddings nicht gefunden: {filepath}")
        return {}

def get_emoji_vector(tweet, emoji_vectors, vector_size=300):
    """Erstellt einen durchschnittlichen Vektor für Emojis in einem Tweet."""
    emojis_in_tweet = [char for char in tweet if char in emoji.EMOJI_DATA]
    vectors = [emoji_vectors[char] for char in emojis_in_tweet if char in emoji_vectors]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

def vectorize_with_emojis(tweets, emoji_vectors, vector_size=300):
    """Vektorisiert eine Liste von Tweets basierend auf Emojis."""
    return np.array([get_emoji_vector(tweet, emoji_vectors, vector_size) for tweet in tweets])

# BERT Embeddings
def load_bert_model(model_name="distilbert-base-uncased"):
    """Lädt das BERT-Modell und den Tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def vectorize_with_bert(tweets, tokenizer, model, max_length=128):
    """
    Erstellt BERT-Embeddings für eine Liste von Tweets.

    Args:
        tweets (list): Liste von Tweets.
        tokenizer: Tokenizer für das BERT-Modell.
        model: Vortrainiertes BERT-Modell.
        max_length (int): Maximale Länge der Eingabesequenzen.

    Returns:
        np.array: Vektorisierte Tweets.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tweet_vectors = []
    for tweet in tweets:
        tokens = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
        tweet_vectors.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

    return np.vstack(tweet_vectors)
