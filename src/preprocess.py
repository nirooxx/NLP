import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import emoji
from transformers import AutoTokenizer, AutoModel

# Ressourcen-Setup
def setup_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Tokenisierung und Bereinigung
def clean_and_tokenize(tweet):
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"[^a-zA-Z0-9\s#]", "", tweet)
    return word_tokenize(tweet)

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_tweet(tweet):
    tokens = clean_and_tokenize(tweet)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)

# GloVe Embeddings
def load_glove_embeddings(filepath):
    embeddings = {}
    vector_size = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            try:
                vector = np.array(parts[1:], dtype=np.float32)
                if vector_size is None:
                    vector_size = len(vector)  # Dimension festlegen
                elif len(vector) != vector_size:
                    continue  # Überspringe fehlerhafte Zeilen
                embeddings[word] = vector
            except ValueError:
                continue

    glove_mean = np.mean(list(embeddings.values()), axis=0)
    print(f"{len(embeddings)} GloVe-Embeddings erfolgreich geladen.")
    return embeddings, glove_mean

def vectorize_with_glove(tweets, glove_embeddings, glove_mean, vector_size=200):
    tweet_vectors = []
    for tweet in tweets:
        tokens = tweet.split()
        vectors = [glove_embeddings.get(word, glove_mean) for word in tokens]
        if vectors:
            tweet_vectors.append(np.mean(vectors, axis=0))
        else:
            tweet_vectors.append(np.zeros(vector_size))
    return np.array(tweet_vectors)

# TF-IDF Vektorisierung
def vectorize_with_tfidf(tweets, max_features=20000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(tweets)
    return tfidf_matrix, vectorizer

# Emoji2Vec
def load_emoji2vec(filepath):
    emoji_vectors = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            emoji_char = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            emoji_vectors[emoji_char] = vector
    print(f"{len(emoji_vectors)} Emoji-Vektoren geladen.")
    return emoji_vectors

def get_emoji_vector(tweet, emoji_vectors, vector_size=300):
    emojis_in_tweet = [char for char in tweet if char in emoji.EMOJI_DATA]
    if not emojis_in_tweet:
        return np.zeros(vector_size)
    vectors = [emoji_vectors[char] for char in emojis_in_tweet if char in emoji_vectors]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

def vectorize_with_emojis(tweets, emoji_vectors, vector_size=300):
    return np.array([get_emoji_vector(tweet, emoji_vectors, vector_size) for tweet in tweets])

# BERT Embeddings
def load_bert_model(model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def vectorize_with_bert(tweets, tokenizer, model, max_length=128):
    tweet_vectors = []
    for tweet in tweets:
        tokens = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        outputs = model(**tokens)
        tweet_vectors.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return np.vstack(tweet_vectors)
