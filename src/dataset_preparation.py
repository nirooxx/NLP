import nltk
from nltk.corpus import twitter_samples
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Laden des Datensatzes
def load_twitter_data():
    """
    Lädt die positiven und negativen Tweets aus dem NLTK Twitter-Datensatz.
    """
    nltk.download("twitter_samples", quiet=True)
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    logging.info(f"Beispiel positiver Tweet: {positive_tweets[0]}")
    logging.info(f"Beispiel negativer Tweet: {negative_tweets[0]}")
    return positive_tweets, negative_tweets

# Aufteilen in Trainings-, Validierungs- und Testdaten
def split_data(positive_tweets, negative_tweets, test_size=0.3, valid_size=0.5, balance=False):
    """
    Teilt die Daten in Trainings-, Validierungs- und Testdatensätze auf,
    stellt sicher, dass die Klassenverteilung gleichmäßig ist.
    """
    labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)
    tweets = positive_tweets + negative_tweets

    # Stratifizierte Aufteilung
    X_train, X_temp, y_train, y_temp = train_test_split(
        tweets, labels, test_size=test_size, stratify=labels, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=valid_size, stratify=y_temp, random_state=42
    )

    if balance:
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample([[tweet] for tweet in X_train], y_train)
        X_train = [item[0] for item in X_train]  # Liste von Listen zurück in normale Liste konvertieren

    logging.info(f"Training: {sum(y_train)} positive, {len(y_train) - sum(y_train)} negative")
    logging.info(f"Validation: {sum(y_valid)} positive, {len(y_valid) - sum(y_valid)} negative")
    logging.info(f"Test: {sum(y_test)} positive, {len(y_test) - sum(y_test)} negative")

    return X_train, X_valid, X_test, y_train, y_valid, y_test

if __name__ == "__main__":
    positive_tweets, negative_tweets = load_twitter_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(positive_tweets, negative_tweets, balance=True)

    logging.info(f"Train Size: {len(X_train)}, Valid Size: {len(X_valid)}, Test Size: {len(X_test)}")
