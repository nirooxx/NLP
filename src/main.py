from dataset_preparation import load_twitter_data, split_data
from preprocess import (
    preprocess_tweet,
    vectorize_with_tfidf,
    load_glove_embeddings,
    vectorize_with_glove,
    load_emoji2vec,
    vectorize_with_emojis,
    load_bert_model,
    vectorize_with_bert,
)
from train_model import train_lightgbm, evaluate_model, plot_confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # 1. Daten laden
    positive_tweets, negative_tweets = load_twitter_data()

    # 2. Daten aufteilen
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(positive_tweets, negative_tweets)

    # 3. Preprocessing
    print("Starte Preprocessing...")
    X_train_preprocessed = [preprocess_tweet(tweet) for tweet in X_train]
    X_valid_preprocessed = [preprocess_tweet(tweet) for tweet in X_valid]
    X_test_preprocessed = [preprocess_tweet(tweet) for tweet in X_test]
    print("Preprocessing abgeschlossen.")

    # 4. GloVe laden
    glove_path = "./data/glove.twitter.27B.200d.txt"  # Passe den Pfad entsprechend an
    print("Lade GloVe-Embeddings...")
    glove_embeddings, glove_mean = load_glove_embeddings(glove_path)
    print("GloVe-Embeddings erfolgreich geladen.")

    # 5. Emoji2Vec laden
    emoji2vec_path = "./data/emoji2vec.txt"  # Passe den Pfad entsprechend an
    print("Lade Emoji2Vec-Embeddings...")
    emoji_vectors = load_emoji2vec(emoji2vec_path)
    print("Emoji2Vec-Embeddings erfolgreich geladen.")

    # 6. BERT-Modell laden
    print("Lade BERT-Modell...")
    tokenizer, bert_model = load_bert_model("distilbert-base-uncased")
    print("BERT-Modell erfolgreich geladen.")

    # 7. Vektorisieren
    print("Vektorisieren mit TF-IDF...")
    tfidf_matrix_train, tfidf_vectorizer = vectorize_with_tfidf(X_train_preprocessed)
    tfidf_matrix_valid = tfidf_vectorizer.transform(X_valid_preprocessed)
    tfidf_matrix_test = tfidf_vectorizer.transform(X_test_preprocessed)

    print("Vektorisieren mit GloVe...")
    glove_matrix_train = vectorize_with_glove(X_train_preprocessed, glove_embeddings, glove_mean)
    glove_matrix_valid = vectorize_with_glove(X_valid_preprocessed, glove_embeddings, glove_mean)
    glove_matrix_test = vectorize_with_glove(X_test_preprocessed, glove_embeddings, glove_mean)

    print("Vektorisieren mit Emoji2Vec...")
    emoji_matrix_train = vectorize_with_emojis(X_train_preprocessed, emoji_vectors)
    emoji_matrix_valid = vectorize_with_emojis(X_valid_preprocessed, emoji_vectors)
    emoji_matrix_test = vectorize_with_emojis(X_test_preprocessed, emoji_vectors)

    print("Vektorisieren mit BERT...")
    bert_matrix_train = vectorize_with_bert(X_train, tokenizer, bert_model)
    bert_matrix_valid = vectorize_with_bert(X_valid, tokenizer, bert_model)
    bert_matrix_test = vectorize_with_bert(X_test, tokenizer, bert_model)

    # 8. Features kombinieren und skalieren
    print("Kombinieren und Skalieren der Features...")
    combined_train = np.hstack(
        [tfidf_matrix_train.toarray(), glove_matrix_train, emoji_matrix_train, bert_matrix_train]
    )
    combined_valid = np.hstack(
        [tfidf_matrix_valid.toarray(), glove_matrix_valid, emoji_matrix_valid, bert_matrix_valid]
    )
    combined_test = np.hstack(
        [tfidf_matrix_test.toarray(), glove_matrix_test, emoji_matrix_test, bert_matrix_test]
    )

    scaler = StandardScaler()
    combined_train_scaled = scaler.fit_transform(combined_train)
    combined_valid_scaled = scaler.transform(combined_valid)
    combined_test_scaled = scaler.transform(combined_test)
    print("Features erfolgreich kombiniert und skaliert.")

    # 9. Modelltraining (LightGBM)
    print("Starte Training mit LightGBM...")
    model = train_lightgbm(combined_train_scaled, y_train)

    # 10. Bewertung
    print("Bewertung des Modells...")
    accuracy, conf_matrix, report = evaluate_model(model, combined_test_scaled, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)
    plot_confusion_matrix(conf_matrix)
