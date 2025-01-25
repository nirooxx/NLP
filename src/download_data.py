import nltk

def download_nltk_data():
    nltk.download('twitter_samples')
    nltk.download('punkt')  # Für Tokenization
    nltk.download('stopwords')  # Für Stopwords
 #   nltk.download('punkt_tab')
if __name__ == "__main__":
    download_nltk_data()
