from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


def preprocess_text(text):
    """Preprocess the input text by stemming and removing stop words."""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("french"))
    words = text.lower().split()
    return " ".join([stemmer.stem(word) for word in words if word not in stop_words])


def make_features(df):
    y = df["is_comic"]

    df["processed_title"] = df["video_name"].apply(preprocess_text)

    vectorizer = CountVectorizer(min_df=2, max_df=0.95)

    X = vectorizer.fit_transform(df["processed_title"])

    return X, y, vectorizer
