train_path = "amazon_reviews_train.csv"
test_path  = "amazon_reviews_test.csv"


import re
import nltk
import pandas as pd
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Basic cleaning: lowercasing, removing special chars, stopwords, lemmatization"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)


train_df = pd.read_csv(train_path, nrows=50000)
test_df  = pd.read_csv(test_path, nrows=10000)


train_df["text"] = (train_df["title"].astype(str) + " " + train_df["content"].astype(str)).apply(clean_text)
test_df["text"]  = (test_df["title"].astype(str) + " " + test_df["content"].astype(str)).apply(clean_text)

X_train, y_train = train_df["text"], train_df["label"]
X_test,  y_test  = test_df["text"],  test_df["label"]

print("Train size:", len(X_train), " Test size:", len(X_test))


#Predict
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved RF + TF-IDF
rf_loaded = joblib.load("random_forest_model.pkl")
vectorizer_loaded = joblib.load("tfidf_vectorizer.pkl")

def predict_rf(text):
    text_clean = clean_text(text)
    vec = vectorizer_loaded.transform([text_clean])
    pred = rf_loaded.predict(vec)[0]
    return "Positive " if pred == 1 else "Negative "

lstm_loaded = load_model("lstm_model.h5")
tokenizer_loaded = joblib.load("lstm_tokenizer.pkl")

def predict_lstm(text):
    text_clean = clean_text(text)
    seq = tokenizer_loaded.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=200)
    pred = (lstm_loaded.predict(padded) > 0.5).astype("int32")[0][0]
    return "Positive " if pred == 1 else "Negative "

sample_texts = [
    "This product is absolutely fantastic, I loved it!",
    "Worst purchase ever, total waste of money.",
    "It was okay, nothing special but not bad either."
]

for txt in sample_texts:
    print(f"\nINPUT: {txt}")
    print("Random Forest →", predict_rf(txt))
    print("LSTM          →", predict_lstm(txt))