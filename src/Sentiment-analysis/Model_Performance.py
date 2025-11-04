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

#Model Performance

# Download to local machine
'''from google.colab import files
files.download("lstm_model.h5")
files.download("lstm_tokenizer.pkl")'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#import data
from train_test import X_test, y_test
#import predictions from models
from Model1_random_forest import y_pred_rf
from Model2_LSTM import y_pred_lstm

# Collect metrics
results = {
    "Model": ["Random Forest", "LSTM"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_lstm),

    ],
    "Precision": [
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_lstm),

    ],
    "Recall": [
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_lstm),

    ],
    "F1-Score": [
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_lstm),

    ],
}

df_results = pd.DataFrame(results)
print(" Model Performance Summary")
print(df_results)

