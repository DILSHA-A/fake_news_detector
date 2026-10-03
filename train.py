import pandas as pd

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0   # 0 = fake
real["label"] = 1   # 1 = real

df = pd.concat([fake, real], ignore_index=True)
df = df[["title", "text", "label"]]
df = df.drop_duplicates().dropna()
df["content"] = df["title"] + " " + df["text"]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Clean text
def clean(text):
    text = text.lower()
    text = re.sub(r"\(reuters\)", "", text)   # remove source tag that leaks the answer
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["content"] = df["content"].apply(clean)

# 2. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=50000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 5. Evaluate
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=["Fake", "Real"]))
print(confusion_matrix(y_test, pred))

# 6. Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model saved!")