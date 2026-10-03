import re
import joblib
import streamlit as st
import pandas as pd

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean(text):
    text = text.lower()
    text = re.sub(r"\(reuters\)", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

st.title("📰 Fake News Detector")
st.write("Paste a news headline or article to check if it is fake or real.")

user_text = st.text_area("News text", height=200)

if st.button("Check"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([clean(user_text)])
        pred = model.predict(vec)[0]
        conf = model.predict_proba(vec)[0].max() * 100
        if pred == 1:
            st.success(f"✅ Real News ({conf:.1f}% confidence)")
        else:
            st.error(f"❌ Fake News ({conf:.1f}% confidence)")




            import pandas as pd

st.divider()
st.subheader("📂 Batch check (CSV upload)")
st.write("Upload a CSV with a column named **text**.")

file = st.file_uploader("Choose CSV file", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    if "text" not in data.columns:
        st.error("CSV must have a column named 'text'.")
    else:
        data = data.dropna(subset=["text"])
        vecs = vectorizer.transform(data["text"].astype(str).apply(clean))
        data["prediction"] = ["Real" if p == 1 else "Fake" for p in model.predict(vecs)]
        data["confidence_%"] = (model.predict_proba(vecs).max(axis=1) * 100).round(1)
        st.dataframe(data[["text", "prediction", "confidence_%"]])
        st.download_button("Download results", data.to_csv(index=False), "results.csv", "text/csv")