# 📰 Fake News Detector

A machine learning project that classifies news headlines and articles as **real** or **fake** using Natural Language Processing (NLP).

## Overview

News text is converted into numerical features with a TF-IDF vectorizer, and a trained classifier predicts whether the text is real or fake. The included app lets you test the model on your own text.

## Features

- Classifies news text as Real or Fake
- Text vectorization with TF-IDF
- Pre-trained model and vectorizer included (`model.pkl`, `vectorizer.pkl`)
- Simple app for predictions (`app.py`)
- Sample input and output files for quick testing (`test.csv`, `output.txt`)

## Project Structure

```
fake_news_detector/
├── data/              # Dataset (not included, see below)
├── app.py             # App for making predictions
├── train.py           # Model training script
├── model.pkl          # Trained model
├── vectorizer.pkl     # Trained TF-IDF vectorizer
├── test.csv           # Sample input
├── output.txt         # Sample predictions
├── .gitignore
└── README.md
```

## Dataset

The datasets are too large for this repository, so they are not included.

1. Download **Fake.csv** and **True.csv** from Kaggle (search for "Fake and Real News Dataset").
2. Create a folder named `data/` in the project root.
3. Place both files inside it.

## Installation

```bash
git clone https://github.com/DILSHA-A/fake_news_detector.git
cd fake_news_detector
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

**Train the model** (optional, a trained model is already included):

```bash
python train.py
```

**Run the app:**

```bash
streamlit run app.py
```

## Example

| Input | Prediction |
|-------|------------|
| Senate passes budget bill after lengthy debate | Real |
| SHOCKING: Hillary Clinton secretly arrested, media hiding the truth!!! | Fake |

## Tech Stack

- Python
- scikit-learn
- pandas
- Streamlit

## Future Improvements

- Try deep learning models such as LSTM or BERT
- Show a confidence score with each prediction
- Deploy the app online

## Author

**DILSHA A** · [GitHub](https://github.com/DILSHA-A)
