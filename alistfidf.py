import tkinter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sympy.codegen.ast import integer
from tkinter import Frame




#model her çalıştığında kendini yine eğitir. Hızlı çalıştığı için kaydetme gereği duymadım.
# --- MODEL EĞİTİMİ ---
def guestfidf(test_text, result_labels_file,frame):


    for label in result_labels_file.values():
        label.place_forget()

    # ----Veri yükleme----
    df = pd.read_csv("data/20000_turkish_news_title.csv")
    df = df.dropna(subset=["title", "clickbait"])

    texts = df["title"].astype(str)
    labels = df["clickbait"].astype(int)

    # Eğitim / test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression modeli
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)

    # ----İç fonksiyon----
    def predict_clickbait(text):
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        proba = model.predict_proba(text_tfidf)[0]
        return prediction, proba

    # ----Kullanım----
    pred, proba = predict_clickbait(test_text)

    result_labels_file["result"].config(text="️Result: Clickbait" if pred == 1 else "Result:Normal")
    result_labels_file["probabilities"].config(text=f"Probabilities:{proba}")
    result_labels_file["np"].config(text=f"Normal Probability:{proba[0]}")
    result_labels_file["cb"].config(text=f"Clikbait Probability:{proba[1]}")
    result_labels_file["error"].config(text=" No content was found to analyze from the file.")


    print("Result:", "Clickbait" if pred == 1 else "Normal")
    print("Probabilities:", proba)
    print("normal probability:", proba[0])
    print("clickbait probability:", proba[1])
    return result_labels_file
