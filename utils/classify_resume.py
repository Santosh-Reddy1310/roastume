import pickle
import re

with open("model/resume_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

label_map = [
    "HR", "Data Science", "Advocate", "Arts", "Web Designing", "Mechanical Engineer",
    "Sales", "Health and fitness", "Civil Engineer", "Java Developer", "Business Analyst",
    "SAP Developer", "Automation Testing", "Electrical Engineering", "Operations Manager",
    "Python Developer", "DevOps Engineer", "Network Security Engineer", "PMO", "Database"
]

ml_keywords = [
    "machine learning", "tensorflow", "scikit", "llm", "vertex", "generative", "pytorch"
]

def clean_resume(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def predict_resume_category(text):
    clean = clean_resume(text)

    # Keyword boosting logic
    if any(k in clean for k in ml_keywords):
        return "Data Science"

    vec = vectorizer.transform([clean])
    pred = clf.predict(vec)[0]
    return label_map[pred]
