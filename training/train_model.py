import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/resume_dataset.csv")

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    text = re.sub(r'\d+', '', text)
    return text.lower()

df['cleaned'] = df['Resume'].apply(clean_text)

# ATS-style feature: resume length
df['resume_len'] = df['cleaned'].apply(lambda x: len(x.split()))

# ML Training
X = df['cleaned']
y = LabelEncoder().fit_transform(df['Category'])

vectorizer = TfidfVectorizer(max_features=1500)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

# Save models
with open("model/resume_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
