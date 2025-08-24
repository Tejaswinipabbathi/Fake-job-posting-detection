# train_model.py
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load and clean dataset
df = pd.read_excel("FakeJobPostings2.xlsx")
df.fillna("", inplace=True)

# Use the existing pre-cleaned text column
df['text'] = df['text'].apply(lambda x: re.sub(r"[^a-zA-Z0-9]", " ", str(x).lower()))


X = df['text']
y = df['fraudulent']

vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

smote = SMOTE()
X_res, y_res = smote.fit_resample(X_vec, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("✅ Model and vectorizer saved.")
