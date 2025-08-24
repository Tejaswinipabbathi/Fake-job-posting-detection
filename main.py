# 📦 Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE
# 📁 Load dataset
df = pd.read_excel(r'C:\Users\teju0\OneDrive\Desktop\ml\Fake-Job-Posting-Detection-master\FakeJobPostings.xlsx')

# 🧹 Clean & preprocess data
df[['title', 'company_profile', 'description', 'requirements', 'benefits']] = df[[
    'title', 'company_profile', 'description', 'requirements', 'benefits']].fillna("")

# Combine all text fields into one
df['text'] = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits']
df['text'] = df['text'].apply(lambda x: re.sub(r"[^a-zA-Z0-9]", " ", x.lower()))

# 🔢 Count how many jobs are fake and how many are real
counts = df['fraudulent'].value_counts()
print(f"✅ Real job posts: {counts[0]}")
print(f"🚨 Fake job posts: {counts[1]}")

# 🧠 Feature Extraction
X = df['text']
y = df['fraudulent']
vectorizer = TfidfVectorizer(max_features=1000)
X_vectorized = vectorizer.fit_transform(X)

# ⚖️ Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_vectorized, y)

# Plot BEFORE SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x='fraudulent', data=df, palette='Set2')
plt.title("Real vs Fake Job Posts (Original Data)")
plt.xticks([0, 1], ['Real', 'Fake'])
plt.xlabel("Job Post Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 🔀 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# 🤖 Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 📈 Predict & Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc:.2f}\n")
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))

# 📊 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 🧪 Try with a custom example
sample = "Earn $5000 per week from home with no skills or experience needed!"
sample_clean = re.sub(r"[^a-zA-Z0-9]", " ", sample.lower())
sample_vec = vectorizer.transform([sample_clean])
prob = model.predict_proba(sample_vec)[0][1]
result = 1 if prob >= 0.5 else 0

print("\n🧪 Sample Job Description Prediction:")
print("Result:", "🚨 FAKE JOB POST" if result == 1 else "✅ REAL JOB POST")
print(f"🔢 Confidence of being fake: {prob:.2f}")
import pickle

# Save trained vectorizer
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

# Save trained model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
