🕵️ Fake Job Posting Detection


A machine learning project that identifies whether a job posting is real or fake using text processing, TF-IDF vectorization, and a Random Forest Classifier. This helps job seekers avoid fraudulent ads.

📌 Features

Preprocessing of raw job postings (title, description, requirements, etc.)

TF-IDF Vectorization for text feature extraction

Random Forest Classifier for classification

SMOTE to handle class imbalance

Web interface built with Flask (app.py)

Model persistence with Pickle (model.pkl, vectorizer.pkl)

🛠️ Installation & Setup




1️⃣ Clone Repository
git clone https://github.com/Tejaswinipabbathi/Fake-Job-Posting-Detection.git



cd Fake-Job-Posting-Detection

2️⃣ Create Virtual Environment (Recommended)


python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

3️⃣ Install Dependencies


pip install -r requirements.txt

▶️ Usage


Train the Model
python train_model.py

Run the Flask App
python app.py


Then open: http://127.0.0.1:5000/ in your browser

📊 Results

Accuracy: ~95% (with SMOTE + Random Forest)

Confusion matrix, classification report included in main.py

Example Prediction:

Input: "Earn $5000 per week from home with no skills required!"
Output: 🚨 FAKE JOB POST (Confidence: 0.92)

📈 Dataset

Dataset: Fake Job Postings Dataset

Contains ~18,000 job ads (real & fake)

Includes: title, location, company profile, description, requirements, benefits

🚀 Future Improvements

Deploy as a Web App (Heroku/Render)

Use Deep Learning (LSTMs, Transformers) for better text understanding

Improve UI with modern frontend

👩‍💻 Author
Tejaswini Pabbathi
