🕵️ Fake Job Posting Detection

A Machine Learning project that detects whether a job posting is real or fake using Natural Language Processing (NLP) and a Random Forest Classifier.

This project helps identify fraudulent job ads that often mislead job seekers.

📌 Features

Preprocessing of raw job postings (title, description, requirements, etc.)

TF-IDF Vectorization for text feature extraction

Random Forest Classifier for classification

SMOTE to handle class imbalance

Web interface built with Flask (app.py)

Model persistence with Pickle (model.pkl, vectorizer.pkl)

📂 Project Structure
Fake-Job-Posting-Detection/
├── data/
│   └── FakeJobPostings2.xlsx         # Dataset
├── models/
│   ├── model.pkl                     # Trained Random Forest model
│   └── vectorizer.pkl                # TF-IDF vectorizer
├── templates/
│   └── index.html                    # Web UI (Flask)
├── static/
│   └── style.css                     # CSS for UI
├── app.py                            # Flask web app
├── main.py                           # Script to train/test model
├── train_model.py                    # Training pipeline
├── run_all.py                        # Combined execution script
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation

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


Then open: http://127.0.0.1:5000/
 in your browser

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
