import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from utils.preprocess import preprocess_text

os.makedirs('models', exist_ok=True)

print("Loading ISOT dataset from data/ folder...")
fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')
fake['label'] = 'FAKE'
true['label'] = 'REAL'
df = pd.concat([fake, true], ignore_index=True)
df['text'] = df['title'] + ' ' + df['text']
df = df[['text', 'label']].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Loaded {len(df)} samples: {df['label'].value_counts().to_dict()}")

print("Preprocessing...")
df['cleaned'] = df['text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

print("Vectorizing with 5000 features...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

print("Training models...")
lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_tf, y_train)
nb = MultinomialNB().fit(X_train_tf, y_train)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42).fit(X_train_tf, y_train)

# Save with maximum compression
print("Saving models with compression...")
joblib.dump(lr, 'models/lr_model.pkl', compress=9)
joblib.dump(nb, 'models/nb_model.pkl', compress=9)
joblib.dump(pac, 'models/pac_model.pkl', compress=9)
joblib.dump(vectorizer, 'models/vectorizer.pkl', compress=9)

print("\nEvaluation:")
for name, model in [('LR', lr), ('NB', nb), ('PAC', pac)]:
    pred = model.predict(X_test_tf)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, pos_label='FAKE')
    rec = recall_score(y_test, pred, pos_label='FAKE')
    f1 = f1_score(y_test, pred, pos_label='FAKE')
    print(f"{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

print("\n✅ Training complete! Run: streamlit run app.py")