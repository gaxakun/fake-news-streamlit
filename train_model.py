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

print("Loading datasets...")

# 1. Old ISOT dataset
fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')
fake['label'] = 'FAKE'
true['label'] = 'REAL'
fake['text'] = fake['title'] + ' ' + fake['text']
true['text'] = true['title'] + ' ' + true['text']
df_isot = pd.concat([fake[['text', 'label']], true[['text', 'label']]], ignore_index=True)
print(f"ISOT: {len(df_isot)} samples")

# 2. Cleaned dataset: news_clean_only.csv
df_new1 = pd.read_csv('data/news_clean_only.csv')
print("Columns in news_clean_only.csv:", df_new1.columns.tolist())

# Detect text column (could be 'content', 'text', 'article')
text_col = None
for col in ['clean_text', 'text', 'content', 'article', 'headline', 'title']:
    if col in df_new1.columns:
        text_col = col
        break
if text_col is None:
    raise KeyError("No text column found in news_clean_only.csv")

# Detect label column (could be 'label_number', 'label', 'class')
label_col = None
for col in ['label_number', 'label', 'class', 'type']:
    if col in df_new1.columns:
        label_col = col
        break
if label_col is None:
    raise KeyError("No label column found in news_clean_only.csv")

# Convert labels to 'FAKE'/'REAL'
if label_col == 'label_number':
    df_new1['label'] = df_new1['label_number'].map({0: 'FAKE', 1: 'REAL'})
else:
    df_new1['label'] = df_new1[label_col].apply(lambda x: 'FAKE' if str(x).upper() in ['FAKE', '0'] else 'REAL')

df_new1['text'] = df_new1[text_col]
df_new1 = df_new1[['text', 'label']].dropna()
print(f"news_clean_only.csv: {len(df_new1)} samples")

# 3. Fake vs Real dataset: fake_real_news_dataset.csv
df_new2 = pd.read_csv('data/fake_real_news_dataset.csv')
print("Columns in fake_real_news_dataset.csv:", df_new2.columns.tolist())

# Detect text column (likely 'title' + 'content')
if 'title' in df_new2.columns and 'content' in df_new2.columns:
    df_new2['text'] = df_new2['title'] + ' ' + df_new2['content']
elif 'text' in df_new2.columns:
    text_col2 = 'text'
    df_new2['text'] = df_new2[text_col2]
elif 'article' in df_new2.columns:
    df_new2['text'] = df_new2['article']
else:
    raise KeyError("No text column found in fake_real_news_dataset.csv")

# Detect label column
if 'label' in df_new2.columns:
    df_new2['label'] = df_new2['label'].apply(lambda x: 'FAKE' if str(x).upper() in ['FAKE', '0', 'F'] else 'REAL')
elif 'class' in df_new2.columns:
    df_new2['label'] = df_new2['class'].apply(lambda x: 'FAKE' if str(x).upper() in ['FAKE', '0', 'F'] else 'REAL')
else:
    raise KeyError("No label column found in fake_real_news_dataset.csv")

df_new2 = df_new2[['text', 'label']].dropna()
print(f"fake_real_news_dataset.csv: {len(df_new2)} samples")

# Combine all datasets
df = pd.concat([df_isot, df_new1, df_new2], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nTotal combined: {len(df)} samples")
print(df['label'].value_counts())

print("\nPreprocessing...")
df['cleaned'] = df['text'].apply(preprocess_text)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

print("Training models...")
lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_tf, y_train)
nb = MultinomialNB().fit(X_train_tf, y_train)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42).fit(X_train_tf, y_train)

print("Saving models...")
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

print("\n✅ Training complete. Run: streamlit run app.py")