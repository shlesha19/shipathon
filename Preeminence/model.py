
# tf-idf + sdg classifier method

import pandas as pd
import re
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# data loading from datasets
train_df = pd.read_csv(r"C:\Users\Shlesha Gupta\Downloads\Preeminence\training.csv")

test_df  = pd.read_csv(r"C:\Users\Shlesha Gupta\Downloads\Preeminence\testing.csv")

TEXT_COL = "Lyrics"
LABEL_COL = "Genre"

#text cleaning
def clean_text(text):
    text = str(text).lower()
    return re.sub(r"[^a-z\s]", "", text)

train_df[TEXT_COL] = train_df[TEXT_COL].apply(clean_text)
test_df[TEXT_COL]  = test_df[TEXT_COL].apply(clean_text)

#encoding labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df[LABEL_COL])

#tf-idf
tfidf = TfidfVectorizer(
    max_features=8000,        # ↓ smaller = faster
    ngram_range=(1, 1),       # unigrams only
    stop_words="english"
)

X = tfidf.fit_transform(train_df[TEXT_COL])
X_test_final = tfidf.transform(test_df[TEXT_COL])

#train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# sgd classifier implementation
model = SGDClassifier(
    loss="log_loss",          # logistic regression
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# TRAIN
model.fit(X_train, y_train)

#validate
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# cv score
cv_score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()
print("CV Accuracy:", cv_score)

# TEST PREDICTIONS
test_preds_encoded = model.predict(X_test_final)
test_preds = label_encoder.inverse_transform(test_preds_encoded)

# output
output = pd.DataFrame({
    "Song": test_df["Song"],
    "Predicted_Genre": test_preds
})
print(output.head())

# SAVE FILE
output.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")


joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("\n✓ Models saved successfully!")
print("  - model.pkl")
print("  - tfidf.pkl")
print("  - label_encoder.pkl")