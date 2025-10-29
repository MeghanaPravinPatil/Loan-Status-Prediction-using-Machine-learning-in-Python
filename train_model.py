# train_model.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/loan_data.csv"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "loan_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

features = [c for c in df.columns if c != 'Loan_Status']
X = df[features]
y = df['Loan_Status']

num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

num_trans = Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_trans = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

pre = ColumnTransformer([('num', num_trans, num_cols), ('cat', cat_trans, cat_cols)])

model = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_FILE)
print("Model saved at", MODEL_FILE)
