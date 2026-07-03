


#  STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

warnings.filterwarnings('ignore')
print("✅ Libraries imported successfully!")



#  STEP 2: Load Dataset

df = pd.read_csv('credit_risk_dataset.csv')

print("\n📊 Dataset Shape:", df.shape)
print("\n📋 First 5 Rows:")
print(df.head())
print("\n🔍 Column Names:", df.columns.tolist())



# ✅ STEP 3: EDA

print("\n📈 Dataset Info:")
print(df.info())

print("\n❓ Missing Values:")
print(df.isnull().sum())

print("\n🎯 Target Column Distribution:")
print(df['loan_status'].value_counts())

plt.figure(figsize=(6, 4))
df['loan_status'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.title('Loan Status Distribution (0 = No Default, 1 = Default)')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('loan_status_distribution.png')
plt.close()
print("✅ EDA plot saved!")



# ✅ STEP 4: Fix ALL Missing Values (CORRECTED)


# Fill ALL numerical columns with median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill ALL categorical columns with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Drop any remaining NaN rows just in case
df.dropna(inplace=True)

print("\n✅ All missing values fixed!")
print("Remaining missing values:", df.isnull().sum().sum())



# ✅ STEP 5: Encode Categorical Columns

categorical_cols = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

df = pd.get_dummies(df, columns=categorical_cols)
print("\n✅ Categorical columns encoded!")
print("New Shape:", df.shape)



# ✅ STEP 6: Split & Scale

X = df.drop('loan_status', axis=1)
y = df['loan_status']

print("\n✅ Features Shape:", X.shape)
print("✅ Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Final check — no NaN allowed before training
assert not np.isnan(X_train).any(), "NaN found in X_train!"
assert not np.isnan(X_test).any(),  "NaN found in X_test!"

print("✅ Data split and scaled successfully!")
print(f"   Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")



# ✅ STEP 7: Train 3 Models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42)
}

trained_models = {}

print("\n🚀 Training Models...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"✅ {name} — Trained!")



# ✅ STEP 8: Evaluate All Models

print("\n" + "="*60)
print("📊 MODEL EVALUATION RESULTS")
print("="*60)

results = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    results[name] = auc

    print(f"\n{'='*40}")
    print(f"🔹 {name}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred))
    print(f"🏆 ROC-AUC Score: {auc:.4f}")



# ✅ STEP 9: ROC Curve

plt.figure(figsize=(8, 6))

for name, model in trained_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Credit Scoring Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()
print("✅ ROC Curve saved!")



# ✅ STEP 10: Confusion Matrix

best_model_name = max(results, key=results.get)
best_model      = trained_models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name} (AUC = {results[best_model_name]:.4f})")

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title(f'Confusion Matrix — {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("✅ Confusion Matrix saved!")



# ✅ STEP 11: Feature Importance (Random Forest)

if "Random Forest" in trained_models:
    rf_model    = trained_models["Random Forest"]
    feature_imp = pd.Series(
        rf_model.feature_importances_,
        index=df.drop('loan_status', axis=1).columns
    ).sort_values(ascending=False)[:15]

    plt.figure(figsize=(10, 6))
    feature_imp.plot(kind='bar', color='steelblue')
    plt.title('Top 15 Feature Importances — Random Forest')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("✅ Feature Importance saved!")



# ✅ STEP 12: Save Model & Scaler

joblib.dump(best_model, 'credit_scoring_model.pkl')
joblib.dump(scaler,     'scaler.pkl')

print("\n✅ Model saved as 'credit_scoring_model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")



# ✅ STEP 13: Open All Output Images

print("\n📊 Opening all output images...")
os.startfile('loan_status_distribution.png')
os.startfile('roc_curve.png')
os.startfile('confusion_matrix.png')
os.startfile('feature_importance.png')


# ✅ FINAL SUMMARY

print("\n" + "="*60)
print("🎯 FINAL SUMMARY — Model Performance")
print("="*60)
for name, auc in results.items():
    print(f"   {name:25s} → ROC-AUC: {auc:.4f}")

print(f"\n🏆 Best Model: {best_model_name}")
print("\n✅ Task 1 Complete! Files saved:")
print("   📁 credit_scoring_model.pkl")
print("   📁 scaler.pkl")
print("   📁 roc_curve.png")
print("   📁 confusion_matrix.png")
print("   📁 feature_importance.png")
print("   📁 loan_status_distribution.png")
print("\n🚀 Now upload to GitHub & post on LinkedIn!")
