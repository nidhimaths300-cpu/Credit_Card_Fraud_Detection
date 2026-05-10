import pandas as pd
df = pd.read_csv('creditcard.csv')
print(df.head)
# 1. Check karein ki kitne Fraud (1) aur kitne Normal (0) cases hain
print("\n --- Class Distribution ---")
print(df['Class'].value_counts())
# 2. Percentage mein dekhein 
fraud_percentage = (df['Class'].value_counts()[1] / len(df))*100
print(f"\nFraud Transactions: {fraud_percentage: .2f}%")
from sklearn.preprocessing import StandardScaler
# Amount ko scale karenge taki wo model ko confuse na kare
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
# Purane 'Amount' aur 'Time' columns ko hata dein
df.drop(['Amount', 'Time'], axis=1, inplace=True )
X = df.drop('Class', axis=1) #Inputs
Y = df['Class'] # Target Result
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
# Data Split karenge
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
# Model Train karenge
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
# Prediction
Y_pred = model.predict(X_test)
# Result Check krenge
print("\n--- Confusion Matrix ---")
print(confusion_matrix(Y_test, Y_pred))
print("\n--- Classification Report ---")
print(classification_report(Y_test, Y_pred))
import matplotlib.pyplot as plt
import seaborn as sns
# Confusion Matrix ka Heatmap
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Credit Card Fraud')
plt.show()
# SMOTE (Synthetic Minority Over- sampling Technique)
from imblearn.over_sampling import SMOTE
# 1. SOMTE se data balance karna
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X_train, Y_train)
print(f"\nBalancing ke baad Fraud cases:{sum(Y_res==1)}")
# 2. Balanced data per dubara train karna
model.fit(X_res, Y_res)
# 3. Naya Prediction
Y_pred_new = model.predict(X_test)
# 4. New Results Dekhna
print("\n--- Balanced Model Classification Report ---")
print(classification_report(Y_test, Y_pred_new)) 
# Naya Confusion Matrix naye predictions ke liye
cm_new = confusion_matrix(Y_test, Y_pred_new)
plt.figure(figsize=(8,6))
sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix after SMOTE (Recall: 0.92)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()