import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
breast = pd.read_csv("breast cancer.csv")

# Drop unwanted column if it exists
if 'Unnamed: 32' in breast.columns:
    breast.drop('Unnamed: 32', axis=1, inplace=True)

# Encode the target variable
target_col = 'diagnosis'
le = LabelEncoder()
breast[target_col] = le.fit_transform(breast[target_col])

# Split dataset
X = breast.drop(target_col, axis=1)
y = breast[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate model
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Accuracy: {accuracy_lr:.4f}')

# Predict on a new sample input
input_text = np.array([
    -0.23717126, -0.64487029, -0.11382239, -0.57427777, -0.60294971,
     1.0897546 ,  0.91543814,  0.41448279,  0.09311633,  1.78465117,
     2.11520208,  0.28454765, -0.31910982,  0.2980991 ,  0.01968238,
    -0.47096352,  0.45757106,  0.28733283, -0.23125455,  0.26417944,
     0.66325388,  0.12170193,  0.42656325,  0.36885508,  0.02065602,
     1.39513782,  2.0973271 ,  2.01276347,  0.61938913,  2.9421769 ,
     3.15970842
])
pred = lr.predict(input_text.reshape(1, -1))
print("Cancerous" if pred[0] == 1 else "Not Cancerous")

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(lr, model_file)