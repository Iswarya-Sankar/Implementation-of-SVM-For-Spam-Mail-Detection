# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Import the required libraries:
      pandas
      train_test_split
      CountVectorizer
      SVC
      accuracy_score and classification_report
3. Load the spam.csv dataset using pandas.
4. Select the required columns from the dataset:
      v1 → label
      v2 → message
5. Rename the columns as:
    label
    message
6. Convert the labels into numerical values:
      ham → 0
      spam → 1
7. Separate input and output data:
      X → message column
      y   → label column
8. Convert text messages into numerical vectors using CountVectorizer.
9. Split the dataset into training and testing data using train_test_split.
10. Create the SVM classifier model using linear kernel.
11. Train the SVM model using training data.
12. Predict the output using test data.
13. Calculate the accuracy of the model.
14. Display the classification report.
15. Stop the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Iswarya S
RegisterNumber: 212225040135
*/\

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(df.head())

# Select necessary columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels into numeric values
# ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Input and output data
X = df['message']
y = df['label']

# Convert text messages into numerical vectors
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SVM model
svm_model = SVC(kernel='linear')

# Train the model
svm_model.fit(X_train, y_train)

# Predict test data
y_pred = svm_model.predict(X_test)

# Print accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Print classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
```

## Output:
<img width="620" height="450" alt="image" src="https://github.com/user-attachments/assets/5a610783-0a63-4116-bd0b-8ec92d2f1233" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
