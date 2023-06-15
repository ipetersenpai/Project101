import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("heart_disease.csv")

X = dataset.drop('target', axis=1)
y = dataset['target']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

rf = RandomForestClassifier()

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("10 Cross-Validation")

for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    rf.fit(X_train_fold, y_train_fold)
    y_pred = rf.predict(X_val_fold)
    
    accuracy = accuracy_score(y_val_fold, y_pred)
    precision = precision_score(y_val_fold, y_pred, average='macro')
    recall = recall_score(y_val_fold, y_pred, average='macro')
    cm = confusion_matrix(y_val_fold, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_val_fold, y_pred, average='macro')
    sensitivity = recall_score(y_val_fold, y_pred, pos_label=1)

    print(f"\nFold {i}:")
    print('Accuracy:', format(accuracy, '.3f'))
    print('Precision:', format(precision, '.3f'))
    print('Recall:', format(recall, '.3f'))
    print('Specificity:', format(specificity, '.3f'))
    print('Sensitivity:', format(sensitivity, '.3f'))
    print('F1 Score:', format(f1, '.3f'))

# Scatter plot
plt.scatter(X_test['age'], X_test['chol'], c=y_test, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Scatter Plot of Age vs. Cholesterol with Target Labels')
plt.show()
