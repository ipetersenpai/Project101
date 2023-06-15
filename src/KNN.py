import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("heart_disease.csv")

X = dataset.drop('target', axis=1)
y = dataset['target']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# Create a K-Nearest Neighbors classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# 10-fold cross-validation
print('10 Cross-Validation:')
for i, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=10).split(X, y_encoded), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_test, y_pred, average='macro')
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    
    print(f'\nFold{i}:')
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
plt.title('Scatter Plot of Age vs. Cholesterol with KNN Predictions')
plt.show()
