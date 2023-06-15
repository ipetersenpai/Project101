import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn import tree

dataset = pd.read_csv("heart_disease.csv")

X = dataset.drop('target', axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dtc = DecisionTreeClassifier()

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("10 Cross-Validation")

for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    dtc.fit(X_train_fold, y_train_fold)
    y_pred = dtc.predict(X_val_fold)

    print(f"\nFold {i}:")
    accuracy = accuracy_score(y_val_fold, y_pred)
    precision = precision_score(y_val_fold, y_pred, average='weighted')
    recall = recall_score(y_val_fold, y_pred, average='weighted')
    cm = confusion_matrix(y_val_fold, y_pred, labels=y.unique())
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    f1 = f1_score(y_val_fold, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F1 Score:", f1)


class_names = y.unique().astype(str)

plt.figure(figsize=(10, 8))
tree.plot_tree(dtc, feature_names=X.columns, class_names=class_names, filled=True)
plt.show()
