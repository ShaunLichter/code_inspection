
Below you will be given a set of code delimited by three quotes, i.e. """.
Your task is to answer several questions:
1. Does this code contains machine learning,
2. If it does have machine learning, what algorithm is used in the code?
3. What performance metrics are used to evaluate the model?
4. What hyperparameters are used to train the model?

The code to evaluate is
"""from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_split=4)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')"""


Answer 
"1. Yes, this code contains machine learning.
2. The algorithm used in the code is ExtraTreesClassifier.
3. The performance metrics used to evaluate the model are accuracy, precision, recall, and F1 score.
4. The hyperparameters used to train the model are n_estimators=100, max_depth=3, and min_samples_split=4."