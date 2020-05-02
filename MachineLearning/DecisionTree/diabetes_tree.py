# https://www.datacamp.com/community/tutorials/decision-tree-classification-python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("indians-diabetes.csv", header=0, names=col_names)
print(pima.head())

# split dataset in features and target variable and perform 'Feature Selection'
# in this case, simply remove 'skin'
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols]
y = pima.label

# split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=24)

# Create Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model accuracy, how often is the classifier correct?
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Visualizing Decision Trees
plot_tree(clf, filled=True)
plt.show()

''' Optimizing Decision Tree Performance: read on link'''

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
plot_tree(clf, filled=True)
plt.show()