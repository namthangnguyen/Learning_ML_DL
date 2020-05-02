from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# load the iris datasets
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=24)

# fit a GaussianNB model to the data 
clf = GaussianNB()
clf.fit(X_train, y_train)

# make predictitons
expected = y_test
predicted = clf.predict(X_test)

# sumarize the fit of the model
print('Training size = %d, accuracy = %.2f%%' % (X_train.shape[0], accuracy_score(expected, predicted) * 100))
print(classification_report(expected, predicted))
