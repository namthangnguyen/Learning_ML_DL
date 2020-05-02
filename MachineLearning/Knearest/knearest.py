import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # evaluation method

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)
print('Print results for 20 test data points:')
print('Ground truth          : ', y_test[20:40])

# K = 1, với mỗi test data point, ta chỉ xét một điểm training data gần nhất và lấy label của nó để tự đoán điểm test
clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("1NN's predicted labels: ", y_pred[20:40])
print('Accuracy of 1NN: ', accuracy_score(y_test, y_pred))

# Nếu chỉ xét điểm gần nhất sẽ dẫn đến kết quả sai nếu điểm đó là nhiễu
# Để tăng độ chính xác ta có thể tăng số lượng điểm lân cận lên, class nào chiếm đa số thì ta sẽ lấy kết quả đó
# Kỹ thuật này có tên "major voting"
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy of 10NN: ', accuracy_score(y_test, y_pred))

# Đánh trọng số cho các điểm lân cận
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy of 10NN (with 1/distance weights): ', accuracy_score(y_test, y_pred))


# Trọng số tự định nghĩa
def myWeight(distance):
    sigma2 = .5
    return np.exp(-distance**2/sigma2)


clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights=myWeight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy of 10NN (customized weights): ', accuracy_score(y_test, y_pred))
