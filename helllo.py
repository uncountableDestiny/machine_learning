from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
ds = load_iris()

X = ds['data']
y = ds['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print('acc_train', acc_train)
print(f'acc test', acc_test)

