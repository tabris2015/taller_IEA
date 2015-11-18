from sklearn import svm
X = [[0,0], [0,1], [1,0], [1,1]]
Y = [0, 1, 1, 0]
clf = svm.SVC()
clf.fit(X, Y) 

print clf.predict([[0,0]])
print clf.predict([[1,0]])
print clf.predict([[0,1]])
print clf.predict([[1,1]])