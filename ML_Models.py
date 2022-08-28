#Splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Naive Bayes Classification
from sklearn.naive_bayes import MultinomialNB
m1 = MultinomialNB()
m1.fit(x_train, y_train)
print(m1.score(x_train,y_train))
print(m1.score(x_test,y_test))

#Predictions
ypred_m1 = m1.predict(x_test)
print(ypred_m1[:500])

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, ypred_m1)
print(cm)
print(classification_report(y_test,ypred_m1))

#KNN Classification
from sklearn.neighbors import KNeighborsClassifier
m2 = KNeighborsClassifier(n_neighbors = 1)
m2.fit(x_train,y_train)
print(m2.score(x_train,y_train))
print(m2.score(x_test,y_test))

#Predictions
ypred_m2 = m2.predict(x_test)
print(ypred_m2[:500])

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, ypred_m2)
print(cm)
print(classification_report(y_test,ypred_m2))

#SVM with linear kernel
m3 = SVC(kernel = 'linear', C=0.1, gamma=0.001)
m3.fit(x_train,y_train)
print(m3.score(x_train,y_train))
print(m3.score(x_test,y_test))
ypred_m3 = m3.predict(x_test)
print(ypred_m3[:500])

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, ypred_m3)
print(cm)
print(classification_report(y_test,ypred_m3))

#SVM with rbf kernel
m4 = SVC(kernel = 'rbf', C=100, gamma = 0.01)
m4.fit(x_train, y_train)
print(m4.score(x_train, y_train))
print(m4.score(x_test, y_test))
ypred_m4 = m4.predict(x_test)
print(ypred_m4[:500])

#Classification report and Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, ypred_m4)
print(cm)
print(classification_report(y_test,ypred_m4))
