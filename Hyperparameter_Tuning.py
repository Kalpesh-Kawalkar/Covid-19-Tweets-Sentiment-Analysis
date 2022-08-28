#Error Calculation
error = []
for i in range(1,60):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

#K Value vs Mean Error
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.plot(range(1, 60), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate for K value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

#HyperParameter Tuning for SVM Classifier
from sklearn.svm import SVC
parameter_grid = {'C' : [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel' : ['rbf']}

#Random Grid Search
from sklearn.model_selection import RandomizedSearchCV
rf_grid = RandomizedSearchCV (estimator = SVM_model, param_distributions = parameter_grid, cv = 3, verbose = 2, n_jobs = 4)
rf_grid.fit(x,y)

rf_grid.best_estimator_