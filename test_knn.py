from KNNClassifier import KNNClassifier
import numpy as np

# Example with random data
rows = 100000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows*cols).reshape((rows,cols))
y_train = np.random.randint(2, size=rows)
print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')

knn = KNNClassifier(k=2)
knn.fit(X_train, y_train)

# Create random indices to test
test_size = 1000
X_test = np.random.randint(rows, size=test_size)

# Generate Predictions
predictions = knn.predict(X_train[X_test])
#print(f'Prediction {predictions}')
#print(f'Label      {y_train[X_test]}')
# Calculate the number of equal elements
print(f'correct {np.sum(y_train[X_test] == predictions)}')

