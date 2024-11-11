# from KNNClassifier import KNNClassifier
import numpy as np
from joblib import Parallel, delayed
import time

# Example with random data
rows = 100000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows * cols).reshape((rows, cols))
y_train = np.random.randint(2, size=rows)
print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')

# Create random indices to test
test_size = 1000
X_test = np.random.randint(rows, size=test_size)

# The version with parallel

# Define the KNN classifier class
class KNNClassifier_p:
    def __init__(self, k=3):
        self.k = k  # Number of neighbors

    def fit(self, X, y):
        self.X_train = X  # Store the training data
        self.y_train = y  # Store the training labels

    def euclidean_distance(self, x1, x2):
        # Calculate the Euclidean distance between two points
        diff = (x1 - x2)
        sqr_diff = diff ** 2
        sqr_diff_sum = np.sum(sqr_diff)
        return np.sqrt(sqr_diff_sum)

    def predict(self, X):
        # Predict labels for the provided data using parallel processing
        y_pred = Parallel(n_jobs=-1)(delayed(self._predict)(x) for x in X)  # Use all available cores
        return np.array(y_pred)

    def _predict(self, x):
        # Calculate distances from the input point to all training points
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort distances and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label among the k nearest neighbors
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Create an instance of the KNN classifier
knn_p = KNNClassifier_p(k=2)
# Fit the model
knn_p.fit(X_train, y_train)

# Initialize variables to record total correct predictions and total time
total_correct_predictions = 0
correct_predictions = []
times = []
cpu_times = []

# Perform 30 runs of training and prediction
for i in range(30):
    # Record start time
    start_time = time.time()
    start_cpu = time.process_time()

    # Make predictions using the test samples
    predictions = knn_p.predict(X_train[X_test])
    correct_count = np.sum(y_train[X_test] == predictions)  # Count correct predictions

    # Update total correct predictions
    total_correct_predictions += correct_count

    # Record end time
    end_time = time.time()
    end_cpu = time.process_time()
    elapsed_cpu_time = end_cpu - start_cpu
    elapsed_time = end_time - start_time
    times.append(elapsed_time)
    cpu_times.append(elapsed_cpu_time)

    # Record and print results for this run
    correct_predictions.append(correct_count)
    print(f"Run {i+1}: Correct Predictions = {correct_count}, Time = {elapsed_time:.4f} seconds, Time (CPU) = {elapsed_cpu_time:.4f} seconds")

    # Output the results for this run
    print(f'Run {i+1}: Correct Predictions = {correct_count}, Time = {elapsed_time:.4f} seconds')

# Calculate total times, averages and standard deviations
average_correct = total_correct_predictions / 30
average_time = np.mean(times)
average_cpu_time = np.mean(cpu_times)
sd_time = np.std(times)
sd_cpu_time = np.std(cpu_times)
total_time = np.sum(times)
total_cpu_time = np.sum(cpu_times)
print(f"\nAverage Correct Predictions over 30 runs: {average_correct}")
print(f"Total Execution Time over 30 runs: {total_time:.4f} seconds")
print(f"Average Running Time over 30 runs: {average_time:.4f} seconds")
print(f"Average CPU Running Time over 30 runs: {average_cpu_time:.4f} seconds")
print(f"Standard Deviation of Running Time over 30 runs: {sd_time:.4f} seconds")
print(f"Standard Deviation of CPU Running Time over 30 runs: {sd_cpu_time:.4f} seconds")
