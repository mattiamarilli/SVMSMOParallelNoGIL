import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from smoparallel import SVM

def benchmark_svm(dataset_sizes, feature_sizes, thread_counts, runs=10, test_size=0.2, random_state=42,
                  output_file="svm_benchmark_report.txt"):
    """
    Run a benchmark of the custom SVM using synthetic datasets, varying dataset size,
    number of features, and number of threads for parallel RBF computation.

    Args:
        dataset_sizes (list[int]): List of dataset sizes (number of samples).
        feature_sizes (list[int]): List of numbers of features.
        thread_counts (list[int]): List of thread counts to test.
        runs (int): Number of repetitions for averaging results.
        test_size (float): Fraction of data to use as test set.
        random_state (int): Seed for reproducibility.
        output_file (str): File path to save the benchmark report.

    Returns:
        list: List of dictionaries containing averaged benchmark results.
    """
    results = []

    with open(output_file, "w") as f:
        f.write("SVM Benchmark Report (Average over {} runs)\n".format(runs))
        f.write("=" * 100 + "\n\n")

        # Loop over dataset sizes
        for num_samples in dataset_sizes:
            for num_features in feature_sizes:
                f.write(f"Dataset: {num_samples} samples, {num_features} features\n")
                f.write("-" * 100 + "\n")

                # Generate synthetic classification dataset
                X, y = make_classification(
                    n_samples=num_samples,
                    n_features=num_features,
                    n_informative=int(num_features * 0.6),
                    n_redundant=int(num_features * 0.1),
                    n_classes=2,
                    n_clusters_per_class=1,
                    random_state=random_state
                )
                # Convert labels from 0/1 to -1/+1
                y = 2 * y - 1

                # Split dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                # Standardize features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Set gamma for RBF kernel
                gamma = 1.0 / num_features

                # Structure to store results for this dataset
                dataset_result = {"num_samples": num_samples, "num_features": num_features, "threads": {}}

                # Loop over different thread counts
                for num_threads in thread_counts:
                    training_times = []
                    sum_columns_times = []
                    prediction_times = []
                    accuracies = []

                    # Repeat multiple runs to average results
                    for run in range(runs):
                        # Initialize the SVM model
                        model = SVM(
                            numthreads=num_threads,
                            c=1.0,
                            kkt_thr=1e-3,
                            max_iter=2000,
                            kernel_type='rbf',
                            gamma_rbf=gamma
                        )

                        # Measure training time
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        training_times.append(time.time() - start_time)
                        sum_columns_times.append(model.sum_columns_calculation_time)  # Time spent computing RBF columns

                        # Measure prediction time
                        start_time = time.time()
                        y_pred, _ = model.predict(X_test)
                        prediction_times.append(time.time() - start_time)
                        accuracies.append(np.mean(y_pred == y_test))

                    # Compute averages
                    avg_training_time = np.mean(training_times)
                    avg_sum_columns_time = np.mean(sum_columns_times)
                    avg_prediction_time = np.mean(prediction_times)
                    avg_accuracy = np.mean(accuracies)

                    # Save averaged results for this thread count
                    dataset_result["threads"][num_threads] = {
                        "training_time": avg_training_time,
                        "sum_columns_time": avg_sum_columns_time,
                        "prediction_time": avg_prediction_time,
                        "accuracy": avg_accuracy
                    }

                # Compute speedup relative to single-threaded execution
                base_time = dataset_result["threads"][1]["sum_columns_time"]

                # Write the results to the report
                f.write(
                    f"{'Threads':>8} | {'Train (s)':>10} | {'Columns (s)':>12} | {'Pred (s)':>9} | {'Accuracy (%)':>13} | {'Speedup':>7} | {'Efficiency (%)':>14}\n")
                f.write("-" * 100 + "\n")
                for num_threads in thread_counts:
                    r = dataset_result["threads"][num_threads]
                    speedup = base_time / r["sum_columns_time"]
                    efficiency = (speedup / num_threads) * 100
                    f.write(
                        f"{num_threads:>8} | {r['training_time']:>10.3f} | {r['sum_columns_time']:>12.3f} | {r['prediction_time']:>9.3f} | {r['accuracy'] * 100:>13.2f} | {speedup:>7.2f} | {efficiency:>14.1f}\n")
                f.write("\n\n")
                results.append(dataset_result)

    print(f"Benchmark completed. Report saved to '{output_file}'")
    return results


if __name__ == "__main__":
    # Dataset sizes to test
    dataset_sizes = [3000]
    # Feature sizes to test
    feature_sizes = [20,50,100,200,400,600]
    # Thread counts to test
    thread_counts = [1,2,4,8,16]

    # Run the benchmark (10 runs per configuration)
    benchmark_svm(dataset_sizes, feature_sizes, thread_counts, runs=10)
