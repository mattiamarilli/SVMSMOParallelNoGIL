import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from smoparallel import SVM


def benchmark_svm(dataset_sizes, feature_sizes, thread_counts, runs=10, test_size=0.2, random_state=42,
                  output_file="svm_benchmark_report.txt"):
    results = []

    with open(output_file, "w") as f:
        f.write("SVM Benchmark Report (Media su {} run)\n".format(runs))
        f.write("=" * 100 + "\n\n")

        for num_samples in dataset_sizes:
            for num_features in feature_sizes:
                f.write(f"Dataset: {num_samples} samples, {num_features} features\n")
                f.write("-" * 100 + "\n")

                # --- Generazione dataset ---
                X, y = make_classification(
                    n_samples=num_samples,
                    n_features=num_features,
                    n_informative=int(num_features * 0.6),
                    n_redundant=int(num_features * 0.1),
                    n_classes=2,
                    n_clusters_per_class=1,
                    random_state=random_state
                )
                y = 2 * y - 1  # Converte {0,1} → {-1,1}

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                gamma = 1.0 / num_features

                dataset_result = {"num_samples": num_samples, "num_features": num_features, "threads": {}}

                # --- Esegui test per ogni configurazione di thread ---
                for num_threads in thread_counts:
                    training_times = []
                    sum_columns_times = []
                    prediction_times = []
                    accuracies = []

                    for run in range(runs):
                        model = SVM(
                            numthreads=num_threads,
                            c=1.0,
                            kkt_thr=1e-3,
                            max_iter=2000,
                            kernel_type='rbf',
                            gamma_rbf=gamma
                        )

                        # Training
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        training_times.append(time.time() - start_time)
                        sum_columns_times.append(model.sum_columns_calculation_time)

                        # Prediction & Accuracy
                        start_time = time.time()
                        y_pred, _ = model.predict(X_test)
                        prediction_times.append(time.time() - start_time)
                        accuracies.append(np.mean(y_pred == y_test))

                    # --- Media su tutte le run ---
                    avg_training_time = np.mean(training_times)
                    avg_sum_columns_time = np.mean(sum_columns_times)
                    avg_prediction_time = np.mean(prediction_times)
                    avg_accuracy = np.mean(accuracies)

                    dataset_result["threads"][num_threads] = {
                        "training_time": avg_training_time,
                        "sum_columns_time": avg_sum_columns_time,
                        "prediction_time": avg_prediction_time,
                        "accuracy": avg_accuracy
                    }

                # --- Calcolo speedup ed efficiency rispetto a 1 thread ---
                base_time = dataset_result["threads"][1]["sum_columns_time"]

                f.write(
                    f"{'Threads':>8} | {'Train (s)':>10} | {'Colonne (s)':>12} | {'Pred (s)':>9} | {'Accuracy (%)':>13} | {'Speedup':>7} | {'Efficiency (%)':>14}\n")
                f.write("-" * 100 + "\n")
                for num_threads in thread_counts:
                    r = dataset_result["threads"][num_threads]
                    speedup = base_time / r["sum_columns_time"]
                    efficiency = (speedup / num_threads) * 100
                    f.write(
                        f"{num_threads:>8} | {r['training_time']:>10.3f} | {r['sum_columns_time']:>12.3f} | {r['prediction_time']:>9.3f} | {r['accuracy'] * 100:>13.2f} | {speedup:>7.2f} | {efficiency:>14.1f}\n")
                f.write("\n\n")
                results.append(dataset_result)

    print(f"Benchmark completato. Report salvato su '{output_file}'")
    return results


if __name__ == "__main__":
    dataset_sizes = [3000]
    feature_sizes = [20,50,100,200,400]
    thread_counts = [1,2,4,8,16]

    benchmark_svm(dataset_sizes, feature_sizes, thread_counts, runs=1)
