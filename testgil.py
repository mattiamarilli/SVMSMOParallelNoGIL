#!/usr/bin/env python3
import sys
import sysconfig
import time
from threading import Thread
from multiprocessing import Process


# Decorator to measure execution time of functions
def calculate_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.4f} seconds.")
        return result

    return wrapper


# Compute-intensive task: Iterative Fibonacci calculation
def compute_fibonacci(n):
    """Compute Fibonacci number for a given n iteratively."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# Single-threaded task execution
@calculate_execution_time
def run_single_threaded(nums):
    for num in nums:
        compute_fibonacci(num)


# Multi-threaded task execution
@calculate_execution_time
def run_multi_threaded(nums):
    threads = [Thread(target=compute_fibonacci, args=(num,)) for num in nums]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


# Multi-processing task execution
@calculate_execution_time
def run_multi_processing(nums):
    processes = [Process(target=compute_fibonacci, args=(num,)) for num in nums]
    for process in processes:
        process.start()
    for process in processes:
        process.join()


# Main execution function
def main():
    # Check Python version and GIL status for Python 3.13+
    print(f"Python Version: {sys.version}")

    py_version = float(".".join(sys.version.split()[0].split(".")[:2]))
    status = sysconfig.get_config_var("Py_GIL_DISABLED")

    if py_version >= 3.13:
        status = sys._is_gil_enabled()

    if status is None:
        print("GIL cannot be disabled for Python <= 3.12")
    elif status == 0:
        print("GIL is currently disabled")
    elif status == 1:
        print("GIL is currently active")

    # Run tasks on the same input size for comparison
    nums = [300000] * 8

    print("\nRunning Single-Threaded Task:")
    run_single_threaded(nums)

    print("\nRunning Multi-Threaded Task:")
    run_multi_threaded(nums)

    print("\nRunning Multi-Processing Task:")
    run_multi_processing(nums)


if __name__ == "__main__":
    main()