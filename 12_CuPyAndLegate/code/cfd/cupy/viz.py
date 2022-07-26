import matplotlib
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the performance data for double, constant timesteps
    gridsize_500ts = np.array([41, 81, 128, 256, 512])
    data_cpu_500ts = np.array([0.70, 1.30, 2.80, 13.93, 61.11])
    data_gpu_500ts = np.array([6.49, 6.54, 6.44, 6.55, 6.42])

    gridsize_4000ts = np.array([41, 81, 128, 256, 512])
    data_cpu_4000ts = np.array([5.41, 10.76, 21.97, 51.22, 494.04])
    data_gpu_4000ts = np.array([51.78, 51.53, 51.89, 51.22, 51.66])

    # Load the performance data for double, constant gridsize

    # Plot
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.plot(gridsize_500ts, data_cpu_500ts, marker='o', label='Numpy')
    plt.plot(gridsize_500ts, data_gpu_500ts, marker='o', label='Cupy')
    plt.xticks([41, 81, 128, 256, 512])
    plt.title("Cavity Flow Numpy vs. Cupy (500 Timesteps)")
    plt.xlabel("Grid Size", size=14)
    plt.ylabel("Wall time (seconds)", size=14)
    plt.legend()
    plt.subplot(122)
    plt.plot(gridsize_4000ts, data_cpu_4000ts, marker='o', label='Numpy')
    plt.plot(gridsize_4000ts, data_gpu_4000ts, marker='o', label='Cupy')
    plt.xticks([41, 81, 128, 256, 512])
    plt.title("Cavity Flow Numpy vs. Cupy (4000 Timesteps)")
    plt.xlabel("Grid Size", size=14)
    plt.ylabel("Wall time (seconds)", size=14)
    plt.legend()
    # plt.subplot(221)
    # plt.plot(gridsize_4000ts, data_cpu_4000ts, marker='o', label='Numpy')
    # plt.plot(gridsize_4000ts, data_gpu_4000ts, marker='o', label='Cupy')
    # plt.xticks([41, 81, 128, 256, 512])
    # plt.xlabel("Grid Size")
    # plt.ylabel("Wall time (seconds)")
    # plt.legend()
    # plt.subplot(222)
    # plt.plot(gridsize_4000ts, data_cpu_4000ts, marker='o', label='Numpy')
    # plt.plot(gridsize_4000ts, data_gpu_4000ts, marker='o', label='Cupy')
    # plt.xticks([41, 81, 128, 256, 512])
    # plt.xlabel("Grid Size")
    # plt.ylabel("Wall time (seconds)")
    # plt.legend()