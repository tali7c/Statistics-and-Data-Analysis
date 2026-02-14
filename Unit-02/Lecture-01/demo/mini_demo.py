import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("data/income_small.csv")
    values = df["income"].to_numpy()

    mean = np.mean(values)
    median = np.median(values)
    mode = pd.Series(values).mode().tolist()

    print("Income (small) dataset:")
    print(values.tolist())
    print(f"Mean:   {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode:   {mode}")

    # Add a large outlier for demonstration
    values_outlier = np.append(values, 200)
    mean_out = np.mean(values_outlier)
    median_out = np.median(values_outlier)

    print("\nAfter adding outlier (200):")
    print(f"Mean:   {mean_out:.2f}")
    print(f"Median: {median_out:.2f}")

    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(values_outlier, bins=8, edgecolor="black")
    plt.title("Income Dataset with Outlier")
    plt.xlabel("Income (thousands)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("images/income_histogram.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
