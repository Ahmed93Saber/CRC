import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_patch_counts(h5_dir):
    """
    Iterates over all .h5 files in a directory and returns an array of patch counts.
    """
    counts = []
    file_names = []

    if not os.path.exists(h5_dir):
        print(f"Directory not found: {h5_dir}")
        return np.array(counts), file_names

    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    print(f"Found {len(h5_files)} .h5 files in {h5_dir}. Analyzing...")

    # Using tqdm to show a progress bar (useful for large WSI datasets)
    for file in tqdm(h5_files, desc="Reading h5 files"):
        file_path = os.path.join(h5_dir, file)
        try:
            with h5py.File(file_path, "r") as f:
                # In your datasets.py, features are accessed via f["features"]
                num_patches = f["features"].shape[0]
                counts.append(num_patches)
                file_names.append(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return np.array(counts), file_names


def analyze_and_plot(counts, title="Patch Count Distribution"):
    """
    Calculates percentiles and plots a histogram of the patch distribution.
    """
    if len(counts) == 0:
        print("No data to analyze.")
        return

    # Calculate key statistics
    stats = {
        "Total Slides": len(counts),
        "Min": np.min(counts),
        "Max": np.max(counts),
        "Mean": np.mean(counts),
        "Median (50th)": np.percentile(counts, 50),
        "75th Percentile": np.percentile(counts, 75),
        "80th Percentile": np.percentile(counts, 80),
        "90th Percentile": np.percentile(counts, 90),
        "95th Percentile": np.percentile(counts, 95),
        "99th Percentile": np.percentile(counts, 99)
    }

    print(f"\n--- Statistics for {title} ---")
    for k, v in stats.items():
        if k == "Total Slides":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.0f} patches")

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

    # Overlay percentile lines
    plt.axvline(stats["Median (50th)"], color='red', linestyle='dashed', linewidth=2,
                label=f'Median: {stats["Median (50th)"]:.0f}')
    plt.axvline(stats["80th Percentile"], color='orange', linestyle='dashed', linewidth=2,
                label=f'80th Pct: {stats["80th Percentile"]:.0f}')
    plt.axvline(stats["90th Percentile"], color='green', linestyle='dashed', linewidth=2,
                label=f'90th Pct: {stats["90th Percentile"]:.0f}')

    plt.title(title, fontsize=14)
    plt.xlabel("Number of Patches per Slide", fontsize=12)
    plt.ylabel("Frequency (Number of Slides)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Pointing to the V2 paths found in your main.py
    TRAIN_H5_DIR = r"W:\pathologie\bioinfo-archive\TridentPipelineOutput\CRC\CaltagironeUNIV2\20x_256px_0px_overlap\features_uni_v2"
    TEST_H5_DIR = r"W:\pathologie\bioinfo-archive\TridentPipelineOutput\CRC\HunCRCUNIV2\20x_256px_0px_overlap\features_uni_v2"

    # 1. Analyze Training Data
    print("Analyzing Training Cohort...")
    train_counts, _ = get_patch_counts(TRAIN_H5_DIR)
    analyze_and_plot(train_counts, title="Training Dataset (CAL) Patch Counts")

    # 2. Analyze Testing Data (Optional, uncomment to run)
    print("\nAnalyzing Testing Cohort...")
    test_counts, _ = get_patch_counts(TEST_H5_DIR)
    analyze_and_plot(test_counts, title="Testing Dataset (HunCRC) Patch Counts")