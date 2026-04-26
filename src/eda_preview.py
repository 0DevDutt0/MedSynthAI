import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pathlib import Path
from loguru import logger

def main():
    """Run exploratory data analysis on the synthetic dataset."""
    logger.info("Starting exploratory data analysis...")

    # Load the dataset
    logger.info("Loading synthetic dataset...")
    df = pd.read_csv("data/raw_synthetic.csv.gz")

    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Log basic statistics
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Log missing value statistics
    missing_stats = df.isnull().sum()
    logger.info("Missing value counts:")
    for col, count in missing_stats.items():
        if count > 0:
            logger.info(f"  {col}: {count} ({count/len(df)*100:.2f}%)")

    # Generate missingno matrix plot
    logger.info("Generating missingno matrix plot...")
    plt.figure(figsize=(12, 8))
    msno.matrix(df, figsize=(12, 8))
    plt.title("Missing Data Matrix")
    plt.savefig("reports/missingno_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Generate missingno bar plot
    logger.info("Generating missingno bar plot...")
    plt.figure(figsize=(12, 6))
    msno.bar(df, figsize=(12, 6))
    plt.title("Missing Data Bar Chart")
    plt.savefig("reports/missingno_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Log some basic info about the dataset
    logger.info("Dataset information:")
    logger.info(df.info())

    # Log descriptive statistics
    logger.info("Descriptive statistics:")
    logger.info(df.describe())

    logger.info("Exploratory data analysis complete!")

if __name__ == "__main__":
    main()