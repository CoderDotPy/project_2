# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "json",
#   "openai",
#   "sklearn",
#   "requests",
# ]
# ///

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import openai
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set up OpenAI token
openai.api_key = os.environ["AIPROXY_TOKEN"]


def load_data(file_path):
    # Load the CSV file into a pandas DataFrame
    return pd.read_csv(file_path)


def analyze_data(df):
    # Basic summary statistics
    summary_stats = df.describe()

    # Count missing values
    missing_values = df.isnull().sum()

    # Correlation matrix
    correlation_matrix = df.corr()

    # Check for outliers (IQR method)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

    # KMeans clustering as an example of clustering
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))  # Use numeric columns
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(df_scaled)

    return {
        "summary_stats": summary_stats,
        "missing_values": missing_values,
        "correlation_matrix": correlation_matrix,
        "outliers": outliers,
        "clusters": clusters
    }


def create_visualizations(analysis_results, output_dir):
    # Save correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(analysis_results['correlation_matrix'], annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    correlation_image = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(correlation_image)
    plt.close()

    # Plot outliers
    plt.figure(figsize=(10, 8))
    analysis_results['outliers'].plot(kind='bar', color='red')
    plt.title("Outliers Detection")
    outliers_image = os.path.join(output_dir, "outliers.png")
    plt.savefig(outliers_image)
    plt.close()

    return [correlation_image, outliers_image]


def generate_narrative(analysis_results, image_paths):
    # Combine analysis and visualizations into a markdown-friendly format
    narrative = """
# Data Analysis Report

## Data Summary
Here is a quick summary of the dataset:

{summary_stats}

## Missing Values
We found the following missing values across columns:
{missing_values}

## Correlation Matrix
The correlation matrix highlights relationships between variables. The following chart shows these correlations.

![Correlation Matrix](./{correlation_image})

## Outliers Detection
The outliers found in the dataset are shown in the chart below.

![Outliers](./{outliers_image})

## Clustering
Based on KMeans clustering, we identified {n_clusters} clusters in the data. These clusters represent groupings with potential significance.

"""
    narrative = narrative.format(
        summary_stats=analysis_results["summary_stats"].to_markdown(),
        missing_values=analysis_results["missing_values"].to_markdown(),
        correlation_image=image_paths[0].split('/')[-1],
        outliers_image=image_paths[1].split('/')[-1],
        n_clusters=3
    )

    return narrative


def save_readme(narrative, output_dir):
    with open(os.path.join(output_dir, "README.md"), "w") as readme_file:
        readme_file.write(narrative)


def main(file_path):
    output_dir = os.getcwd()
    df = load_data(file_path)
    analysis_results = analyze_data(df)
    image_paths = create_visualizations(analysis_results, output_dir)
    narrative = generate_narrative(analysis_results, image_paths)
    save_readme(narrative, output_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
    else:
        main(sys.argv[1])
