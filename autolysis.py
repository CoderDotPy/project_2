# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "tabulate",
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
from sklearn.impute import SimpleImputer

# Ensure the AI Proxy token is set as an environment variable
openai.api_key = os.getenv("AIPROXY_TOKEN")

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')  # Try 'latin1' or 'ISO-8859-1' encoding
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_data(df):
    """Perform basic data analysis on the dataset."""
    numeric_df = df.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='mean')  # Handle missing values by replacing with column mean
    numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

    # Summary statistics and missing values count
    summary_stats = numeric_df_imputed.describe()
    missing_values = numeric_df.isnull().sum()

    # Correlation matrix
    correlation_matrix = numeric_df_imputed.corr()

    # Outlier detection using the IQR method
    Q1 = numeric_df_imputed.quantile(0.25)
    Q3 = numeric_df_imputed.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df_imputed < (Q1 - 1.5 * IQR)) | (numeric_df_imputed > (Q3 + 1.5 * IQR))).sum()

    # KMeans clustering
    df_scaled = StandardScaler().fit_transform(numeric_df_imputed)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

    return {
        "summary_stats": summary_stats,
        "missing_values": missing_values,
        "correlation_matrix": correlation_matrix,
        "outliers": outliers,
        "clusters": clusters
    }

def create_visualizations(analysis_results, output_dir):
    """Generate visualizations and save them as PNG images."""
    # Correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(analysis_results['correlation_matrix'], annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    correlation_image = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(correlation_image)
    plt.close()

    # Outliers bar plot
    plt.figure(figsize=(10, 8))
    analysis_results['outliers'].plot(kind='bar', color='red')
    plt.title("Outliers Detection")
    outliers_image = os.path.join(output_dir, "outliers.png")
    plt.savefig(outliers_image)
    plt.close()

    return [correlation_image, outliers_image]

def generate_narrative(analysis_results, image_paths):
    """Generate a narrative for the analysis using the LLM."""
    # Format the input data for the LLM
    analysis_data = {
        "summary_stats": analysis_results["summary_stats"].to_dict(),
        "missing_values": analysis_results["missing_values"].to_dict(),
        "correlation_matrix": analysis_results["correlation_matrix"].to_dict(),
        "outliers": analysis_results["outliers"].to_dict(),
        "n_clusters": 3
    }

    prompt = f"""
    I have a dataset with the following analysis:

    - Summary Stats: {analysis_data["summary_stats"]}
    - Missing Values: {analysis_data["missing_values"]}
    - Correlation Matrix: {analysis_data["correlation_matrix"]}
    - Outliers: {analysis_data["outliers"]}
    
    Based on this, please provide a narrative describing the data, the insights uncovered, and the implications of the findings. Also, summarize any clusters identified in the data.
    """

    # Query the LLM for a detailed narrative
    response = openai.Completion.create(
        model="gpt-4o-mini",  # Using GPT-4o-Mini as per the project requirement
        prompt=prompt,
        max_tokens=500
    )
    
    narrative = response.choices[0].text.strip()

    # Format the Markdown for the README
    narrative_markdown = f"""
# Data Analysis Report

## Data Summary
Here is a quick summary of the dataset:

{analysis_results['summary_stats'].to_markdown()}

## Missing Values
We found the following missing values across columns:
{analysis_results['missing_values'].to_markdown()}

## Correlation Matrix
The correlation matrix highlights relationships between variables. The following chart shows these correlations.

![Correlation Matrix](./{image_paths[0].split('/')[-1]})

## Outliers Detection
The outliers found in the dataset are shown in the chart below.

![Outliers](./{image_paths[1].split('/')[-1]})

## Clustering
Based on KMeans clustering, we identified {analysis_data["n_clusters"]} clusters in the data. These clusters represent groupings with potential significance.

## Insights and Recommendations
{narrative}
    """
    return narrative_markdown

def save_readme(narrative, output_dir):
    """Save the generated narrative as README.md."""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as readme_file:
        readme_file.write(narrative)

def main(file_path):
    """Main function to run the analysis."""
    output_dir = os.getcwd()
    df = load_data(file_path)

    if df is None:
        return

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
