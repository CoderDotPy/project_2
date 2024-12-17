
# Data Analysis Report

## Data Summary
Here is a quick summary of the dataset:

|       |       year |   Life Ladder |   Log GDP per capita |   Social support |   Healthy life expectancy at birth |   Freedom to make life choices |     Generosity |   Perceptions of corruption |   Positive affect |   Negative affect |
|:------|-----------:|--------------:|---------------------:|-----------------:|-----------------------------------:|-------------------------------:|---------------:|----------------------------:|------------------:|------------------:|
| count | 2363       |    2363       |           2363       |      2363        |                         2363       |                    2363        | 2363           |                 2363        |       2363        |      2363         |
| mean  | 2014.76    |       5.48357 |              9.39967 |         0.809369 |                           63.4018  |                       0.750282 |    9.77213e-05 |                    0.743971 |          0.651882 |         0.273151  |
| std   |    5.05944 |       1.12552 |              1.14522 |         0.120878 |                            6.75077 |                       0.138291 |    0.158596    |                    0.179907 |          0.105699 |         0.0868355 |
| min   | 2005       |       1.281   |              5.527   |         0.228    |                            6.72    |                       0.228    |   -0.34        |                    0.035    |          0.179    |         0.083     |
| 25%   | 2011       |       4.647   |              8.52    |         0.744    |                           59.545   |                       0.662    |   -0.108       |                    0.696    |          0.573    |         0.209     |
| 50%   | 2015       |       5.449   |              9.492   |         0.834    |                           64.9     |                       0.769    |   -0.015       |                    0.79     |          0.662    |         0.263     |
| 75%   | 2019       |       6.3235  |             10.382   |         0.904    |                           68.4     |                       0.861    |    0.088       |                    0.864    |          0.7365   |         0.326     |
| max   | 2023       |       8.019   |             11.676   |         0.987    |                           74.6     |                       0.985    |    0.7         |                    0.983    |          0.884    |         0.705     |

## Missing Values
We found the following missing values across columns:
|                                  |   0 |
|:---------------------------------|----:|
| year                             |   0 |
| Life Ladder                      |   0 |
| Log GDP per capita               |  28 |
| Social support                   |  13 |
| Healthy life expectancy at birth |  63 |
| Freedom to make life choices     |  36 |
| Generosity                       |  81 |
| Perceptions of corruption        | 125 |
| Positive affect                  |  24 |
| Negative affect                  |  16 |

## Correlation Matrix
The correlation matrix highlights relationships between variables. The following chart shows these correlations.

![Correlation Matrix](https://github.com/CoderDotPy/project_2/blob/happiness/correlation_matrix.png?raw=true)

## Outliers Detection
The outliers found in the dataset are shown in the chart below.

![Outliers](https://github.com/CoderDotPy/project_2/blob/happiness/outliers.png?raw=true)

## Clustering
Based on KMeans clustering, we identified 3 clusters in the data. These clusters represent groupings with potential significance.

