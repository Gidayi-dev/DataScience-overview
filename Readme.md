# Data Science with Python

This repository provides an overview of essential data science skills using Python, covering data manipulation, merging, statistical analysis, visualization, and function writing.

## Table of Contents
- **Transforming DataFrames**
- **Aggregating DataFrames**
- **Slicing and Indexing DataFrames**
- **Creating and Visualizing DataFrames**
- **Merging and Joining Data**
- **Statistical Analysis**
- **Data Visualization with Matplotlib**
- **Data Visualization with Seaborn**
- **Python Functions and Iterators**

---

## 1. Transforming DataFrames
Transforming data in pandas involves modifying columns, renaming, and applying operations.

```python
import pandas as pd

df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]})
df['Age in Months'] = df['Age'] * 12
print(df)
```

---

## 2. Aggregating DataFrames
Aggregation helps summarize data using functions like sum, mean, and count.

```python
print(df.groupby('Name').mean())
```

---

## 3. Slicing and Indexing DataFrames
Selecting specific rows and columns using slicing and indexing.

```python
print(df.loc[0])  # Access by label
print(df.iloc[1])  # Access by position
```

---

## 4. Creating and Visualizing DataFrames
You can generate data programmatically and visualize it with pandas and Matplotlib.

```python
import matplotlib.pyplot as plt

df['Age'].plot(kind='bar')
plt.show()
```

---

## 5. Merging and Joining Data
Joining multiple tables using different merge types.

```python
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [2, 3], 'Score': [88, 92]})
merged_df = pd.merge(df1, df2, on='ID', how='inner')
print(merged_df)
```

---

## 6. Statistical Analysis
Performing summary statistics and probability distributions.

```python
import numpy as np

data = np.random.normal(loc=50, scale=15, size=100)
print(f'Mean: {np.mean(data)}, Std Dev: {np.std(data)}')
```

---

## 7. Data Visualization with Matplotlib
Creating basic plots.

```python
time = np.arange(0, 10, 0.1)
values = np.sin(time)
plt.plot(time, values)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Sine Wave')
plt.show()
```

---

## 8. Data Visualization with Seaborn
Enhancing plots with Seaborn.

```python
import seaborn as sns

df = sns.load_dataset('tips')
sns.scatterplot(x='total_bill', y='tip', data=df)
plt.show()
```

---

## 9. Python Functions and Iterators
Writing functions and using iterators efficiently.

```python
def square(n):
    return n ** 2

print(square(4))
```

---



# DataScience-overview
