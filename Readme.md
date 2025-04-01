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
## 10. Python Iterators, List Comprehensions, and Generators

---
### 1. Using Iterators in Python
### What are Iterators and Iterables?
In Python, an **iterable** is an object that can return its elements one at a time, such as lists, tuples, and strings. An **iterator** is an object that implements the `__iter__()` and `__next__()` methods.

Example:
```python
my_list = [1, 2, 3]
iterator = iter(my_list)  # Convert list to iterator
print(next(iterator))  # Output: 1
print(next(iterator))  # Output: 2
print(next(iterator))  # Output: 3
```

### Using Iterators in Loops
Python's `for` loop automatically handles iterators:
```python
for num in my_list:
    print(num)
```

### Using `enumerate()`
The `enumerate()` function allows you to iterate with an index:
```python
names = ['Alice', 'Bob', 'Charlie']
for index, name in enumerate(names):
    print(index, name)
```

### Using `zip()` to Combine Iterables
`zip()` pairs elements from multiple iterables:
```python
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
for num, letter in zip(list1, list2):
    print(num, letter)
```

### Unzipping with `*`
```python
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
numbers, letters = zip(*pairs)
print(numbers)  # Output: (1, 2, 3)
print(letters)  # Output: ('a', 'b', 'c')
```

### Processing Large Files with Iterators
Instead of loading an entire file into memory, use iterators:
```python
with open('large_file.txt') as file:
    for line in file:
        process(line)  # Handle each line one at a time
```

---
### 2. List Comprehensions and Generators
### List Comprehensions
List comprehensions simplify list creation:
```python
squares = [x**2 for x in range(10)]
print(squares)
```

### Conditional List Comprehensions
```python
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # Output: [0, 2, 4, 6, 8]
```

### Nested List Comprehensions
```python
matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print(matrix)
```

### Dictionary Comprehensions
```python
squared_dict = {x: x**2 for x in range(5)}
print(squared_dict)
```

### Introduction to Generators
A **generator** is a special type of iterator that generates values lazily, saving memory.

#### Generator Expression
```python
gen = (x**2 for x in range(5))
print(next(gen))  # Output: 0
print(next(gen))  # Output: 1
```

#### Generator Function
```python
def count_up_to(n):
    count = 1
    while count <= n:
        yield count
        count += 1

counter = count_up_to(5)
print(next(counter))  # Output: 1
```

---
### 3. Bringing It All Together
### Zipping Dictionaries
```python
dict1 = {'a': 1, 'b': 2}
dict2 = {'x': 10, 'y': 20}
for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
    print(k1, v1, k2, v2)
```

### Using List Comprehensions in Data Processing
```python
data = [5, 10, 15, 20]
processed = [x / 5 for x in data]
print(processed)
```

### Using Pandas to Read Large Data in Chunks
```python
import pandas as pd
chunk_size = 1000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    process(chunk)
```

### Writing an Iterator to Load Data in Chunks
```python
def chunk_reader(file, chunk_size=1024):
    with open(file) as f:
        while chunk := f.read(chunk_size):
            yield chunk
```

---

# DataScience-overview
