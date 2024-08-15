---
title: Numpy Quickstart
---
Here's a quick overview of some basic `pandas` concepts and functions:

### 1. **Creating Data Structures**

- **Series**: A one-dimensional labeled array.
  ```python
  import pandas as pd

  data = pd.Series([1, 2, 3, 4, 5])
  ```

- **DataFrame**: A two-dimensional labeled data structure, similar to a table in a database.
  ```python
  data = {
      'Name': ['Alice', 'Bob', 'Charlie'],
      'Age': [25, 30, 35],
      'City': ['New York', 'Los Angeles', 'Chicago']
  }

  df = pd.DataFrame(data)
  ```

### 2. **Viewing Data**

- **Head and Tail**:
  ```python
  df.head()  # View the first 5 rows
  df.tail()  # View the last 5 rows
  ```

- **Information**:
  ```python
  df.info()  # Summary of the DataFrame
  ```

- **Describe**:
  ```python
  df.describe()  # Statistical summary
  ```

### 3. **Indexing and Selecting Data**

- **Selecting Columns**:
  ```python
  df['Name']  # Select a single column
  df[['Name', 'Age']]  # Select multiple columns
  ```

- **Selecting Rows**:
  ```python
  df.iloc[0]  # Select the first row by index
  df.loc[0]  # Select the first row by label
  ```

- **Conditional Selection**:
  ```python
  df[df['Age'] > 30]  # Select rows where Age is greater than 30
  ```

### 4. **Modifying Data**

- **Adding a New Column**:
  ```python
  df['Salary'] = [50000, 60000, 70000]
  ```

- **Modifying Values**:
  ```python
  df.loc[0, 'Age'] = 26  # Modify a single value
  ```

- **Dropping Columns/Rows**:
  ```python
  df.drop('Salary', axis=1, inplace=True)  # Drop the Salary column
  df.drop(0, axis=0, inplace=True)  # Drop the first row
  ```

### 5. **Handling Missing Data**

- **Checking for Missing Data**:
  ```python
  df.isnull()  # Returns a DataFrame of the same shape with True where NaN, False otherwise
  df.isnull().sum()  # Count of missing values in each column
  ```

- **Filling Missing Data**:
  ```python
  df.fillna(0, inplace=True)  # Fill missing values with 0
  ```

- **Dropping Missing Data**:
  ```python
  df.dropna(inplace=True)  # Drop rows with missing values
  ```

### 6. **Grouping and Aggregation**

- **Group By**:
  ```python
  grouped = df.groupby('City').mean()  # Group by City and calculate the mean
  ```

- **Aggregation**:
  ```python
  agg = df.groupby('City').agg({'Age': 'mean', 'Salary': 'sum'})  # Aggregate multiple columns
  ```

### 7. **Merging and Joining DataFrames**

- **Merging**:
  ```python
  df1 = pd.DataFrame({
      'Name': ['Alice', 'Bob', 'Charlie'],
      'Age': [25, 30, 35]
  })

  df2 = pd.DataFrame({
      'Name': ['Alice', 'Bob', 'Charlie'],
      'Salary': [50000, 60000, 70000]
  })

  merged_df = pd.merge(df1, df2, on='Name')
  ```

- **Joining**:
  ```python
  df1 = df1.set_index('Name')
  df2 = df2.set_index('Name')

  joined_df = df1.join(df2)
  ```

### 8. **Reading and Writing Data**

- **CSV**:
  ```python
  df = pd.read_csv('data.csv')  # Read CSV file
  df.to_csv('output.csv', index=False)  # Write to CSV file
  ```

- **Excel**:
  ```python
  df = pd.read_excel('data.xlsx')  # Read Excel file
  df.to_excel('output.xlsx', index=False)  # Write to Excel file
  ```

- **JSON**:
  ```python
  df = pd.read_json('data.json')  # Read JSON file
  df.to_json('output.json')  # Write to JSON file
  ```

### 9. **Pivot Tables**

- **Pivot Table**:
  ```python
  df = pd.DataFrame({
      'Date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
      'City': ['New York', 'New York', 'Chicago', 'Chicago'],
      'Sales': [200, 150, 100, 175]
  })

  pivot = df.pivot_table(values='Sales', index='Date', columns='City', aggfunc='sum')
  ```

### 10. **Basic Plotting**

- **Plotting with Matplotlib**:
  ```python
  import matplotlib.pyplot as plt

  df['Age'].plot(kind='hist')
  plt.show()
  ```

These are some basic `pandas` functionalities to get you started!