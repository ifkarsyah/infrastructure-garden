Here's a quick overview of some basic `NumPy` concepts and functions:

### 1. **Creating Arrays**

- **1D Array**: 
  ```python
  import numpy as np

  arr = np.array([1, 2, 3, 4, 5])
  ```

- **2D Array**:
  ```python
  arr = np.array([[1, 2, 3], [4, 5, 6]])
  ```

- **Zeros Array**:
  ```python
  zeros_arr = np.zeros((3, 3))
  ```

- **Ones Array**:
  ```python
  ones_arr = np.ones((2, 2))
  ```

- **Empty Array**:
  ```python
  empty_arr = np.empty((2, 3))
  ```

- **Array with a range of values**:
  ```python
  range_arr = np.arange(0, 10, 2)  # array([0, 2, 4, 6, 8])
  ```

- **Array with evenly spaced numbers**:
  ```python
  linspace_arr = np.linspace(0, 1, 5)  # array([0.  , 0.25, 0.5 , 0.75, 1.  ])
  ```

### 2. **Basic Array Operations**

- **Element-wise Operations**:
  ```python
  arr = np.array([1, 2, 3, 4])
  arr + 2  # array([3, 4, 5, 6])
  arr * 3  # array([3, 6, 9, 12])
  ```

- **Array Addition**:
  ```python
  arr1 = np.array([1, 2, 3])
  arr2 = np.array([4, 5, 6])
  sum_arr = arr1 + arr2  # array([5, 7, 9])
  ```

- **Dot Product**:
  ```python
  arr1 = np.array([1, 2, 3])
  arr2 = np.array([4, 5, 6])
  dot_product = np.dot(arr1, arr2)  # 32
  ```

### 3. **Indexing and Slicing**

- **Accessing elements**:
  ```python
  arr = np.array([1, 2, 3, 4, 5])
  print(arr[0])  # 1
  print(arr[-1])  # 5
  ```

- **Slicing**:
  ```python
  arr = np.array([1, 2, 3, 4, 5])
  print(arr[1:4])  # array([2, 3, 4])
  ```

- **2D Array Indexing**:
  ```python
  arr = np.array([[1, 2, 3], [4, 5, 6]])
  print(arr[0, 2])  # 3
  print(arr[:, 1])  # array([2, 5])
  ```

### 4. **Reshaping Arrays**

- **Reshape**:
  ```python
  arr = np.array([1, 2, 3, 4, 5, 6])
  reshaped_arr = arr.reshape((2, 3))
  ```

- **Flatten**:
  ```python
  arr = np.array([[1, 2, 3], [4, 5, 6]])
  flat_arr = arr.flatten()
  ```

### 5. **Array Functions**

- **Sum**:
  ```python
  arr = np.array([1, 2, 3, 4])
  sum_arr = np.sum(arr)  # 10
  ```

- **Mean**:
  ```python
  arr = np.array([1, 2, 3, 4])
  mean_arr = np.mean(arr)  # 2.5
  ```

- **Standard Deviation**:
  ```python
  arr = np.array([1, 2, 3, 4])
  std_dev = np.std(arr)  # 1.118033988749895
  ```

- **Transpose**:
  ```python
  arr = np.array([[1, 2, 3], [4, 5, 6]])
  transposed_arr = np.transpose(arr)  # array([[1, 4], [2, 5], [3, 6]])
  ```

### 6. **Random Numbers**

- **Random Array**:
  ```python
  random_arr = np.random.rand(3, 3)  # 3x3 array with random values between 0 and 1
  ```

- **Random Integers**:
  ```python
  random_int_arr = np.random.randint(0, 10, (2, 2))  # 2x2 array with random integers between 0 and 9
  ```

This should give you a solid starting point with `NumPy` basics!