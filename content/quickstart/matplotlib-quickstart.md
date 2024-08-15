Here's an overview of some basic `matplotlib` concepts and functions:

### 1. **Importing Matplotlib**

```python
import matplotlib.pyplot as plt
```

### 2. **Basic Plotting**

- **Line Plot**:
  ```python
  x = [1, 2, 3, 4, 5]
  y = [2, 3, 5, 7, 11]

  plt.plot(x, y)
  plt.title('Basic Line Plot')
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.show()
  ```

- **Scatter Plot**:
  ```python
  plt.scatter(x, y)
  plt.title('Basic Scatter Plot')
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.show()
  ```

* Difference:
	* **Line Plot (`plot`)** is best when you want to emphasize the trend or relationship over a continuous range, with a line connecting data points.
	* **Scatter Plot (`scatter`)** is best for showing individual data points and the relationship between two variables without any connecting line.
### 3. **Customizing Plots**

- **Adding Titles and Labels**:
  ```python
  plt.plot(x, y)
  plt.title('Custom Title')
  plt.xlabel('Custom X Label')
  plt.ylabel('Custom Y Label')
  ```

- **Changing Line Styles and Colors**:
  ```python
  plt.plot(x, y, color='green', linestyle='--', marker='o')
  plt.show()
  ```

- **Adding a Legend**:
  ```python
  plt.plot(x, y, label='Prime Numbers')
  plt.legend()
  plt.show()
  ```

### 4. **Multiple Plots**

- **Plotting Multiple Lines**:
  ```python
  y2 = [1, 4, 9, 16, 25]
  plt.plot(x, y, label='Primes')
  plt.plot(x, y2, label='Squares')
  plt.legend()
  plt.show()
  ```

- **Subplots**:
  ```python
  plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
  plt.plot(x, y)
  plt.title('Prime Numbers')

  plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
  plt.plot(x, y2)
  plt.title('Square Numbers')

  plt.show()
  ```

### 5. **Bar Plots**

- **Vertical Bar Plot**:
  ```python
  categories = ['A', 'B', 'C', 'D']
  values = [5, 7, 3, 4]

  plt.bar(categories, values)
  plt.title('Basic Bar Plot')
  plt.show()
  ```

- **Horizontal Bar Plot**:
  ```python
  plt.barh(categories, values)
  plt.title('Basic Horizontal Bar Plot')
  plt.show()
  ```

### 6. **Histograms**

- **Basic Histogram**:
  ```python
  data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

  plt.hist(data, bins=5)
  plt.title('Basic Histogram')
  plt.show()
  ```

### 7. **Pie Charts**

- **Basic Pie Chart**:
  ```python
  sizes = [15, 30, 45, 10]
  labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']

  plt.pie(sizes, labels=labels, autopct='%1.1f%%')
  plt.title('Basic Pie Chart')
  plt.show()
  ```

### 8. **Box Plots**

- **Basic Box Plot**:
  ```python
  data = [np.random.normal(0, std, 100) for std in range(1, 4)]

  plt.boxplot(data, vert=True, patch_artist=True)
  plt.title('Basic Box Plot')
  plt.show()
  ```

### 9. **Saving Plots**

- **Save a Plot**:
  ```python
  plt.plot(x, y)
  plt.title('Saved Plot')
  plt.savefig('plot.png')
  ```

### 10. **Basic Plot Customization**

- **Grid**:
  ```python
  plt.plot(x, y)
  plt.grid(True)
  plt.show()
  ```

- **Figure Size**:
  ```python
  plt.figure(figsize=(8, 6))
  plt.plot(x, y)
  plt.show()
  ```

- **Changing Ticks**:
  ```python
  plt.plot(x, y)
  plt.xticks([1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E'])
  plt.yticks([2, 3, 5, 7, 11], ['Two', 'Three', 'Five', 'Seven', 'Eleven'])
  plt.show()
  ```

These basics will get you started with `matplotlib` and help you create a variety of visualizations!