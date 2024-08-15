---
title: "Book Resume: Hands-on Machine Learning with Scikit-Learn, Keras, and Tensorflow"
tags:
  - "#book-resume"
---
This book is a comprehensive guide for learning machine learning, covering both fundamental concepts and practical applications. It primarily focuses on Python libraries such as [[sklearn]], [[Keras]], and [[TensorFlow]], and it’s structured to help readers develop a deep understanding of both the theoretical and practical aspects of machine learning.

The book is often divided into two main parts:
1. **Fundamentals of Machine Learning:** Covers the basics of machine learning, including data preprocessing, linear models, decision trees, ensemble methods, and more. It primarily uses Scikit-Learn for practical implementations.
2. **Deep Learning and Neural Networks:** Delves into deep learning concepts, starting from simple neural networks to more complex architectures like convolutional and recurrent neural networks. This part uses TensorFlow and Keras for implementation.
# Part I: Fundamental of Machine Learning
## Chapter 1: The Machine Learning Landscape
Introduces the basics of machine learning, including different types of learning (supervised, unsupervised, reinforcement) and typical machine learning tasks. It also gives an overview of the major challenges in machine learning.
### What is Machine Learning?
Machine Learning is the science (and art) of programming computers so they can learn from data.

### Why Use Machine Learning
#### Traditional Approach
![[1.1-traditional-approach.png]]
In a traditional, rule-based spam filter, developers manually define rules that the system uses to classify emails as spam or not spam. Here's how it might work:
1. **Rule Definition**: Developers create a set of rules based on heuristics, common patterns, and known spam indicators. For example:
	- If an email contains certain keywords like "free money," "win a prize," or "urgent," mark it as spam.
	- If the email is from a known spam domain, mark it as spam.
	- If the email has more than 50 recipients, mark it as spam.
	- If the email has suspicious attachments or links, mark it as spam.
2. **Hard-Coded Rules**: These rules are hard-coded into the system. The spam filter scans each incoming email and applies the rules to decide whether to classify the email as spam or not.
3. **Challenges**:
	- **Maintenance**: Developers need to continuously update the rules as spammers change their tactics.
	- **False Positives/Negatives**: Legitimate emails might be wrongly classified as spam (false positives), or spam emails might not be caught (false negatives).
	- **Inflexibility**: The system may fail to adapt to new, previously unseen spam techniques.
	- **Scalability**: As the volume of emails increases, adding and maintaining rules becomes more complex and less efficient.

#### ML Approach
![[1.2-ml-approach.png]]
In a machine learning-based spam filter, the system learns from a large dataset of labeled emails (spam and not spam) and automatically creates a model that can classify new emails. Here's how it might work:
1. **Data Collection**: A large dataset of emails, labeled as "spam" or "not spam," is collected. This labeled data is used to train the model.
2. **Feature Extraction**: The system analyzes the content of the emails and extracts features that might be useful for classification. For example:
	- Word frequency (e.g., how often certain words like "free," "win," "buy now" appear).
	- Presence of certain phrases or patterns.
	- Metadata like the sender’s email address, the number of recipients, and the presence of links or attachments.
3. **Model Training**: The extracted features and their corresponding labels (spam or not spam) are fed into a machine learning algorithm. Common algorithms for this task include:
	- **Naive Bayes**: Calculates the probability that an email is spam based on the occurrence of certain features.
	- **Logistic Regression**: Models the relationship between features and the probability that an email is spam.
	- **Support Vector Machines (SVMs)**: Finds the optimal boundary that separates spam from non-spam emails.
	- **Neural Networks**: Can model complex patterns and interactions between features.
4. **Prediction**: Once the model is trained, it can classify new, unseen emails. When a new email arrives, the model extracts features from it and predicts whether it is spam or not.
5. **Advantages**:
	- **Adaptability**: The model can learn and adapt to new spam techniques automatically, without requiring manual rule updates.
	- **Scalability**: The system can handle a large volume of emails and complex patterns more efficiently than a rule-based system.

### Types of Machine Learning Systems

There are so many different types of Machine Learning systems that it is useful to classify them in broad categories based on: 
- Whether or not they are trained with human supervision (supervised, unsupervised, semisupervised, and Reinforcement Learning)
- Whether or not they can learn incrementally on the fly (online versus batch learning) 
- Whether they work by simply comparing new data points to known data points, or instead detect patterns in the training data and build a predictive model, much like scientists do (instance-based versus model-based learning)

**These criteria are not exclusive**; you can combine them in any way you like. For example, a state-of-the-art spam filter may learn on the fly using a deep neural network model trained using examples of spam and ham; this makes it an online, modelbased, supervised learning system. 

Let’s look at each of these criteria a bit more closely
#### Supervised vs. Unsupervised Learning
- **Supervised Learning:** The model is trained on labeled data, where the output (target) is known. Common algorithms include linear regression, logistic regression, support vector machines, and neural networks.
- **Unsupervised Learning:** The model is trained on unlabeled data, and it tries to find patterns or structure in the input data. Examples include clustering algorithms (e.g., K-Means) and dimensionality reduction techniques (e.g., PCA).
- **Semi-supervised Learning:** A mix of both supervised and unsupervised learning, where the model is trained on a small amount of labeled data and a large amount of unlabeled data.
- **Reinforcement Learning:** The model learns by interacting with an environment and receiving rewards or penalties based on its actions. This is commonly used in game-playing AI, robotics, etc.
#### Batch vs. Online Learning
- **Batch Learning:** The model is trained on the entire dataset at once and updated periodically.
- **Online Learning:** The model is trained incrementally by feeding it data instances sequentially, which is useful for streaming data or systems that need to adapt quickly to new data.
#### Instance-Based vs. Model-Based Learning
- **Instance-Based Learning:** The model memorizes the training examples and makes predictions based on similarity to new instances. Examples include k-nearest neighbors.
- **Model-Based Learning:** The model generalizes from the training data by learning a function or model that maps inputs to outputs. Examples include linear regression and decision trees.
### Main Challenges of Machine Learning

In short, since your main task is to select a learning algorithm and train it on some data, the two things that can go wrong are “bad algorithm” and “bad data.” Let’s start with examples of bad data.
#### Bad Data
- **Insufficient Quantity of Training Data:** Machine learning models require a significant amount of data to learn effectively, and gathering this data can be challenging.
- **Nonrepresentative Training Data:** If the training data isn’t representative of the real-world scenario, the model may perform poorly.
- **Poor-Quality Data:** Noisy or incomplete data can negatively impact model performance. The truth is, most data scientists spend a significant part of their time doing just that. For example:
	- If some instances are clearly outliers, it may help to simply discard them or try to fix the errors manually.
	- If some instances are missing a few features (e.g., 5% of your customers did not specify their age), you must decide whether you want to ignore this attribute alto‐ gether, ignore these instances, fill in the missing values (e.g., with the median age), or train one model with the feature and one model without it, and so on.
- **Irrelevant Features:** The model’s performance heavily depends on the features used for training. Feature engineering, which involves selecting and transforming relevant features, is crucial. This process involves:
	- **Feature selection**: selecting the most useful features to train on among existing features.
	- **Feature extraction**: combining existing features to produce a more useful one (as we saw earlier, **dimensionality reduction** algorithms can help). 
	- **Creating new** features by gathering new data
Now that we have looked at many examples of bad data, let’s look at a couple of exam‐ ples of bad algorithms.
#### Bad Algorithm
* **Overfitting:** The model learns the training data too well, including noise and outliers, which leads to poor generalization to new data.
* **Underfitting:** The model is too simple to capture the underlying patterns in the data, leading to poor performance on both training and test data.

There’s just one last important topic to cover: once you have trained a model, you don’t want to just “hope” it generalizes to new cases. You want to evaluate it, and finetune it if necessary. Let’s see how.

### Testing and Validating
- **Generalization:** The chapter introduces the concept of generalization, which refers to the model’s ability to perform well on unseen data.
- **Validation Set and Test Set:** The importance of splitting the data into training, validation, and test sets to evaluate the model’s performance is discussed. This helps in tuning hyperparameters and assessing how well the model will perform in production.
## Chapter 2: End-to-End Machine Learning Project

In this chapter, you will go through an example project end to end, pretending to be a recently hired data scientist in a real estate company. Here are the main steps you will go through:
1. Look at the big picture. 
2. Get the data. 
3. Discover and visualize the data to gain insights. 
4. Prepare the data for Machine Learning algorithms. 
5. Select a model and train it. 
6. Fine-tune your model. 
7. Present your solution. 
8. Launch, monitor, and maintain your system.
### Working with Real Data
When you are learning about Machine Learning it is best to actually experiment with real-world data, not just artificial datasets. Fortunately, there are thousands of open datasets to choose from, ranging across all sorts of domains. Here are a few places you can look to get data:
- Popular open data repositories:
	- https://archive.ics.uci.edu/
	- https://www.kaggle.com/datasets
	- https://registry.opendata.aws/
- Meta portals (they list open data repositories):
	- https://dataportals.org/
	- https://data.nasdaq.com/institutional-investors
- Other pages listing many popular open data repositories:
	- https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
	- https://www.reddit.com/r/datasets
In this chapter we chose the California Housing Prices dataset from the StatLib repository. This dataset was based on data from the 1990 California census. It is not exactly recent (you could still afford a nice house in the Bay Area at the time), but it has many qualities for learning, so we will pretend it is recent data. We also added a categorical attribute and removed a few features for teaching purposes.

### Look at the Big Picture
- **Overview:** The chapter starts by introducing a real-world dataset: the **California housing prices dataset**. This dataset includes various features like the median income, population, and housing prices for different districts in California.
- **Goal:** The main objective is to create a model that predicts the median house value in any district based on the available features.
#### Frame the Problem

**The first question to ask your boss is what exactly is the business objective**; building a model is probably not the end goal. How does the company expect to use and benefit from this model? 

This is important because it will determine how you frame the problem, what algorithms you will select, what performance measure you will use to evaluate your model, and how much effort you should spend tweaking it.

Your boss answers that your model’s output (a prediction of a district’s median housing price) will be fed to another Machine Learning system (see Figure 2-2), along with many other signals.. This downstream system will determine whether it is worth investing in a given area or not. Getting this right is critical, as it directly affects revenue.
![[2.2-ml-pipeline.png]]

**The next question to ask is what the current solution looks like (if any)**. It will often give you a reference performance, as well as insights on how to solve the problem. Your boss answers that the district housing prices are currently estimated manually by experts: a team gathers up-to-date information about a district, and when they cannot get the median housing price, they estimate it using complex rules.

This is costly and time-consuming, and their estimates are not great; in cases where they manage to find out the actual median housing price, they often realize that their estimates were off by more than 20%. This is why the company thinks that it would be useful to train a model to predict a district’s median housing price given other data about that district. The census data looks like a great dataset to exploit for this purpose, since it includes the median housing prices of thousands of districts, as well as other data.

Okay, with all this information you are now ready to start designing your system. First, you need to frame the problem: is it supervised, unsupervised, or Reinforcement Learning? Is it a classification task, a regression task, or something else? Should you use batch learning or online learning techniques? Before you read on, pause and try to answer these questions for yourself.

The task here is to predict a continuous value (house price), making it a **regression problem**.

Since the dataset includes **labels** (house prices), it’s a **supervised learning** task.

#### Select a Performance Measure

Your next step is to select a performance measure. A typical performance measure for regression problems is the Root Mean Square Error (RMSE). It gives an idea of how much error the system typically makes in its predictions, with a higher weight for large errors.
$$ \text{RMSE}_{X, h} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (h(x_i) - y_i)^2} $$Where: 
- $m$ is the total number of observations. 
- $h(x_i)$ is the predicted value for the $i$-th observation. 
- $y_i$ is the actual value for the $i$-th observation. 

Even though the RMSE is generally the preferred performance measure for regression tasks, in some contexts you may prefer to use another function. For example, suppose that there are many outlier districts. In that case, you may consider using the Mean Absolute Error (also called the Average Absolute Deviation; see Equation 2-2):
$$ \text{MAE}_{X, h} = \frac{1}{m} \sum_{i=1}^{m} |h(x_i) - y_i| $$ Where: 
- $m$ is the total number of observations.
- $h(x_i)$ is the predicted value for the $i$-th observation. 
- $y_i$ is the actual value for the $i$-th observation. 
 
#### Check the Assumptions

Lastly, it is good practice to list and verify the assumptions that were made so far (by you or others); this can catch serious issues early on. 

For example, the district prices that your system outputs are going to be fed into a downstream Machine Learning system, and we assume that these prices are going to be used as such. But what if the downstream system actually converts the prices into categories (e.g., “cheap,” “medium,” or “expensive”) and then uses those categories instead of the prices themselves? 

In this case, getting the price perfectly right is not important at all; your system just needs to get the category right. If that’s so, then the problem should have been framed as a classification task, not a regression task. 

You don’t want to find this out after working on a regression system for months. Fortunately, after talking with the team in charge of the downstream system, you are confident that they do indeed need the actual prices, not just categories. 

Great! You’re all set, the lights are green, and you can start coding now!

### Get the Data

It’s time to get your hands dirty. Don’t hesitate to pick up your laptop and walk through the following code examples in a Jupyter notebook. The full Jupyter notebook is available at https://github.com/ageron/handson-ml2.

- **Data Acquisition:**
    - The chapter discusses how to access and load the dataset into your workspace. This might involve downloading the data, loading it into memory using Python libraries like Pandas, and inspecting the structure.
    - **Automate Data Handling:** It’s suggested to write functions to automate the data handling process. This ensures that you can easily load and prepare the data whenever necessary, especially as the project evolves.
- **Exploring the Data:**
    - **Quick Overview:** The dataset is briefly explored using Pandas to get a sense of the data structure, including the shape, column names, and a few statistical summaries.
    - **Data Visualization:** The chapter emphasizes the importance of visualizing data to spot trends, correlations, or potential issues. Techniques like histograms and scatter plots are used to explore the relationships between features.
### Discover and Visualize the Data to Gain Insights
- **Understanding Distributions:**
    - The distribution of each feature is examined, often using histograms, to understand their spread, skewness, and potential outliers.
- **Geographical Data Visualization:**
    - Since the dataset includes geographical data (latitude and longitude), it’s helpful to visualize housing prices on a map. This can reveal spatial patterns and correlations.
- **Correlations:**
    - The chapter discusses how to compute and visualize the correlation matrix. Correlation coefficients help identify which features are strongly related to the target variable (house prices).
    - **Scatter Matrix:** For selected pairs of attributes, scatter plots are used to visualize their relationships and identify patterns that might be useful in prediction.
- **Feature Combinations:**
    - New features can sometimes be created by combining existing ones. The chapter illustrates this by creating a feature like "rooms per household" to capture a different aspect of the data.
### Prepare the Data for Machine Learning Algorithms
- **Data Cleaning:**
    - **Handling Missing Values:** The chapter explains various strategies for dealing with missing data, such as removing missing entries, imputing missing values, or using data imputation techniques.
- **Handling Text and Categorical Attributes:**
    - **Encoding Categorical Variables:** Categorical data needs to be converted into numerical format to be used by machine learning algorithms. Techniques like one-hot encoding are discussed.
- **Feature Scaling:**
    - **Normalization and Standardization:** The chapter stresses the importance of feature scaling. Algorithms like gradient descent perform better when features are scaled to similar ranges.
- **Transformation Pipelines:**
    - **Building Pipelines:** A pipeline is a sequence of data transformation steps applied to the data in a specific order. Scikit-Learn’s `Pipeline` class is introduced to streamline the data preparation process, making it easier to combine multiple transformation steps and apply them consistently across training and test datasets.
### Select and Train a Model

- **Choosing a Model:**
    
    - The chapter introduces several basic models, including **Linear Regression** and **Decision Trees**. These models are chosen to illustrate the variety of approaches you can take to solve the problem.
    - **Training the Model:** Each model is trained on the dataset using Scikit-Learn’s API.
- **Performance Evaluation:**
    
    - **Cross-Validation:** The importance of using cross-validation to evaluate the model’s performance is highlighted. Cross-validation splits the training data into multiple subsets, training and validating the model on each subset to get a more reliable estimate of its performance.
- **Overfitting and Underfitting:**
    
    - **Overfitting:** The concept of overfitting is introduced, where the model performs well on training data but poorly on unseen data. Techniques like limiting the depth of decision trees or using simpler models are discussed to prevent overfitting.
    - **Model Comparison:** The chapter suggests trying out multiple models (e.g., Random Forests) and comparing their performance using cross-validation scores.
### Fine-Tune the Model

- **Hyperparameter Tuning:**
    
    - **Grid Search:** The chapter explains how to fine-tune the model by searching for the best hyperparameters using techniques like grid search. This involves systematically exploring combinations of hyperparameters to find the one that yields the best performance.
    - **Randomized Search:** An alternative to grid search is randomized search, which samples a few random combinations of hyperparameters, often yielding good results with less computational cost.
- **Ensemble Methods:**
    
    - **Combining Models:** The concept of ensemble learning is introduced, where multiple models are combined to improve overall performance. Techniques like bagging, boosting, or stacking are briefly mentioned.
- **Model Evaluation:**
    
    - **Final Model Evaluation:** After fine-tuning, the final model is evaluated on the test set to assess its generalization performance. The importance of not touching the test set until the very end is emphasized to avoid data leakage.
### Present the Solution

- **Model Deployment:**
    
    - **Saving the Model:** The chapter discusses how to save the trained model so it can be reused or deployed in a production environment.
    - **Documentation:** It’s essential to document the entire process, including the data preparation steps, model selection criteria, and evaluation results. This helps communicate your findings and ensures reproducibility.
- **Insights and Recommendations:**
    
    - Beyond just delivering a model, the chapter encourages presenting insights gained during the project. For example, understanding the most influential features in predicting house prices can provide valuable business insights.

### Launch, Monitor, and Maintain the System

- **Deploying to Production:**
    
    - Once the model is deployed, it’s important to monitor its performance over time. Real-world data can drift, meaning the distribution of data might change, requiring the model to be retrained periodically.
- **Model Updates:**
    
    - The chapter touches on the need to update the model as new data becomes available or as the underlying data distribution changes. This ensures that the model remains accurate and useful over time.
## Chapter 3: Classification
Focuses on classification tasks and introduces key algorithms such as logistic regression, support vector machines, and more. The chapter also discusses performance measures and how to evaluate classifiers.
## Chapter 4: Training Models
Explains the process of training machine learning models, including gradient descent, batch gradient descent, stochastic gradient descent, and polynomial regression.
## Chapter 5: Support Vector Machines
Provides an in-depth look at Support Vector Machines (SVMs), including how they work, their hyperparameters, and how to use them for classification and regression.

## Chapter 6: Decision Trees

Discusses decision trees and their implementation. The chapter also covers the concepts of entropy and information gain, and introduces ensemble methods like random forests.


## Chapter 7: Ensemble Learning and Random Forests
Explores ensemble methods like bagging, boosting, and stacking. Random forests, an ensemble of decision trees, are discussed in detail.


## Chapter 8: Dimensionality Reduction
Introduces techniques for reducing the dimensionality of data, such as Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA). The chapter also covers manifold learning.


## Chapter 9: Unsupervised Learning Techniques
Focuses on unsupervised learning, including clustering algorithms like K-Means and hierarchical clustering. The chapter also discusses Gaussian mixture models and anomaly detection.


## Chapter 10: Introduction to Artificial Neural Networks with Keras
Begins the exploration of deep learning with an introduction to artificial neural networks (ANNs) and the Keras library. The chapter covers the basics of building, training, and evaluating neural networks.


# Part II: Deep Learning and Neural Network

## Chapter 11: Training Deep Neural Networks
Discusses techniques to train deep neural networks, including the vanishing/exploding gradient problem, the use of optimizers like Adam, and regularization methods like dropout.


## Chapter 12: Custom Models and Training with TensorFlow
Explains how to build custom models, layers, and loss functions in TensorFlow. The chapter also covers advanced techniques for training models, such as the use of callbacks.


## Chapter 13: Loading and Preprocessing Data with TensorFlow
Focuses on how to efficiently load and preprocess large datasets using TensorFlow’s data API. It covers topics like data augmentation and the use of TensorFlow Datasets.


## Chapter 14: Computer Vision using CNN
Introduces Convolutional Neural Networks (CNNs) and their applications in computer vision tasks like image classification. The chapter covers key concepts like convolutional layers, pooling layers, and the use of popular architectures like ResNet.



