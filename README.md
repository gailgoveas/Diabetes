## Classification using KNN and Naive Bayes

This repository provides code for performing classification tasks using the K-Nearest Neighbors (KNN) and Naive Bayes algorithms on the **Pima Indians Diabetes** dataset. The goal is to predict whether a patient has diabetes or not based on various medical features.


### Dataset
The dataset used in this project, "diabetes.csv", contains information about diabetes patients. It includes features such as glucose level, blood pressure, skin thickness, insulin level, BMI, age, and pregnancy information. The target variable, "Outcome", indicates whether a patient has diabetes (1) or not (0).


### Dependencies
The code requires the following dependencies:

- pandas: For data manipulation and analysis
- numpy: For numerical computations
- matplotlib: For data visualization
- scikit-learn: For implementing machine learning algorithms

You can install these dependencies using pip:

```python

pip install pandas numpy matplotlib scikit-learn

```
### Methodology
The code follows the following methodology:

1. **Importing Modules**: The necessary modules, such as pandas, numpy, time, matplotlib, and scikit-learn modules are imported.

2. **Loading the Data**: The dataset is loaded into a pandas DataFrame using the `pd.read_csv()` function.

3. **Preprocessing and Data Scaling**: The target variable is separated from the features, and the features are scaled using the standard scaler. The scaled features are then concatenated with the target variable.

4. **Splitting the Data**: The dataset is split into training and testing sets using the `train_test_split()` function from scikit-learn.

5. **K-Nearest Neighbors (KNN)**:
   - Determining the optimal K value: The KNN classifier is trained and evaluated for different values of K to determine the optimal K value using accuracy scores.
   - Plotting the accuracy: The accuracy scores for different K values are plotted to determine the best K value. 

6. **K-Fold Cross-Validation**: The KNN classifier is evaluated using k-fold cross-validation with 5 folds. The average cross-validation score and standard deviation are calculated.


8. **Naive Bayes (NB) Models**:
   The Gaussian NB, Multinomial NB, and Bernoulli NB classifiers are trained and evaluated using k-fold cross-validation. The average cross-validation score and standard deviation are calculated.

9. **Final Testing on Holdout Set**: The NB classifiers are evaluated using the testing data (holdout set). The accuracy scores are calculated and printed.

10. **Leave-One-Out Cross-Validation**: The KNN and Gaussian NB classifiers are trained and evaluated using the leave-one-out cross-validation method. The mean accuracy and standard deviation are calculated, and the runtimes are measured.

11. **Model Selection and Conclusion**: Based on the evaluation results, the Gaussian NB model is selected as the best model for the given dataset. 

The code combines data preprocessing, model training and evaluation, cross-validation techniques, and result analysis to perform classification tasks using KNN and NB models.

### Model Comparison and Selection

The performance of the KNN and Gaussian NB models was evaluated using both k-fold cross-validation and leave-one-out classification methods. The results are as follows:

- **KNN Model**:
  - K-Fold Cross-Validation: Accuracy of 75.53%
  - Leave-One-Out Classification: Accuracy of 74.73%
  - Confusion Matrix Evaluation: Accuracy of 76.62%

- **Gaussian NB Model**:
  - K-Fold Cross-Validation: Accuracy of 76.62%
  - Leave-One-Out Classification: Accuracy of 75.39%
  - Confusion Matrix Evaluation: Accuracy of 76.62%

Based on these results, it can be observed that both models performed similarly in terms of accuracy when evaluated using k-fold cross-validation and confusion matrix. However, when using the leave-one-out classification method, the Gaussian NB model outperformed the KNN model in terms of accuracy (75.39% vs. 74.73%). Additionally, the Gaussian NB model demonstrated a lower runtime.

Considering these factors, the Gaussian NB model is selected as the preferred model for the given dataset.

Feel free to refer to the code and results in this repository to understand the performance of both models and make informed decisions for your own classification tasks.


### License
This project is licensed under the MIT License.
