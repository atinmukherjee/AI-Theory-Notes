# Support Vector Machine (SVM)
## 1. Motivation

Many real-world classification problems require a decision boundary that not only separates different classes but also provides a **large margin of separation** between them.

Earlier methods such as **Logistic Regression** and **Linear Discriminant Analysis (LDA)** can produce effective linear decision boundaries. However, they may not explicitly optimize the **margin between classes**, and linear models cannot directly handle cases where the classes are **not linearly separable**.

SVM addresses these challenges by finding an **optimal separating hyperplane** that maximizes the margin between classes.

### Key Motivations

* **Maximum-Margin Separation:** SVM explicitly searches for a decision boundary that maximizes the margin between classes.
* **Better Generalization:** A larger margin can lead to better generalization on unseen data.
* **Nonlinear Classification:** Using the **kernel trick**, SVM can create nonlinear decision boundaries.
* **Handling Outliers:** Soft-margin SVM allows controlled margin violations instead of forcing every training instance to be perfectly separated.
* **High-Dimensional Data:** SVM can work effectively in high-dimensional feature spaces, particularly when the number of features is large relative to the number of samples.

### Core Idea

SVM tries to find the **best possible separating hyperplane** by balancing two objectives:

$$
\boxed{\text{Maximize the Margin} \quad + \quad \text{Minimize Margin Violations}}
$$

This makes SVM a powerful approach for classification problems where **separation, generalization, and nonlinear decision boundaries** are important.
## 2. Hyperplane and Geometry
## 3. Maximal Margin Classifier
## 4. Support Vector Classifier (Soft Margin)
**Support Vector Machine (SVM)** is a powerful and versatile supervised machine learning algorithm primarily used for **linear and non-linear classification**, although it can also be extended to **regression** problems. The fundamental idea behind SVM is to find an optimal **hyperplane** that separates data points belonging to different classes. A **hyperplane** is a mathematical decision boundary that divides a feature space into two regions. In a two-dimensional space, the hyperplane is a **line**; in three dimensions, it is a **plane**; and in higher-dimensional spaces, it is referred to as a **hyperplane**. For a binary classification problem, SVM searches for the hyperplane that not only separates the different classes but also **maximizes the margin**, i.e., the margin is the distance between the closest support vectors of the two classes. A larger margin indicates a greater degree of confidence in the classification. The margin is a measure of how well-separated the classes are in feature space. SVMs are designed to find the hyperplane that maximizes this margin. Therefore, sometime SVM also called as **Maximun Margin Classifier**. 

Therefore, we can define a Support Vector Machine as:

> **A Support Vector Machine (SVM) is a supervised machine learning algorithm that classifies data by finding an optimal decision boundary (hyperplane) that maximizes the margin between different classes in an N-dimensional feature space.**

The data points closest to the optimal hyperplane are called **support vectors**. These points play a critical role in determining the position and orientation of the decision boundary. <p align="center">
  <img src="image/svm.png"
       alt="Optimal Hyperplane, Margin and Support Vectors"
       width="350"
       height = "350">
</p>

**Fig. 1.** Illustration of the optimal hyperplane, margin, and support vectors in SVM.

> [!TIP]
> 💡 **SVM is sensitive to feature scales** because it determines the decision boundary by maximizing the geometric margin. If one feature has a much larger numerical range than another, it can disproportionately influence the distance calculations and consequently affect the orientation of the optimal hyperplane. Therefore, feature scaling, such as **StandardScaler**, is generally recommended before training an SVM.
## 5. Kernel Trick and Non-linear SVM
## 6. Mathematical Formulation
## 7. Geometric Interpretation
## 8. SVM Parameters (C, gamma, kernel)
## 9. Practical Implementation (sklearn)
## 9. Practical Implementation (scikit-learn)

The theoretical concepts of SVM can be implemented using **scikit-learn**. A practical SVM workflow should include **data splitting, feature scaling, kernel selection, hyperparameter tuning, cross-validation, and final evaluation**.

### 9.1 SVM Workflow

A complete SVM workflow is:

```text
Dataset
   ↓
Train / Test Split
   ↓
Pipeline
   ├── Feature Scaling
   └── SVM
   ↓
Hyperparameter Tuning
   ├── Kernel
   ├── C
   └── γ
   ↓
Cross-Validation
   ↓
Best Model
   ↓
Final Evaluation on Test Set
```

The important principle is:

$$
\boxed{
\text{Training Data}
\rightarrow
\text{Cross-Validation + Hyperparameter Tuning}
\rightarrow
\text{Best Model}
\rightarrow
\text{Unseen Test Data}
}
$$

The test set should remain untouched until the final evaluation.

---

### 9.2 Import Required Libraries

```python
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
```

---

### 9.3 Load the Dataset

For demonstration, we use the **Iris dataset**.

```python
iris = load_iris()

X = iris.data
y = iris.target

print("Feature shape:", X.shape)
print("Classes:", np.unique(y))
```

The Iris dataset contains:

* 150 samples
* 4 numerical features
* 3 classes

---

### 9.4 Split the Dataset

First, divide the dataset into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Here:

* `test_size=0.2` → 20% of the data is reserved for testing
* `random_state=42` → makes the split reproducible
* `stratify=y` → maintains the class distribution

---

### 9.5 Build an SVM Pipeline

Feature scaling is important because SVM is sensitive to feature magnitude.

Instead of scaling the data separately, combine preprocessing and SVM into a **Pipeline**.

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])
```

The pipeline performs:

$$
\text{Input Data}
\rightarrow
\text{StandardScaler}
\rightarrow
\text{SVM}
$$

This also helps prevent **data leakage during cross-validation**.

---

### 9.6 Define the Hyperparameter Search Space

The most important SVM hyperparameters include:

* **Kernel**
* **\(C\)**
* **\(\gamma\)** for nonlinear kernels such as RBF

```python
param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1]
}
```

#### Meaning of \(C\)

\(C\) controls the penalty for margin violations.

* Small \(C\) → wider margin, more violations allowed
* Large \(C\) → stronger penalty for violations

#### Meaning of \(\gamma\)

For the RBF kernel:

$$
K(x_i,x_j)
=
\exp(-\gamma\|x_i-x_j\|^2)
$$

* Small \(\gamma\) → smoother decision boundary
* Large \(\gamma\) → more localized influence and potentially more complex boundary

---

### 9.7 Hyperparameter Tuning with GridSearchCV

Use `GridSearchCV` to find a suitable combination of hyperparameters.

```python
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
```

Here:

* `cv=5` → 5-fold cross-validation
* `scoring="accuracy"` → accuracy is used to compare models
* `n_jobs=-1` → uses all available CPU cores

Conceptually:

```text
Training Data
      ↓
   Split into
    5 folds
      ↓
Train on 4 folds
Validate on 1 fold
      ↓
Repeat 5 times
      ↓
Compare Hyperparameters
      ↓
Select Best Parameters
```

---

### 9.8 Obtain the Best Hyperparameters

```python
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Score:")
print(grid_search.best_score_)
```

For example, the output might look like:

```text
Best Parameters:
{
    'svm__C': 10,
    'svm__gamma': 'scale',
    'svm__kernel': 'rbf'
}
```

The exact values may vary depending on the dataset and search space.

---

### 9.9 Obtain the Best Model

`GridSearchCV` automatically identifies the best-performing configuration and refits the model on the complete training dataset.

```python
best_model = grid_search.best_estimator_
```

The selected model can now be used for prediction.

---

### 9.10 Evaluate on the Test Set

The test set has not been used during hyperparameter selection.

Therefore, it can now be used for the final evaluation.

```python
y_pred = best_model.predict(X_test)
```

#### Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
```

#### Classification Report

```python
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)
```

The classification report provides:

* Precision
* Recall
* F1-score
* Support

#### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```

---

### 9.11 Inspect Support Vectors

One of the important characteristics of SVM is the concept of **support vectors**.

For an `SVC` model, they can be accessed using:

```python
svm_model = best_model.named_steps["svm"]

print("Number of support vectors:")
print(svm_model.n_support_)

print("\nTotal support vectors:")
print(len(svm_model.support_))
```

The support-vector samples can be accessed using:

```python
support_vectors = svm_model.support_vectors_

print(support_vectors)
```

These samples play a central role in determining the SVM decision boundary.

---

## 9.12 Complete Implementation

The complete workflow can be written compactly as follows:

```python
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

# --------------------------------------------------
# 2. Train / Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. Build Pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

# --------------------------------------------------
# 4. Define Hyperparameter Search Space
# --------------------------------------------------

param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1]
}

# --------------------------------------------------
# 5. Hyperparameter Tuning + Cross-Validation
# --------------------------------------------------

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# --------------------------------------------------
# 6. Best Parameters
# --------------------------------------------------

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest CV Accuracy:")
print(grid_search.best_score_)

# --------------------------------------------------
# 7. Best Model
# --------------------------------------------------

best_model = grid_search.best_estimator_

# --------------------------------------------------
# 8. Test Set Prediction
# --------------------------------------------------

y_pred = best_model.predict(X_test)

# --------------------------------------------------
# 9. Final Evaluation
# --------------------------------------------------

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 10. Support Vectors
# --------------------------------------------------

svm_model = best_model.named_steps["svm"]

print("\nNumber of Support Vectors:")
print(svm_model.n_support_)

print("\nTotal Support Vectors:")
print(len(svm_model.support_))
```

---

## 9.13 Practical Workflow Summary

The complete SVM implementation can be summarized as:

$$
\boxed{
\text{Dataset}
\rightarrow
\text{Train/Test Split}
\rightarrow
\text{Scaling}
\rightarrow
\text{SVM Pipeline}
\rightarrow
\text{Hyperparameter Tuning}
\rightarrow
\text{Cross-Validation}
\rightarrow
\text{Best Model}
\rightarrow
\text{Test Evaluation}
}
$$

### Key Parameters

| Parameter      | Purpose                                                     |
| -------------- | ----------------------------------------------------------- |
| `kernel`       | Determines the type of decision boundary                    |
| `C`            | Controls the penalty for margin violations                  |
| `gamma`        | Controls the influence of samples in RBF/polynomial kernels |
| `cv`           | Number of cross-validation folds                            |
| `class_weight` | Helps handle class imbalance                                |

> **Important:** Hyperparameter tuning should be performed using the **training data and cross-validation**. The test set should be used only once the final model has been selected.

### Key Takeaway

A good SVM implementation is not simply:

```text
Train → Predict
```

It is:

```text
Split
  ↓
Scale
  ↓
Choose Kernel
  ↓
Tune Hyperparameters
  ↓
Cross-Validate
  ↓
Select Best Model
  ↓
Evaluate on Unseen Test Data
```

This workflow provides a more reliable estimate of how the trained SVM will perform on unseen data.

## 10. Examples (Iris, Digits)
## 11. Practical Tips

SVM can be a powerful classifier, but its performance depends strongly on **feature scaling, kernel selection, and hyperparameter tuning**.

### ✔ 1. Always Scale Numerical Features

SVM is sensitive to feature magnitude because it relies on **distances, dot products, and geometric margins**.

For example, if one feature ranges from \(0\)–\(1\) while another ranges from \(0\)–\(10,000\), the larger-scale feature can dominate the geometry of the feature space.

A common choice is **StandardScaler**:

$$
x' = \frac{x-\mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

> **Important:** Fit the scaler only on the training data to avoid **data leakage**.

---

### ✔ 2. Try RBF Kernel for Nonlinear Problems

The **Radial Basis Function (RBF)** kernel is often a strong starting point when the relationship between features and classes is nonlinear.

$$
K(x_i,x_j)=\exp(-\gamma\|x_i-x_j\|^2)
$$

In scikit-learn:

```python
from sklearn.svm import SVC

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)
```

However, RBF should not be considered universally optimal. For a linearly separable or approximately linear problem, a **linear kernel** may be simpler and faster.

---

### ✔ 3. Tune \(C\) Carefully

\(C\) controls the trade-off between:

$$
\boxed{\text{Large Margin} \quad \leftrightarrow \quad \text{Training Errors}}
$$

* **Small \(C\)** → wider margin, more violations allowed
* **Large \(C\)** → narrower margin, stronger penalty for violations

A useful starting search might be:

```python
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}
```

---

### ✔ 4. Tune \(\gamma\) for RBF SVM

For the RBF kernel, \(\gamma\) controls how strongly each training point influences its surrounding region.

$$
K(x_i,x_j)
=
e^{-\gamma\|x_i-x_j\|^2}
$$

* **Small \(\gamma\)** → broader influence → smoother decision boundary
* **Large \(\gamma\)** → localized influence → more complex decision boundary

Therefore:

$$
\boxed{\gamma \uparrow \Rightarrow \text{more complex boundary}}
$$

$$
\boxed{\gamma \downarrow \Rightarrow \text{smoother boundary}}
$$

A common search:

```python
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1]
}
```

---

### ✔ 5. Use GridSearchCV or RandomizedSearchCV

Instead of manually guessing \(C\) and \(\gamma\), use cross-validation.

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    SVC(kernel="rbf"),
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)
```

> **Note:** Choose the scoring metric according to the problem. For imbalanced classification, **accuracy alone may be misleading**.

---

### ✔ 6. SVM Works Well in High-Dimensional Feature Spaces

SVM can be particularly useful when the number of features \(p\) is large relative to the number of samples \(n\):

$$
\boxed{p>n}
$$

This occurs in applications such as:

* text classification
* gene-expression analysis
* image descriptors
* signal processing
* biomedical feature analysis

However, high dimensionality does **not automatically guarantee good SVM performance**. Feature quality, regularization, and computational cost still matter.

---

### ✔ 7. Use Linear SVM for Very Large Datasets

Kernel SVMs can become computationally expensive as the number of training samples increases.

For very large datasets, consider:

```python
from sklearn.svm import LinearSVC

model = LinearSVC(C=1.0)
model.fit(X_train_scaled, y_train)
```

A linear model is often much more scalable than a kernel SVM when the decision boundary is approximately linear.

---

### ✔ 8. Be Careful with Outliers

SVM is **not immune to outliers**.

In particular, hard-margin SVM can be extremely sensitive to outliers because every point must satisfy the margin constraint.

Soft-margin SVM provides some flexibility through slack variables:

$$
y_i(w^Tx_i+b)\geq1-\xi_i
$$

Therefore, always inspect the data for:

* extreme observations
* measurement errors
* mislabeled samples
* unusual feature values

---

### ✔ 9. Handle Class Imbalance

When classes are highly imbalanced, consider using class weights:

```python
model = SVC(
    kernel="rbf",
    class_weight="balanced"
)
```

This gives greater importance to minority-class samples.

Evaluate the model using appropriate metrics such as:

* Precision
* Recall
* F1-score
* Balanced Accuracy
* ROC-AUC
* PR-AUC

rather than relying only on accuracy.

---

### ✔ 10. Avoid Data Leakage

Scaling, feature selection, dimensionality reduction, and other preprocessing operations should be learned **only from the training data**.

A safer approach is to use a pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

pipeline.fit(X_train, y_train)
```

This ensures that preprocessing is correctly performed within the training process.

---

### ✔ 11. Consider PCA for Extremely High-Dimensional Data

When the number of features is extremely large, dimensionality reduction may help reduce computational cost and noise.

For example:

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("svm", SVC(kernel="rbf"))
])
```

> **Note:** PCA is not always necessary. If the original features are meaningful and computational cost is manageable, removing dimensions may actually discard useful information.

---

### ✔ 12. Use Cross-Validation for Reliable Evaluation

Do not judge an SVM only from a single train-test split.

Use cross-validation when the dataset size permits:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="f1_macro"
)

print("Mean F1:", scores.mean())
```

Report:

$$
\boxed{\text{Mean Performance} \pm \text{Standard Deviation}}
$$

rather than only reporting the best result.

---

## Quick Decision Guide

| Situation                      | Recommended Approach                  |
| ------------------------------ | ------------------------------------- |
| Approximately linear data      | Linear SVM                            |
| Nonlinear data                 | RBF SVM                               |
| Features have different scales | StandardScaler                        |
| Small/medium dataset           | Kernel SVM                            |
| Very large dataset             | LinearSVC / scalable alternatives     |
| Class imbalance                | `class_weight="balanced"`             |
| Need nonlinear boundary        | RBF / polynomial kernel               |
| High-dimensional data          | Linear SVM is often a strong baseline |
| Need hyperparameter selection  | GridSearchCV / RandomizedSearchCV     |
| Many noisy dimensions          | Consider feature selection/PCA        |
| Outliers present               | Soft-margin SVM + data inspection     |
| Data preprocessing required    | Use Pipeline                          |

---

## ⭐ Golden Rules for SVM

> **1. Scale your features.**
> **2. Start with a linear SVM when the problem appears linear.**
> **3. Try RBF when nonlinear relationships are expected.**
> **4. Tune \(C\) and \(\gamma\).**
> **5. Use cross-validation.**
> **6. Do not rely only on accuracy for imbalanced data.**
> **7. Watch for outliers and mislabeled samples.**
> **8. Prevent data leakage with pipelines.**
> **9. Use linear methods for very large datasets when appropriate.**
> **10. Compare SVM against simpler baselines rather than assuming it is always superior.**

### The practical SVM workflow

$$
\boxed{
\text{Data}
\rightarrow
\text{Train/Test Split}
\rightarrow
\text{Scaling}
\rightarrow
\text{Choose Kernel}
\rightarrow
\text{Tune }C,\gamma
\rightarrow
\text{Cross-Validation}
\rightarrow
\text{Evaluate}
}
$$

**Core principle:** SVM is powerful not simply because it finds a separating boundary, but because it combines **margin maximization, regularization, and kernel-based nonlinear modeling**.

## 12. Summary
**SVM is a maximum‑margin classifier that uses support vectors to define an optimal decision boundary. It can handle both linear and non‑linear classification using kernel functions and is robust, powerful, and widely used in practice.**
