# Support Vector Machine (SVM)
In this content we will discuss about the support Vector Machine (SVM), an approch for classification that was developed in 1990s. The SVM is often considered one of the best "out of the box" classifiers. Many real-world classification problems require a decision boundary that not only separates different classes but also provides a **large margin of separation** between them.Earlier methods such as **Logistic Regression** and **Linear Discriminant Analysis (LDA)** can produce effective linear decision boundaries. However, they may not explicitly optimize the **margin between classes**, and linear models cannot directly handle cases where the classes are **not linearly separable**. SVM addresses these challenges by finding an **optimal separating hyperplane** that maximizes the margin between classes.

>[!Note]
>Then Support vector Machine is a generalization of a simple and intutive classifier called *Maximal margin classifier*. It is elegent and simple but it cannot be applied to most of the dataset, since it requires classes be seperable by a linear boundary. <ins>Therefore, the *Support Vector Classifier* is introduced as a extension of *Maximal margin classsifier*. Support Vector Machine (SVM), which is further extension of the support vector classifier in order to accommodate non-linear class boundaries.</ins>
 Since the Support Vector Machine (SVM) can be viewed as a generalization of the Maximal Margin Classifier, it is useful to first understand the fundamental concepts of the Maximal Margin Classifier before delving into SVMs.

## Maximal Margin Classifier
**A Maximal Margin Classifier (MMC) is a linear classification method that finds a hyperplane separating two classes while maximizing the minimum distance (margin) between the hyperplane and the training observations.** 

In simple terms, among all possible separating hyperplanes, it chooses the one that leaves the largest possible margin between the two classes.

>[!IMPORTANT]
>**The Maximal Margin Classifier works only when the training data are perfectly linearly separable. If the classes overlap or contain noisy observations, a separating hyperplane may not exist. This limitation motivates the development of the Support Vector Classifier (Soft-Margin SVM).**

### Hyperplane and Geometry
In p-dimension space , a *hyperplane* is a flat affine subspace of dimension $(p-1)$. Here *affine* indicates that the subspace need not pass through the originFor 2D space  a hyperplane is a flat one-dimensional subspace means a *line* and for 3D space hyperplane is a flat two-dimensional subspace means *plane*. 

The mathematical definition of a hyperplane is quite simple. In two dimension, a hyperplane is defined by the equation
$$ 
\beta_0 + \beta_1 X_1 + \beta_2 X_2 = 0
$$
where:

$\beta_0$ is the intercept.
$\beta_1$ and $\beta_2$ are the coefficients (weights) associated with the features $X_1$ and $X_2$.
$X_1$ and $X_2$ are the two feature values of a data point.
A data point can be represented as:

$$
X=(X_1,X_2)^T
$$

Note that eq.**(1)** is simply the equation of a line, since indeed in two dimensions a hyperplane is a line. For p-dimension space we can extent the equation **(1)** :
$$
\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p = 0
$$

$$
\beta_0 + \sum_{j=0}^p \beta_j X_j = 0
$$

defines a hyperplane in p-dimension feature space.

Suppose that **X** does not satisfy **(2)** . A data point $$X=(X_1,X_2,\ldots,X_p)^T$$ may either lie on the hyperplane or on one of its two sides.
If

$$
\beta_0 + \sum_{j=0}^p \beta_j X_j = 0
$$

then the point (X) lies exactly on the hyperplane.

If

$$
\beta_0 + \sum_{j=0}^p \beta_j X_j > 0

$$

then (X) lies on one side of the hyperplane.

If

$$
\beta_0 + \sum_{j=0}^p \beta_j X_j < 0
$$

Therefore, a hyperplane divides the (p)-dimensional feature space into two regions (half-spaces).

**This property is fundamental to classification methods such as the Maximal Margin Classifier and Support Vector Machine (SVM), where the hyperplane is used to separate observations belonging to different classes.**
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
If data is in two dimension then hyperplane is one dimension. if you have n number of dimension then hyperpnae is the n-1 dimension.
## 7. Geometric Interpretation
## 8. SVM Parameters (`C`, `γ`, Kernel)

The performance of an SVM depends strongly on the choice of its **hyperparameters**. The three important parameters are:

* **`C`** — controls the penalty for margin violations.
* **`γ` (gamma)** — controls the influence of individual training samples in the **RBF kernel**.
* **`kernel`** — determines the mathematical function used to construct the decision boundary.

---

### 8.1 Parameter `C`

The parameter **`C`** controls how strongly the SVM penalizes training samples that violate the margin.

In soft-margin SVM, the optimization problem is:

$$
\min_{w,b,\xi}
\frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i
$$

where:

* $\frac{1}{2}|w|^2$ controls the **margin size**
* $\xi_i$ represents the **margin violation**
* $C$ controls the penalty associated with these violations

### Small `C`

A small value of `C` means that margin violations are penalized less.

Therefore, the model can tolerate more training errors in order to obtain a **wider margin**.

> **Small `C` → Wider margin → More violations allowed → Simpler decision boundary**

### Large `C`

A large value of `C` means that margin violations are penalized heavily.

The model therefore tries harder to classify training samples correctly, which can result in a **narrower margin** and a more complex boundary.

> **Large `C` → Narrower margin → Fewer violations allowed → Stronger fit to training data**

### Intuition

Think of `C` as the **strictness of the SVM**:

$$
\boxed{
\text{Small }C \rightarrow \text{More tolerance}
}
$$

$$
\boxed{
\text{Large }C \rightarrow \text{Less tolerance}
}
$$

---

## 8.2 Parameter `γ` (Gamma)

The parameter **`γ`** is mainly important for the **RBF kernel**.

The RBF kernel is defined as:

$$
K(x_i,x_j)
=
\exp\left(-\gamma\|x_i-x_j\|^2\right)
$$

Gamma determines how quickly the influence of a training sample decreases as the distance from that sample increases.

### Small `γ`

A small value of gamma gives each training sample a **larger region of influence**.

This generally produces a smoother and less complex decision boundary.

> **Small `γ` → Wider influence → Smoother boundary → Lower complexity**

### Large `γ`

A large value of gamma gives each training sample a **smaller region of influence**.

The model can therefore create a more flexible and complex decision boundary.

> **Large `γ` → Narrow influence → More complex boundary → Higher risk of overfitting**

### Intuition

Gamma controls the **reach of each training sample**:

$$
\boxed{
\text{Small }\gamma \rightarrow \text{Broad influence}
}
$$

$$
\boxed{
\text{Large }\gamma \rightarrow \text{Localized influence}
}
$$

> [!NOTE]
> **`C` and `γ` control different aspects of the model.**
> `C` controls how strongly the model penalizes margin violations, whereas `γ` controls the locality and flexibility of the RBF decision boundary.

---

## 8.3 Kernel

The **kernel** determines how SVM represents relationships between data points.

A kernel allows SVM to construct nonlinear decision boundaries without explicitly transforming the original features into a higher-dimensional space.

This idea is known as the **Kernel Trick**.

### Common SVM Kernels

| Kernel    | Main Idea                   | Typical Use                                |
| --------- | --------------------------- | ------------------------------------------ |
| `linear`  | Linear decision boundary    | Linearly separable / high-dimensional data |
| `poly`    | Polynomial relationship     | Polynomial nonlinear patterns              |
| `rbf`     | Flexible nonlinear boundary | General nonlinear problems                 |
| `sigmoid` | Sigmoid-shaped similarity   | Less commonly used                         |

### Linear Kernel

The linear kernel is:

$$
K(x_i,x_j)=x_i^Tx_j
$$

It produces a linear decision boundary:

$$
w^Tx+b=0
$$

---

### Polynomial Kernel

The polynomial kernel can be written as:

$$
K(x_i,x_j)
=
(\gamma x_i^Tx_j+r)^d
$$

where:

* $\gamma$ controls the influence of the input
* $r$ is a coefficient
* $d$ is the polynomial degree

It can model polynomial nonlinear relationships.

---

### RBF Kernel

The Radial Basis Function (RBF) kernel is:

$$
K(x_i,x_j)
=
\exp\left(-\gamma\|x_i-x_j\|^2\right)
$$

The RBF kernel is widely used because it can model complex nonlinear relationships.

Its flexibility is controlled mainly by **`γ`**.

---

## 8.4 Relationship Between `C` and `γ`

For an RBF-SVM, `C` and `γ` influence the model in different ways.

| Parameter | Small Value                   | Large Value                       |
| --------- | ----------------------------- | --------------------------------- |
| **`C`**   | Wider margin, more violations | Narrower margin, fewer violations |
| **`γ`**   | Smoother boundary             | More complex boundary             |

A useful conceptual view is:

$$
\boxed{
C \rightarrow \text{Penalty for errors / margin violations}
}
$$

$$
\boxed{
\gamma \rightarrow \text{Locality / complexity of the RBF boundary}
}
$$

Therefore, both parameters usually need to be considered together when tuning an RBF-SVM.

---

## 8.5 Hyperparameter Tuning

The optimal values of `C` and `γ` depend on the dataset.

Rather than selecting them manually, they can be searched using **cross-validation**.

For example:

```python
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1]
}
```

With scikit-learn:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid = GridSearchCV(
    SVC(kernel="rbf"),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
```

The important principle is:

$$
\boxed{
\text{Choose parameters using training data + Cross-Validation}
}
$$

The **test set should remain untouched** until the final evaluation.

---

## 8.6 Quick Summary

| Parameter    | Controls                      | Key Effect                                 |
| ------------ | ----------------------------- | ------------------------------------------ |
| **`C`**      | Penalty for margin violations | Controls margin–error trade-off            |
| **`γ`**      | Sample influence in RBF       | Controls boundary locality/complexity      |
| **`kernel`** | Feature similarity function   | Determines linear/nonlinear representation |

### Core Idea

$$
\boxed{
\text{Kernel}
\rightarrow
\text{Type of Decision Boundary}
}
$$

$$
\boxed{
C
\rightarrow
\text{Penalty for Margin Violations}
}
$$

$$
\boxed{
\gamma
\rightarrow
\text{Locality of the RBF Boundary}
}
$$

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

## 10.Example: Breast Cancer Classification

In the previous section, we implemented SVM using the **Iris dataset**. Here, we use a different dataset to demonstrate how SVM can be applied to a more realistic **binary classification problem with multiple numerical features**.

We use the **Breast Cancer Wisconsin Diagnostic dataset**, which is available directly through `scikit-learn`.

The task is to classify samples into two classes:

* **Malignant**
* **Benign**

The dataset contains **569 samples and 30 numerical features**.

---

### 10.1 Import Required Libraries

```python
import numpy as np

from sklearn.datasets import load_breast_cancer
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

### 10.2 Load the Dataset

```python
data = load_breast_cancer()

X = data.data
y = data.target

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Classes:", data.target_names)
```

Expected output:

```text
Feature shape: (569, 30)
Target shape: (569,)
Classes: ['malignant' 'benign']
```

Therefore:

$$
X \in \mathbb{R}^{569\times30}
$$

There are:

$$
n=569 \quad \text{samples}
$$

and

$$
p=30 \quad \text{features}
$$

---

### 10.3 Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
```

The test set is kept separate and is not used during hyperparameter tuning.

---

### 10.4 Build the SVM Pipeline

Because SVM is sensitive to feature scale, we first standardize the features.

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])
```

The pipeline performs:

$$
\text{Raw Features}
\rightarrow
\text{Standardization}
\rightarrow
\text{SVM}
$$

---

### 10.5 Define Hyperparameters

For this example, we will compare **linear** and **RBF** kernels.

```python
param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1]
}
```

Here:

* `kernel` → determines the type of decision boundary
* \(C\) → controls the penalty for margin violations
* \(\gamma\) → controls the influence of individual samples for nonlinear kernels

---

### 10.6 Hyperparameter Tuning

Use `GridSearchCV` with 5-fold cross-validation.

```python
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
```

The training data is divided into five folds:

```text
Fold 1 → Validation
Fold 2 → Training
Fold 3 → Training
Fold 4 → Training
Fold 5 → Training
```

The process is repeated so that every fold is used for validation.

The average validation performance is used to compare different hyperparameter combinations.

---

### 10.7 Best Hyperparameters

```python
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)
```

The exact best parameters depend on the search space and dataset split.

---

### 10.8 Evaluate on the Test Set

After selecting the best model using the training data, evaluate it on the previously unseen test set.

```python
best_model = grid_search.best_estimator_

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
        target_names=data.target_names
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

### 10.9 Inspect Support Vectors

The support vectors can be inspected from the trained `SVC` model.

```python
svm_model = best_model.named_steps["svm"]

print("Number of support vectors:")
print(svm_model.n_support_)

print("\nTotal support vectors:")
print(len(svm_model.support_))
```

The support vectors are the training samples that play an important role in defining the SVM decision boundary.

---

## 10.10 Complete Example

The complete implementation can be written as:

```python
import numpy as np

from sklearn.datasets import load_breast_cancer
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

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Classes:", data.target_names)

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
# 4. Hyperparameter Search Space
# --------------------------------------------------

param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1]
}

# --------------------------------------------------
# 5. Hyperparameter Tuning
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
# 6. Best Model
# --------------------------------------------------

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Accuracy:")
print(grid_search.best_score_)

best_model = grid_search.best_estimator_

# --------------------------------------------------
# 7. Test Prediction
# --------------------------------------------------

y_pred = best_model.predict(X_test)

# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 9. Support Vectors
# --------------------------------------------------

svm_model = best_model.named_steps["svm"]

print("\nNumber of Support Vectors:")
print(svm_model.n_support_)

print("\nTotal Support Vectors:")
print(len(svm_model.support_))
```

---

### 10.11 What This Example Demonstrates

This example connects the **theory of SVM** with a complete machine-learning workflow:

$$
\boxed{
\begin{aligned}
&\text{Real Dataset}\\
&\downarrow\\
&\text{Train/Test Split}\\
&\downarrow\\
&\text{Feature Scaling}\\
&\downarrow\\
&\text{Linear/RBF Kernel}\\
&\downarrow\\
&\text{Hyperparameter Tuning}\\
&\downarrow\\
&\text{5-Fold Cross-Validation}\\
&\downarrow\\
&\text{Best SVM}\\
&\downarrow\\
&\text{Unseen Test Set}\\
&\downarrow\\
&\text{Performance Evaluation}
\end{aligned}
}
$$

> **Key Insight:** This example demonstrates an important advantage of SVM: it can operate effectively in a feature space with many dimensions, while the kernel mechanism allows nonlinear decision boundaries when required.

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
