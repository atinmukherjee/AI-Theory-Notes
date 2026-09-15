
# Motivation: Why SVM?

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
