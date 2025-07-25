# PyTorch Breast Cancer Classification and League of Legends Match Prediction Project 

This repository provides a detailed walkthrough of two machine learning projects developed using PyTorch. Each project explores a distinct classification problem, covering the entire pipeline from data preprocessing and model implementation to experimental analysis and results interpretation.

---

## 1. Breast Cancer Prediction Model

### Project Overview
This project addresses a critical task in medical diagnostics: the automated classification of breast tumors. The primary objective was to build, train, and evaluate a robust neural network capable of distinguishing between **Malignant (M)** and **Benign (B)** tumors using quantitative features from medical images. This serves as a practical application of deep learning to assist in early and accurate cancer detection. The project explores how choices in model architecture and optimization algorithms impact predictive performance.

### Dataset and Preprocessing
A thorough preprocessing pipeline was established to prepare the data for the neural network.

* **Data Source**: The project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset**, a well-regarded public dataset from the UCI Machine Learning Repository. It contains 569 instances, each with 30 physiologically relevant features.

* **Feature Description**: The 30 input features are real-valued measurements computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image, such as **radius**, **texture**, **perimeter**, **area**, and **smoothness**. These metrics are provided as a mean, standard error, and "worst" (mean of the three largest values) for each image, creating a comprehensive feature set.

* **Handling Class Imbalance**: The original dataset exhibited a class imbalance (357 Benign vs. 212 Malignant). A model trained on such data could develop a bias towards the majority class (Benign), leading to poor detection of the more critical Malignant cases. To mitigate this, the dataset was balanced by creating a 1:1 class ratio. This was achieved by taking a random sample of **200 Malignant** records and **200 Benign** records, resulting in a balanced dataset of 400 instances.

* **Data Splitting and Scaling**:
    * The balanced dataset was partitioned into a training set (80%) and a test set (20%). Stratification (`stratify=y`) was used during the split to ensure that the 1:1 class ratio was preserved in both the training and testing subsets.
    * Features were standardized using `sklearn.preprocessing.StandardScaler`. This process transforms each feature to have a mean of 0 and a standard deviation of 1. Scaling is crucial for neural networks, as it ensures that all features are on a comparable scale, which helps the gradient-based optimizer converge more quickly and stably.
    * Finally, the preprocessed NumPy arrays were converted to PyTorch Tensors. `TensorDataset` and `DataLoader` were used to create efficient data iterators that feed the model data in shuffled batches during training.

### Methodology and Model Architecture
A feed-forward neural network was designed and implemented using PyTorch's `nn.Module`.

* **Network Structure (`ClassificationNet`)**: The model is a custom class inheriting from `nn.Module`. Its architecture consists of:
    1.  An **input layer** (`nn.Linear`) that accepts 30 features and maps them to a hidden layer of a specified size (e.g., 64 units).
    2.  A **ReLU (Rectified Linear Unit)** activation function (`torch.relu`) applied to the hidden layer's output. This introduces non-linearity, allowing the model to learn complex relationships between features.
    3.  An **output layer** (`nn.Linear`) that maps the hidden layer's activations to 2 output logits, corresponding to the scores for the Benign and Malignant classes.

* **Training and Optimization**:
    * **Loss Function**: `nn.CrossEntropyLoss` was chosen as the loss function. It is ideal for multi-class classification as it combines a LogSoftmax layer and a Negative Log-Likelihood loss in one step, making it numerically stable and efficient.
    * **Training Loop**: The model was trained over a set number of epochs. In each epoch, the training `DataLoader` provides batches of data. For each batch, the gradients are cleared (`optimizer.zero_grad()`), a forward pass computes the predictions, the loss is calculated, backpropagation computes the gradients (`loss.backward()`), and the optimizer updates the model's weights (`optimizer.step()`).
    * **Evaluation**: After each training epoch, the model's performance was measured on the unseen test set. This was done in `eval` mode with gradient calculations disabled (`torch.no_grad()`) to prevent the model from learning from the test data.

### Experiments and Results
Several experiments were conducted to analyze the model's behavior under different configurations.

1.  **Baseline Model with Adam Optimizer**: The initial model used 64 hidden units and was trained with the **Adam optimizer** (`lr=0.001`). Adam is an adaptive learning rate optimization algorithm that is computationally efficient and well-suited for a wide range of problems. This model converged very quickly, achieving a low final test loss of **0.1032**, demonstrating its effectiveness.

2.  **Model with SGD Optimizer**: The same architecture was trained using **Stochastic Gradient Descent (SGD)** with a learning rate of 0.001, momentum of 0.9, and weight decay (L2 Regularization) of 0.0001. Momentum helps accelerate convergence, while weight decay helps prevent overfitting by penalizing large weights. This model also performed well, reaching a final test loss of **0.1374**, though its convergence was more gradual than the Adam-optimized model.

3.  **Impact of Model Capacity**: To test sensitivity to model complexity, a third experiment was run with a smaller hidden layer of only **16 units**, using the Adam optimizer. This less complex model still achieved an excellent test loss of **0.1063**, suggesting that the underlying patterns in the data can be captured effectively without requiring a large network, thus reducing the risk of overfitting.

---

## 2. League of Legends Match Prediction

### Project Overview
This project explores the predictability of match outcomes in the popular multiplayer online game League of Legends. A **logistic regression model** was implemented from scratch in PyTorch to predict a team's victory or defeat based on a set of fundamental in-game performance statistics. The primary focus was not just on building a model, but on conducting a comprehensive evaluation of its performance and analyzing which gameplay features are most indicative of a win.

### Dataset and Preprocessing
The dataset used is a custom CSV file containing aggregated statistics from numerous matches.

* **Features**: The model was trained on 8 core in-game metrics:
    * `kills`, `deaths`, `assists`: Core combat statistics.
    * `gold_earned`: The primary economic resource.
    * `cs` (Creep Score): A measure of farming efficiency.
    * `wards_placed`, `wards_killed`: Metrics related to map vision and control.
    * `damage_dealt`: Total damage output to enemy champions.
* **Target**: A binary `win` variable, where 1 represents a victory and 0 a loss.
* **Data Preparation**: The data was split into a training set (800 matches) and a test set (200 matches). Following the same best practices as the first project, `StandardScaler` was applied to normalize the feature space, and the data was organized into batches using PyTorch's `DataLoader`.

### Methodology and Model Architecture
A simple yet interpretable model was chosen for this binary classification task.

* **Model (`logistic_regression`)**: A logistic regression model was defined as a `nn.Module` subclass. Its architecture is composed of:
    1.  A single `nn.Linear` layer that takes the 8 input features and computes a single output value (a logit). This layer learns a weight for each feature, representing its importance and direction of influence.
    2.  A `torch.sigmoid` activation function that transforms the logit into a probability between 0 and 1, representing the model's predicted probability of a win.
* **Training and Evaluation**:
    * **Loss Function**: `nn.BCELoss` (Binary Cross-Entropy Loss) was used. This is the standard loss function for binary classification problems, as it measures the dissimilarity between the predicted probabilities and the actual binary labels (0 or 1).
    * **Evaluation Metrics**: Performance was assessed using a suite of metrics: training and test loss, accuracy, a **confusion matrix** to visualize true vs. predicted outcomes, a **classification report** detailing precision, recall, and F1-score for each class, and an **ROC (Receiver Operating Characteristic) curve** with its corresponding AUC (Area Under the Curve) score to measure the model's discriminative ability.

### Results and Detailed Analysis
* **Overall Performance**: After extensive training (1000 epochs) with regularization (`weight_decay=0.01`), the model's performance was modest. It achieved a final test accuracy of **51%** and an ROC AUC score of **0.51**. An AUC score of 0.5 represents a model with no predictive power (equivalent to random guessing), so 0.51 indicates a very marginal ability to distinguish between wins and losses. This result strongly suggests that a simple linear model is insufficient to capture the complex, dynamic, and non-linear interactions that determine a League of Legends match outcome.

* **In-Depth Evaluation**:
    * The **classification report** showed that the model's precision and recall for both classes were around 50%, confirming the lack of strong predictive signal.
    * The **ROC curve** was very close to the diagonal "no-discrimination" line, visually confirming the low AUC score.
    * Experiments with hyperparameter tuning, such as adjusting the learning rate, did not yield significant improvements, further highlighting the limitations of the linear model and/or the feature set for this complex task.

* **Feature Importance Analysis**: Despite the low overall accuracy, an analysis of the model's learned weights provided valuable insights. The weights from the `nn.Linear` layer were extracted to rank features by importance.
    * **Positive Predictors**: **`gold_earned`** and **`kills`** emerged as the features with the largest positive weights. This aligns perfectly with game intuition, as economic advantage and combat superiority are universally recognized as key drivers of victory.
    * **Negative Predictors**: Features like `cs` and `wards_killed` had small negative weights. While counter-intuitive at first glance, this could be an artifact of the simple model trying to balance correlations in the data, rather than a true reflection of game dynamics.

This analysis demonstrated that even a simple, low-accuracy model can be used to extract interpretable insights that confirm domain knowledge. The model's primary value was not in its predictive power, but as a tool for evaluation and feature analysis.