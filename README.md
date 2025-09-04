# Deep Learning Optimizer & Hyperparameter Tuning Study

A comprehensive project exploring the impact of different optimizers and hyperparameters on the performance of a neural network for a multi-class classification task. This study uses the UCI Wheat Seeds dataset and systematically tunes the learning rate, momentum, and weight decay for the SGD optimizer, comparing the final tuned model against adaptive optimizers like Adam, RMSprop, and Adagrad.

---

## 📋 Table of Contents

- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Findings](#-key-findings)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)

---

## 🌾 Dataset

This project uses the **UCI Seeds Dataset**.

- **Content**: Measurements of geometrical properties of kernels belonging to three different varieties of wheat: Kama, Rosa, and Canadian.
- **Features**: 7 numerical features (area, perimeter, compactness, etc.).
- **Target**: 3 distinct classes (wheat varieties).
- **Size**: 210 samples, perfectly balanced with 70 samples per class.
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/seeds)



---

## 🔬 Methodology

The experiment follows a systematic approach to isolate and analyze the effect of each hyperparameter on model performance.

1.  **Data Preprocessing**: The dataset is loaded, features are normalized using `StandardScaler`, and target labels are encoded. The data is then split into training (80%) and validation (20%) sets.

2.  **Model Architecture**: A sequential neural network is built using TensorFlow/Keras with the following structure:
    -   `Dense(64, activation='relu')`
    -   `Dropout(0.3)`
    -   `Dense(32, activation='relu')`
    -   `Dropout(0.2)`
    -   `Dense(16, activation='relu')`
    -   `Dense(3, activation='softmax')` (Output Layer)

3.  **Learning Rate Search**: The model is trained using SGD with four different learning rates (`0.1`, `0.01`, `0.001`, `0.0001`) to find the optimal starting value.

4.  **Momentum Search**: Using the best learning rate, the model is trained with different momentum values (`0.0`, `0.5`, `0.9`, `0.99`) to analyze its effect on convergence and stability.

5.  **Weight Decay Search**: The best learning rate and momentum are used to test four different decay values (`1e-1`, `1e-2`, `1e-3`, `1e-4`) to find the optimal regularization strength.

6.  **Optimizer Comparison**: The performance of the fully-tuned SGD optimizer is benchmarked against common adaptive optimizers: **Adam**, **Adagrad**, and **RMSprop**.

7.  **Final Evaluation**: A final model is trained using the best-discovered configuration and evaluated on the validation set, with performance detailed in a classification report and confusion matrix.

---

## ✨ Key Findings

The comprehensive study revealed that a **finely-tuned SGD optimizer was demonstrably superior** for this dataset, challenging the common practice of defaulting to adaptive optimizers.

-   **Optimal Configuration**:
    -   **Optimizer**: SGD
    -   **Learning Rate**: `0.1`
    -   **Momentum**: `0.9`
    -   **Weight Decay**: `0.0001`

-   **Performance**: The tuned SGD achieved a **peak validation accuracy of 95.24%** during the search, outperforming Adam (92.86%), RMSprop (92.86%), and Adagrad (90.48%).

-   **Insight**: The results strongly suggest that for smaller, well-structured datasets, an aggressively tuned classical optimizer can find a better solution than general-purpose adaptive algorithms. The final model's main weakness was in classifying one of the three classes, indicating that its features were less distinct than the others.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Deep-Learning-Optimizer-Tuning.git](https://github.com/YourUsername/Deep-Learning-Optimizer-Tuning.git)
    cd Deep-Learning-Optimizer-Tuning
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis:**
    Open and run the `optimizer_study.ipynb` notebook in Jupyter Lab or Jupyter Notebook.

---

## 📦 Dependencies

-   TensorFlow
-   Pandas
-   Scikit-learn
-   Matplotlib
-   Seaborn
