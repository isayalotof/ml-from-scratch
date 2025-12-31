import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def load_iris_data(filepath):
    """Load Iris dataset from CSV"""
    # Column names for Iris dataset
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    # Load data
    data = pd.read_csv(filepath, names=columns)

    # Convert class names to numeric labels
    class_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    data['class'] = data['class'].map(class_mapping)

    # Separate features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return X, y


def main():
    print("=== KNN Classifier (scikit-learn) for Iris Dataset ===\n")

    # Load dataset
    X, y = load_iris_data('../data/iris.csv')
    print(f"Loaded {len(X)} samples with {X.shape[1]} features\n")

    # Print first 3 samples
    print("First 3 samples:")
    for i in range(3):
        print(f"Sample {i}: {X[i]} -> Class {y[i]}")

    # Split dataset (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nSplit dataset: {len(X_train)} train samples, {len(X_test)} test samples\n")

    # Test different K values
    k_values = [3, 5, 7]
    accuracies = []

    print("=== Testing different K values ===")

    for k in k_values:
        print(f"\n--- K = {k} ---")

        # Create and train model
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Print first 10 predictions
        print("Evaluating KNN model...")
        for i in range(min(10, len(X_test))):
            status = "[CORRECT]" if y_pred[i] == y_test[i] else "[WRONG]"
            print(f"Sample {i}: Predicted={y_pred[i]}, Actual={y_test[i]} {status}")

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        correct = int(accuracy * len(y_test))
        print(f"\nAccuracy: {accuracy*100:.2f}% ({correct}/{len(y_test)} correct)")

    # Plot accuracy comparison
    print("\n=== Generating accuracy plot ===")
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, [acc * 100 for acc in accuracies], marker='o', linewidth=2, markersize=8)
    plt.xlabel('K value', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('KNN Accuracy vs K Value (scikit-learn)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.ylim([80, 105])

    # Save plot
    plt.savefig('../results/knn_sklearn_accuracy.png', dpi=300, bbox_inches='tight')
    print("Saved plot to: ../results/knn_sklearn_accuracy.png")

    # Detailed report for best K
    best_k_idx = np.argmax(accuracies)
    best_k = k_values[best_k_idx]
    print(f"\n=== Best K = {best_k} with {accuracies[best_k_idx]*100:.2f}% accuracy ===")

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor', 'Virginica']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Completed successfully! ===")


if __name__ == "__main__":
    main()
