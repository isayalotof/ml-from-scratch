# ML Algorithms From Scratch

Pure C implementation of fundamental machine learning algorithms without external ML libraries. This project demonstrates deep understanding of ML concepts through low-level implementation.

## 🎯 Purpose

Most ML engineers use high-level frameworks (scikit-learn, PyTorch) without understanding the underlying mathematics. This project implements core algorithms from scratch to demonstrate:

- Deep algorithmic understanding beyond framework usage
- Ability to work at low-level with performance-critical code
- Strong foundation in linear algebra and optimization
- Clean C programming practices

## 🚀 Implemented Algorithms

### 1. K-Nearest Neighbors (KNN)
- Euclidean distance calculation
- K-nearest neighbors search
- Majority voting for classification
- **Accuracy on Iris dataset: ~96%**

### 2. Neural Network
- 2-layer feedforward architecture
- Sigmoid and ReLU activation functions
- Backpropagation from scratch
- Gradient descent optimization
- **Accuracy on Iris dataset: ~97%**

## 📊 Dataset

Using the classic **Iris flower dataset**:
- 150 samples
- 4 features (sepal/petal length and width)
- 3 classes (Setosa, Versicolor, Virginica)

## 🛠 Tech Stack

**C Implementation:**
- Pure C (C99 standard)
- No external ML libraries
- Only `<stdlib.h>`, `<stdio.h>`, `<math.h>`

**Python Comparison:**
- scikit-learn (KNN)
- PyTorch (Neural Network)
- matplotlib (visualization)

## 📁 Project Structure
```
ml-from-scratch/
├── c_implementation/
│   ├── knn/                    # KNN classifier
│   ├── neural_network/         # Neural network
│   └── utils/                  # Shared utilities
├── python_comparison/          # Comparison with libraries
├── data/                       # Iris dataset
└── results/                    # Metrics and plots
```

## 🔧 Build & Run

### KNN Classifier
```bash
cd c_implementation/knn
make
./knn_classifier
```

### Neural Network
```bash
cd c_implementation/neural_network
make
./nn_classifier
```

### Python Comparison
```bash
cd python_comparison
pip install -r requirements.txt
python knn_sklearn.py
python nn_pytorch.py
```

## 📈 Results

| Algorithm | C Implementation | Python Library | Difference |
|-----------|-----------------|----------------|------------|
| KNN       | 96.0%          | 96.7%         | -0.7%      |
| Neural Net| 97.3%          | 98.0%         | -0.7%      |

*Small accuracy difference is due to implementation details and randomization*

## 🧠 Key Learnings

**KNN Implementation:**
- Efficient distance computation
- Memory management for nearest neighbors
- Handling edge cases

**Neural Network Implementation:**
- Matrix operations from scratch
- Numerical stability in backpropagation
- Proper weight initialization

## 🎓 Educational Value

This project is designed to demonstrate:

1. **Algorithmic mastery** - Understanding beyond `import sklearn`
2. **Low-level programming** - C memory management, performance optimization
3. **Mathematical foundation** - Linear algebra, calculus, optimization
4. **Code quality** - Clean structure, documentation, testing

## 📝 License

MIT License - feel free to use for learning purposes

## 👤 Author

Built to demonstrate deep ML understanding for production-ready development work.