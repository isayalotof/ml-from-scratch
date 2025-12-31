# ML Algorithms From Scratch

Pure C implementation of fundamental machine learning algorithms without external ML libraries. This project demonstrates deep understanding of ML concepts through low-level implementation.

## 🎯 Purpose

Most ML engineers use high-level frameworks (scikit-learn, PyTorch) without understanding the underlying mathematics. This project implements core algorithms from scratch to demonstrate:

- **Deep algorithmic understanding** beyond framework usage
- **Low-level programming skills** with performance-critical code
- **Strong foundation** in linear algebra, calculus, and optimization
- **Clean C programming** practices with proper memory management

## 🚀 Implemented Algorithms

### 1. K-Nearest Neighbors (KNN)
Classic distance-based classifier implemented with:
- Euclidean distance calculation
- Efficient K-nearest neighbors search using qsort
- Majority voting for classification
- **Performance: 93-100% accuracy** depending on K value

**Technical highlights:**
- Custom comparator for neighbor sorting
- Optimized distance computation
- Proper memory management for dynamic arrays

### 2. Neural Network (Feedforward + Backpropagation)
2-layer fully connected network with complete training pipeline:
- **Architecture:** 4 inputs → 8 hidden neurons → 3 outputs
- **Activations:** ReLU (hidden layer), Softmax (output layer)
- **Optimization:** Stochastic Gradient Descent with backpropagation
- **Weight initialization:** Xavier/Glorot initialization
- **Loss function:** Cross-entropy
- **Performance: 96.67% accuracy** on Iris dataset

**Technical highlights:**
- Forward propagation with matrix operations
- Backpropagation with proper gradient computation
- Numerical stability (softmax with max subtraction)
- Gradient descent weight updates
- All implemented from scratch - no BLAS, no ML libraries

## 📊 Dataset

Using the classic **Iris flower dataset**:
- 150 samples total
- 4 features: sepal length/width, petal length/width
- 3 classes: Setosa, Versicolor, Virginica
- Train/test split: 80/20 (120 train, 30 test)

## 🛠 Tech Stack

**C Implementation:**
- Pure C (C99 standard)
- GCC compiler with `-O2` optimization
- Only standard libraries: `<stdlib.h>`, `<stdio.h>`, `<math.h>`, `<string.h>`
- **No external ML libraries** - everything from scratch

**Python Comparison:**
- scikit-learn 1.3+ (KNN baseline)
- PyTorch 2.0+ (Neural Network baseline)
- NumPy, Pandas (data handling)
- Matplotlib (visualization)

## 📁 Project Structure

```
ml-from-scratch/
├── README.md
├── c_implementation/
│   ├── knn/
│   │   ├── knn.h                   # KNN header
│   │   ├── knn.c                   # KNN implementation
│   │   ├── main.c                  # KNN test program
│   │   └── Makefile                # Build configuration
│   ├── neural_network/
│   │   ├── nn.h                    # Neural network header
│   │   ├── nn.c                    # Network + backprop
│   │   ├── activation.h            # Activation functions
│   │   ├── activation.c            # Sigmoid, ReLU, Softmax
│   │   ├── main.c                  # NN test program
│   │   └── Makefile                # Build configuration
│   └── utils/
│       ├── matrix.h/c              # Matrix operations
│       └── data_loader.h/c         # CSV loading, dataset handling
├── python_comparison/
│   ├── knn_sklearn.py              # scikit-learn KNN
│   ├── nn_pytorch.py               # PyTorch neural network
│   └── requirements.txt            # Python dependencies
├── data/
│   └── iris.csv                    # Iris dataset
└── results/
    ├── knn_sklearn_accuracy.png    # KNN performance plot
    └── nn_pytorch_loss.png         # NN training curve
```

## 🔧 Build & Run

### Prerequisites

**C Compiler:**
- GCC (Linux/macOS: `gcc --version`)
- Windows: MSYS2 MinGW 64-bit

**Python (for comparison scripts):**
- Python 3.8+
- See `python_comparison/requirements.txt`

### KNN Classifier

```bash
cd c_implementation/knn
make
./knn_classifier        # Linux/macOS
./knn_classifier.exe    # Windows
```

**Expected output:**
```
=== KNN Classifier for Iris Dataset ===
Loaded 150 samples with 4 features
...
--- K = 5 ---
Accuracy: 100.00% (30/30 correct)
```

### Neural Network

```bash
cd c_implementation/neural_network
make
./nn_classifier         # Linux/macOS
./nn_classifier.exe     # Windows
```

**Expected output:**
```
=== Neural Network Classifier for Iris Dataset ===
...
Epoch 1/1000 - Loss: 1.3559
Epoch 100/1000 - Loss: 0.1126
...
Epoch 1000/1000 - Loss: 0.0810
Accuracy: 96.67% (29/30 correct)
```

### Python Comparison

```bash
cd python_comparison
pip install -r requirements.txt
python knn_sklearn.py
python nn_pytorch.py
```

## 📈 Results Comparison

### K-Nearest Neighbors

| K Value | C Implementation | scikit-learn | Difference |
|---------|-----------------|--------------|------------|
| K=3     | 93.33%         | 96.67%      | -3.34%     |
| K=5     | 100.00%        | 96.67%      | +3.33%     |
| K=7     | 100.00%        | 96.67%      | +3.33%     |

*Performance varies due to random train/test split*

### Neural Network

| Metric | C Implementation | PyTorch | Difference |
|--------|-----------------|---------|------------|
| Final Loss | 0.0810 | ~0.08 | ≈ 0% |
| Test Accuracy | 96.67% | ~97% | ≈ 0% |
| Training Time | Fast | Fast | Comparable |

*Both implementations show similar convergence and accuracy*

## 🧠 Key Learnings

### KNN Implementation
- **Distance calculation** - Efficient Euclidean distance without unnecessary sqrt until needed
- **Neighbor search** - Using stdlib's qsort with custom comparator for O(n log n) sorting
- **Memory management** - Dynamic allocation and proper cleanup to avoid leaks
- **Edge cases** - Handling k > n_samples, tie-breaking in voting

### Neural Network Implementation
- **Matrix operations** - Efficient forward/backward passes without external BLAS
- **Backpropagation** - Chain rule application for gradient computation
- **Numerical stability** - Softmax with max subtraction to prevent overflow
- **Weight initialization** - Xavier/Glorot for stable gradient flow
- **Gradient descent** - SGD implementation with proper learning rate

### Why This Matters
Most developers treat ML as a black box:
```python
model = SomeModel()
model.fit(X, y)  # Magic happens here
```

This project shows what's **actually happening** inside `.fit()`:
- How gradients are computed (backprop)
- How weights are updated (SGD)
- How predictions are made (forward pass)
- How loss is calculated and minimized

## 🎓 Educational Value

This project demonstrates:

1. **Algorithm mastery** - Understanding beyond `import sklearn`
2. **Low-level programming** - C memory management, pointer arithmetic
3. **Mathematical foundation** - Linear algebra, calculus, optimization theory
4. **Code quality** - Clean structure, proper documentation, testing
5. **Performance awareness** - Big-O complexity, memory efficiency

Perfect for:
- ML engineers wanting to understand fundamentals
- Demonstrating deep technical knowledge to employers
- Interview preparation (explain backpropagation from scratch)
- Teaching ML concepts with actual working code

## 🚀 Future Enhancements

Potential additions to expand the project:
- [ ] Convolutional Neural Network (CNN) for image classification
- [ ] Decision Trees and Random Forests
- [ ] Support Vector Machines (SVM)
- [ ] Gradient Boosting (XGBoost-style)
- [ ] Recurrent Neural Networks (RNN/LSTM)
- [ ] Additional optimizers (Adam, RMSprop)
- [ ] Batch normalization and dropout
- [ ] Multi-threading for parallel training

## 📝 License

MIT License - Feel free to use for learning and educational purposes.

## 👤 Author

Built to demonstrate deep ML understanding for production-ready development work.

**Tech Stack:** C, Python, Machine Learning, Neural Networks, Backpropagation

**Key Skills:** Algorithm implementation, Low-level programming, Mathematical optimization, Code quality