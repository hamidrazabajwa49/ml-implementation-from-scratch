# 60 Days of AI/ML From Scratch

**Building every component of machine learning from first principles — no ML libraries, just Python.**

---

## What This Is

This repository documents a 60-day structured challenge to implement the full stack of machine learning — from raw linear algebra through to neural networks and research paper implementations — entirely from scratch.

No Scikit-learn. No PyTorch. No shortcuts.

The rule is simple: if I cannot build it, I do not understand it. Every file in this repo exists because I wanted to own the concept, not just use it.

---

## Why From Scratch

Most ML education runs in one direction — here is the API, here is the result, now tune the hyperparameters. That approach produces people who can run models but cannot explain why a loss curve behaves the way it does, or what is actually happening inside a gradient update.

I wanted to go the other direction. Start with the math. Build the operations by hand. Make the mistakes that happen when there is nothing between you and the logic. Then, when I eventually use the high-level libraries, I will know what they are doing underneath.

This is also preparation for AI safety research specifically. You cannot reason carefully about what a model has learned if you treat the model as a black box. The goal of this challenge is to remove every black box I can.

---

## Books and Resources Behind This Work

These are the texts I worked through alongside this challenge. Not a reading list — actual books I used, got stuck in, and built things from.

- **Mathematics for Machine Learning** — Deisenroth, Faisal, Ong
- **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** — Aurélien Géron
- **Data Science from Scratch** — Joel Grus
- **Python Data Science Handbook** — Jake VanderPlas
- **Why Machine Learning Works** — Valiant

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Primary language |
| NumPy | Low-level array operations only |
| Matplotlib | Visualizations |
| Jupyter Notebooks | Exploration and annotation |

No ML frameworks. No statistical shortcut libraries. If the math can be written, it gets written.

---

## Repository Structure

```
60-days-ml-from-scratch/
│
├── phase-1-math-foundations/
│   ├── 01_linear_algebra/
│   ├── 02_calculus_and_gradients/
│   ├── 03_probability_theory/
│   └── 04_descriptive_statistics/
│
├── phase-2-core-ml/
│   ├── 05_data_preprocessing/
│   ├── 06_linear_regression/
│   ├── 07_logistic_regression/
│   ├── 08_knn_and_clustering/
│   ├── 09_decision_trees/
│   └── 10_model_evaluation/
│
├── phase-3-neural-networks/
│   ├── 11_perceptron/
│   ├── 12_multilayer_perceptron/
│   ├── 13_backpropagation/
│   ├── 14_activation_functions/
│   ├── 15_optimizers/
│   └── 16_regularization/
│
├── phase-4-deep-learning/
│   ├── 17_convolutional_networks/
│   ├── 18_recurrent_networks/
│   ├── 19_attention_mechanism/
│   └── 20_transformer_from_scratch/
│
└── phase-5-research-papers/
    ├── paper_01/
    ├── paper_02/
    └── paper_03/
```

---

## Full 60-Day Roadmap

### Phase 1 — Mathematical Foundations (Days 1–15)

This phase is the base layer. Everything in ML rests on these concepts. Skipping them means you will hit a ceiling eventually and not know why.

---

**Days 1–3 | Linear Algebra Engine**

Build vector and matrix operations from scratch. This is not just syntax practice — understanding what a dot product actually computes, and why matrix multiplication works the way it does, is the foundation of understanding every layer in a neural network.

Topics:
- Vectors: addition, scalar multiplication, dot product, magnitude, unit vectors
- Matrices: multiplication, transpose, inverse, determinant
- Eigenvalues and eigenvectors (intuition and computation)
- Geometric interpretation of linear transformations

Output: `linear_algebra.py` — a working module with all operations implemented and tested

---

**Days 4–6 | Calculus and Gradients**

The optimizer does not exist without this. The whole mechanism of learning — gradient descent, backpropagation, weight updates — is applied calculus. Build it so the formulas are not abstract.

Topics:
- Derivatives: definition, rules (chain, product, quotient)
- Partial derivatives and gradient vectors
- The Jacobian and Hessian matrices
- Numerical differentiation vs. analytic differentiation
- Gradient descent implemented step-by-step

Output: `calculus_engine.py` — gradient computation and a working gradient descent loop with visualization

---

**Days 7–9 | Probability Theory**

ML models output probabilities constantly. Most people using them have no intuition for what those numbers actually mean. This section builds that intuition.

Topics:
- Sample spaces, events, probability axioms
- Conditional probability and independence
- Bayes' theorem — derivation and worked examples
- Common distributions: Bernoulli, Binomial, Normal, Poisson
- The Central Limit Theorem — why it matters

Output: `probability.py` — distribution samplers, Bayes computation, and CLT demonstration

---

**Days 10–12 | Descriptive Statistics** ← *Currently here*

Before any model runs, you need to understand your data. Statistics is not preprocessing — it is how you build intuition about what a dataset is telling you before you touch a model.

Topics:
- Measures of central tendency: mean, median, mode — when to use which
- Measures of spread: variance, standard deviation, IQR
- Skewness and kurtosis — what the shape of a distribution signals
- Correlation and covariance — computed by hand, not with Pandas
- Outlier detection methods

Output: `descriptive_stats.py` — full statistics module, verified against manual calculations

---

**Days 13–15 | Information Theory**

This is the bridge from statistics to machine learning. Loss functions like cross-entropy come directly from information theory. Understanding entropy makes the training objective make sense.

Topics:
- Shannon entropy — definition and intuition
- Kullback-Leibler divergence
- Cross-entropy loss — derived from first principles
- Mutual information

Output: `information_theory.py` — entropy and KL divergence computed from scratch

---

### Phase 2 — Core Machine Learning Algorithms (Days 16–30)

This phase implements the classical ML toolkit. Each algorithm gets implemented twice: first to make it work, then to understand why it works.

---

**Days 16–17 | Data Preprocessing**

Topics: normalization vs. standardization, handling missing values, one-hot encoding, train/val/test splitting — all written without Pandas helpers

---

**Days 18–20 | Linear Regression**

Full OOP implementation. Gradient descent and closed-form solution both. Visualize the loss surface. Understand when each method makes sense and why.

Topics: cost function derivation, normal equation, gradient descent convergence, regularization (Ridge/Lasso from scratch)

---

**Days 21–22 | Logistic Regression**

Classification from first principles. The sigmoid function, binary cross-entropy loss, decision boundaries.

---

**Days 23–24 | K-Nearest Neighbors and K-Means Clustering**

Distance metrics by hand. KNN for classification. K-Means convergence loop. Understand why these are not "learning" in the gradient sense, and what that means.

---

**Days 25–26 | Decision Trees**

Entropy-based splitting, Gini impurity, tree construction from scratch. Then Random Forest: understand what bagging actually does for variance.

---

**Days 27–29 | Support Vector Machines**

The maximum-margin classifier. Hard-margin and soft-margin. Why the kernel trick works. This one is hard — the point is to sit with the difficulty.

---

**Day 30 | Model Evaluation**

Confusion matrix, precision, recall, F1, ROC/AUC — all computed without libraries. A proper train/val/test pipeline written from scratch.

---

### Phase 3 — Neural Networks From Scratch (Days 31–45)

This phase is the core of the whole challenge. Building a neural network without a framework is the clearest way to understand what these systems actually do.

---

**Days 31–32 | The Perceptron**

The original single-layer model. Implement the learning rule. Understand why it fails on XOR. That failure is historically important.

---

**Days 33–35 | Multilayer Perceptron**

Forward pass, loss computation, backward pass. Every weight update computed and logged. Verify it learns something simple before going further.

---

**Days 36–38 | Backpropagation**

This gets its own section because most people use it without understanding it. Derive the update equations from the chain rule. Implement a general version that works for arbitrary depth.

---

**Days 39–40 | Activation Functions**

Sigmoid, tanh, ReLU, Leaky ReLU, GELU — implement each, plot the gradient curves, understand the vanishing gradient problem from the numbers, not from a diagram.

---

**Days 41–42 | Optimizers**

SGD, Momentum, RMSProp, Adam — each built step by step. Run the same network with each and compare convergence. The differences are visible in the loss curves.

---

**Days 43–45 | Regularization**

Dropout (implement the mask, not just the concept), L1/L2 weight penalty, batch normalization. Train with and without, compare results.

---

### Phase 4 — Deep Learning Architectures (Days 46–55)

---

**Days 46–48 | Convolutional Neural Networks**

Convolution operation written manually. Pooling layers. A working CNN that processes images without any deep learning library.

---

**Days 49–51 | Recurrent Networks and LSTMs**

Sequential data processing. The vanishing gradient problem in practice. LSTM gating mechanism — each gate implemented and explained.

---

**Days 52–53 | Attention Mechanism**

Query, key, value — where these come from and what they compute. Scaled dot-product attention from scratch. This is the direct foundation of transformers.

---

**Days 54–55 | Transformer From Scratch**

Multi-head attention, positional encoding, the full encoder block. The goal is not to reproduce GPT — it is to understand why every component is there.

---

### Phase 5 — Research Paper Implementations (Days 56–60)

The final phase. Take three papers and implement the core architecture or algorithm they describe.

Papers to be selected from:
- Attention Is All You Need (Vaswani et al., 2017)
- Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)
- Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)
- Word2Vec (Mikolov et al., 2013)
- Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)

---

## Progress Tracker

| Project | Topic | Status |
|---|---|---|
| 01 | Linear Algebra Engine | ✅ Done |
| 02 | Calculus and Gradients | ✅ Done |
| 03 | Probability Theory | ✅ Done |
| 04 | Descriptive Statistics | 🔄 In Progress |
| 05 | Information Theory | ⏳ Upcoming |
| 06 | Data Preprocessing | ⏳ Upcoming |
| 07 | Linear Regression | ⏳ Upcoming |
| 08 | Logistic Regression | ⏳ Upcoming |
| 09 | KNN and K-Means | ⏳ Upcoming |
| 10 | Decision Trees | ⏳ Upcoming |
| 11 | SVM | ⏳ Upcoming |
| 12 | Model Evaluation | ⏳ Upcoming |
| 13 | Perceptron | ⏳ Upcoming |
| 14 | Multilayer Perceptron | ⏳ Upcoming |
| 15 | Backpropagation | ⏳ Upcoming |
| 16 | Activation Functions | ⏳ Upcoming |
| 17 | Optimizers | ⏳ Upcoming |
| 18 | Regularization | ⏳ Upcoming |
| 19 | CNNs | ⏳ Upcoming |
| 20 | RNNs and LSTMs | ⏳ Upcoming |
| 21 | Attention Mechanism | ⏳ Upcoming |
| 22 | Transformer | ⏳ Upcoming |
| 23 | Research Paper 1 | ⏳ Upcoming |
| 24 | Research Paper 2 | ⏳ Upcoming |
| 25 | Research Paper 3 | ⏳ Upcoming |

---

## How to Navigate This Repo

Each project folder contains:
- `implementation.py` — the core code, written to be read, not just run

---

## What Comes After Day 60

The end goal of this challenge is to move into applied AI safety research with a foundation I actually trust. That means generative AI architecture work, LLM behavior analysis, and eventually contributing to evaluation frameworks or alignment research.

This repository is the proof of work for that path.

---

*Started: [15-05-2026] | Target completion: [End Date]*
*All commits pushed daily as work progresses.*
