# ml-implementation-from-scratch

> **60 ML/AI projects built from the ground up — no black-box libraries, just Python and math.**

A structured 60-day public challenge covering the full stack of modern machine learning: from vector algebra and probability theory, through classic ML algorithms, to deployed GenAI systems and paper re-implementations.

Every project is written from scratch to build genuine intuition before reaching for abstractions like NumPy, PyTorch, or LangChain.

---

## Philosophy

Most ML courses hand you a `.fit()` call and call it a day. This repo does the opposite — implement first, then compare against the library to understand what you just built. The goal is to walk away understanding *why* backprop works, *why* attention scales quadratically, and *why* SVMs maximize the margin — not just how to import them.

---

## Structure

The 60 projects are organized into five phases of increasing complexity.

### Phase 1 — Math Foundations `#01–10`

The linear algebra, probability, and optimization bedrock that every ML algorithm is built on.

| # | Project | Key concepts |
|---|---|---|
| **01** | [Vector operations library](./01-vector/) | Dot product, norms L1/L2/L∞, projections, angles |
| 02 | Matrix operations library | Multiply, transpose, determinant, Gaussian elimination |
| 03 | Eigenvalues & eigenvectors | Power iteration, characteristic polynomial, diagonalization |
| 04 | SVD module | Full decomposition, low-rank approximation, image compression |
| 05 | Probability distributions | Gaussian, Bernoulli, Binomial, Poisson — PDF, CDF, sampling |
| 06 | Descriptive statistics | Mean, variance, covariance matrix, skewness, kurtosis |
| 07 | Hypothesis testing | t-test, chi-square, ANOVA, p-values, confidence intervals |
| 08 | Bayesian inference | Prior, likelihood, posterior, conjugate distributions, MAP |
| 09 | Information theory | Entropy, cross-entropy, KL divergence, mutual information |
| 10 | Optimization module | Gradient descent, momentum, RMSProp, Adam with convergence plots |

### Phase 2 — ML Algorithms `#11–25`

Classic algorithms implemented without scikit-learn — then benchmarked against it.

| # | Project | Key concepts |
|---|---|---|
| 11 | Linear regression | OLS + gradient descent, R², Ridge/Lasso |
| 12 | Logistic regression | Sigmoid, binary cross-entropy, softmax multi-class |
| 13 | Decision tree | CART, Gini/entropy splitting, pruning, feature importance |
| 14 | Random forest | Bagging, feature subsampling, OOB error |
| 15 | Naive Bayes | Gaussian NB, Multinomial NB, Laplace smoothing |
| 16 | K-Nearest Neighbors | KD-tree, cross-validation, k selection |
| 17 | SVM | Hard/soft margin, hinge loss, RBF kernel |
| 18 | K-Means clustering | Lloyd's algorithm, K-Means++ init, silhouette score |
| 19 | PCA | Covariance → SVD, variance explained, 2D visualization |
| 20 | Backprop MLP | Forward/backward pass, Xavier init, mini-batch training |
| 21 | CNN | Conv layer, pooling, full backprop — no PyTorch, tested on MNIST |
| 22 | RNN + LSTM ⚡ | Vanishing gradient, gating mechanism, character-level LM |
| 23 | Gradient boosting | Residual fitting, shrinkage, vs XGBoost |
| 24 | Attention mechanism ⚡ | Scaled dot-product, multi-head, positional encoding |
| 25 | Minigrad autograd ⚡ | Computation graph, Value class, reverse-mode autodiff |

### Phase 3 — Real-World Projects `#26–40`

End-to-end pipelines on real datasets — EDA, feature engineering, evaluation.

| # | Project | Domain |
|---|---|---|
| 26 | Iris species classifier | Tabular |
| 27 | Titanic survival prediction | Tabular |
| 28 | House price regression | Tabular |
| 29 | Spam email classifier ⚡ | NLP |
| 30 | Twitter sentiment analysis ⚡ | NLP |
| 31 | MNIST digit classifier | Vision |
| 32 | CIFAR-10 image classifier | Vision |
| 33 | Customer churn prediction | Tabular |
| 34 | Movie recommendation system | Tabular |
| 35 | Named entity recognition ⚡ | NLP |
| 36 | Fake news detector | NLP |
| 37 | Credit card fraud detection | Tabular |
| 38 | Time series forecasting | Tabular |
| 39 | Topic modeling with LDA ⚡ | NLP |
| 40 | Image segmentation (U-Net lite) | Vision |

### Phase 4 — Advanced + Deployed `#41–52`

Production-grade systems: APIs, Docker, HuggingFace Spaces, and live deployments.

| # | Project | Stack |
|---|---|---|
| 41 | Text classification API ⚡ | FastAPI + DistilBERT + Docker |
| 42 | Image classifier web app | CNN + Streamlit |
| 43 | BERT sentiment analyzer ⚡ | Fine-tuned BERT + HuggingFace Hub |
| 44 | YOLOv8 object detection app | Real-time + Gradio |
| 45 | RAG chatbot — PDF Q&A ⚡ | LangChain + FAISS + Streamlit |
| 46 | MLflow experiment tracker | Full pipeline + model registry |
| 47 | Anomaly detection API | Isolation Forest + FastAPI |
| 48 | Multi-label text classifier ⚡ | BCELoss + REST API |
| 49 | Image captioning system | ResNet + LSTM + BLEU |
| 50 | Stock dashboard (deployed) | LSTM + Plotly + Streamlit |
| 51 | Kaggle competition pipeline | Full EDA → stacking ensemble |
| 52 | Resume parser app ⚡ | spaCy NER + Streamlit |

### Phase 5 — Paper Implementations `#53–60`

Landmark ML papers reimplemented from the original equations.

| # | Paper | Key technique |
|---|---|---|
| 53 | Word2Vec ⚡ | Skip-gram + negative sampling |
| 54 | Transformer ⚡ | "Attention Is All You Need" — full encoder-decoder |
| 55 | Original GAN | Goodfellow 2014 — MNIST generation |
| 56 | ResNet | Skip connections, identity shortcuts |
| 57 | Batch normalization | Internal covariate shift, ablation study |
| 58 | Dropout | Srivastava 2014 — inverted dropout, ensemble interpretation |
| 59 | Mini-BERT ⚡ | Masked LM + NSP, from-scratch tokenizer |
| 60 | LoRA fine-tuning ⚡ | Low-rank adaptation, rank ablation |

> ⚡ = GenAI-relevant project

---

## Progress

| Phase | Done | Total |
|---|---|---|
| P1 Math Foundations | 1 | 10 |
| P2 ML Algorithms | 0 | 15 |
| P3 Real-World Projects | 0 | 15 |
| P4 Advanced + Deployed | 0 | 12 |
| P5 Paper Implementations | 0 | 8 |
| **Total** | **1** | **60** |

---

## Requirements

Each project folder is self-contained with its own dependencies. The early math projects use only the Python standard library. Later projects introduce NumPy, PyTorch, HuggingFace Transformers, FastAPI, and LangChain progressively.

```
Python 3.10+
```

Phase-specific requirements are listed in each project's folder.

---

## About

Built by **Hamid Raza Bajwa** as a 60-day public challenge documenting a transition into ML/GenAI engineering.

- GitHub: [github.com/hamidrazabajwa49](https://github.com/hamidrazabajwa49)
- LinkedIn: [linkedin.com/in/hamid-raza-bajwa-564a91377](https://linkedin.com/in/hamid-raza-bajwa-564a91377)
- Email: hamidrazabajwa49@gmail.com

---

*Inspired by Andrej Karpathy's micrograd and the broader "implement before you import" philosophy.*
