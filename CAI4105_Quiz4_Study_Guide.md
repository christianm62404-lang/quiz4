# CAI 4105 — Quiz 4 Study Guide
**Quiz 4: Tuesday, in class on Webcourses. Multiple choice + short-answer / free response. Covers Ensemble Learning & Boosting, Unsupervised Learning (Clustering, Hierarchical, DBSCAN), NLP, and the Neural Network module — *except CNN, which is covered after this quiz*. (CNN is included at the end as post-quiz reference.)**

This quiz rewards *concept recall + one-step reasoning*: "which method / which loss / which fails," a small worked clustering step, a convolution output-size, a softmax. Prioritize in this order:

1. The **conceptual distinctions** that get reused as MC stems (bagging vs boosting, hard vs soft voting, k-means vs DBSCAN, stemming vs lemmatization, sigmoid vs softmax).
2. The **one-step computations**: one k-means assignment+update, a k-modes dissimilarity, an agglomerative merge, a softmax, a CNN output dimension.
3. The **evaluation / data-prep** ideas (imbalanced accuracy, precision/recall, one-vs-rest, data leakage, stratified sampling) — these show up as short free-response.
4. The **"why"** behind each fix (why RF reduces overfitting, why DBSCAN beats k-means on odd shapes, why ReLU beats sigmoid in hidden layers) — know the shape of the argument, not a derivation.

---

## Part 0 — The cheat sheet

Cover the right column and recite these.

| Thing | Key fact |
|---|---|
| Ensemble idea | Aggregate many predictors → beats the best single one ("wisdom of the crowd"); works best when predictors are **diverse/independent** |
| Hard vs soft voting | Hard = majority class (mode); Soft = average **probabilities**, pick highest (needs `predict_proba`; SVC needs `probability=True`) |
| Bagging vs pasting | Bagging = sample **with** replacement; Pasting = **without** (`bootstrap=False`); both parallel |
| OOB evaluation | Score each instance with only the trees that **didn't** sample it (~37% are out-of-bag) — free validation |
| Random Forest | Bagged trees + **random feature subset per split**; predict by majority vote / average; averaging **lowers variance** |
| Feature importance | Avg impurity reduction from nodes using that feature; scaled to sum to 1 |
| Boosting | Train predictors **sequentially**, each corrects the last; weak → strong; **can't fully parallelize** |
| AdaBoost | **Up-weight misclassified** instances (start 1/m); predict by **weighted** vote (better models weigh more, α) |
| Gradient Boosting | Fit each new model to the **residual errors** of the previous |
| Supervised vs unsupervised | Unsupervised has only X, **no labels y** |
| Centroid | Per-attribute **mean**; may be an **imaginary** point not in the data |
| K-means loop | Assign to nearest centroid → recompute centroids (mean) → repeat **until centroids stop moving** |
| WCSS / inertia | Σ squared distances of points to their centroid; **Elbow** = plot WCSS vs K, pick the bend |
| K-modes | Categorical: dissimilarity = **# mismatches** (match 0, differ 1); update leaders by **mode** |
| Hierarchical | Tree of clusters, no preset K; **dendrogram**; Agglomerative (bottom-up) vs Divisive (top-down) |
| Linkages | Single=**min**, Complete=**max**, Average=**mean**, Centroid=**centroid gap**, Ward=**min SSE increase** |
| Dendrogram cut | # clusters = # vertical lines a horizontal threshold crosses |
| DBSCAN params | **eps (ε)** = neighborhood radius; **minPts** = min points for a dense region |
| DBSCAN point types | **Core** (≥minPts within ε), **Border** (near a core), **Noise** (neither); handles arbitrary shapes, no K |
| Bag of Words | Vector of **word counts**, ignores order/grammar; sparse; build with `CountVectorizer` |
| TF-IDF | Term Freq × Inverse Doc Freq; **down-weights common** words, up-weights distinctive ones |
| Stemming vs lemmatization | Stem = crude chop (may be non-word: studies→studi); Lemma = valid base (studies→study); lemmatizer needs **POS** |
| Affixes | Inflectional = form/tense, same class (cats); Derivational = new word/class (happiness) |
| Perceptron | One neuron: **z = b + Σ xᵢwᵢ → φ(z)**; bias shifts activation |
| Deep NN | **≥ 2 hidden layers** |
| Activation purpose | Add **non-linearity** + bound outputs |
| Activation ranges | Step {0,1}; Sigmoid (0,1); Tanh (−1,1); **ReLU max(0,x), [0,∞)** |
| Sigmoid | σ(z)=1/(1+e^−z); σ(0)=0.5; **vanishing gradient** for strong negatives → output layer, not hidden |
| Multi-class output | Exclusive → **softmax** (sum to 1); multi-label → **sigmoid** per node |
| Cost functions | Classification → **cross-entropy**; regression → **MSE** |
| C(W,B,Sr,Er) | W weights, B biases, Sr a sample's input, Er its desired output |
| Backprop | Go back and **adjust weights & biases** from the loss |
| sklearn NN | `MLPClassifier` / `MLPRegressor`; layers via `hidden_layer_sizes=(100,50)`; **Adam** = Adaptive Moment Estimation |
| CNN conv output | **⌊(W − F + 2P)/S⌋ + 1** |
| CNN pool output | **⌊(input − pool)/stride⌋ + 1** |
| Imbalanced accuracy | High accuracy by predicting the majority = useless → use **recall / precision / F1 / confusion matrix** |
| Multiclass w/ binary model | **One-vs-Rest** (N models) or **One-vs-One** (N(N−1)/2 models) |
| Data leakage | Never test on rows used in training → **split first**; use validation / k-fold |
| Stratified sampling | Preserve class proportions in train/test for imbalanced data (`stratify=y`) |

---

## Part 1 — Ensemble Learning, Random Forests & Boosting

### 1.1 The core idea
- **Ensemble learning** aggregates the predictions of a group of predictors (an **ensemble**). The aggregate is usually better than the best single predictor — the **"wisdom of the crowd."**
- Ensembles work best when predictors are **as independent/diverse as possible** — different algorithms make different errors that cancel out.

### 1.2 Voting classifiers
- **Hard voting:** predict the class with the **most votes** (the statistical mode).
- **Soft voting:** average the predicted **probabilities** across classifiers and pick the highest. Usually beats hard voting because it **weights confident votes** more. Requires every classifier to expose `predict_proba()`; for `SVC` you must set `probability=True`.
- Example: LR ≈ 0.864, RF ≈ 0.896, SVC ≈ 0.888 → hard-voting ensemble ≈ 0.904; soft voting ≈ 0.912.

### 1.3 Bagging vs pasting
- **Bagging** (bootstrap aggregating) = sampling **with** replacement. **Pasting** = **without** replacement (`bootstrap=False`).
- **Only bagging** lets the same instance be sampled several times **for the same predictor**.
- Both train and predict **in parallel** (`n_jobs=-1` = all cores) → scale well.

### 1.4 Out-of-Bag (OOB) evaluation
- With bagging, on average ~37% of instances are never drawn for a given predictor.
- **OOB error** = the mean prediction error on each instance, computed using **only the trees that did not have that instance** in their bootstrap sample — a free validation set, no separate hold-out needed.

### 1.5 Random Forest — vs a single Decision Tree
| Aspect | Decision Tree | Random Forest |
|---|---|---|
| Structure | one tree | ensemble of many trees |
| Training | full data, once | bagging: each tree on a random resample |
| Split | best of **all** features | best of a **random subset** of features |
| Predict | one tree decides | **majority vote** (class) / **average** (regression) |
| Variance / overfit | high | **low** (averaging reduces variance) |

- **Feature importance:** measured by how much the nodes using a feature reduce impurity on average, scaled so importances **sum to 1** (`feature_importances_`).
- Two RF parameters to name: `n_estimators` (number of trees), `max_leaf_nodes` / `max_features` (tree size / features per split).

### 1.6 Boosting
- **Boosting** trains predictors **sequentially**, each one correcting its predecessor — turning **weak learners into a strong learner**.
- **AdaBoost:** increase the **weight of misclassified** instances so the next predictor focuses on the hard cases. Instance weights start at `1/m`; each predictor earns a weight **α** by accuracy. Prediction = **weighted majority vote** (better models weigh more). **Drawback: sequential → can't be parallelized**, scales worse than bagging. Overfitting? reduce `n_estimators` or regularize the base estimator.
- **Gradient Boosting:** instead of re-weighting instances, fits each new predictor to the **residual errors** of the previous one. `learning_rate` shrinks each tree's contribution (trades off with `n_estimators`).

### 1.7 Bagging vs boosting in one line
- **Bagging** trains predictors **independently / in parallel** on random resamples → reduces **variance**.
- **Boosting** trains them **sequentially**, each fixing the last's mistakes → reduces **bias**.

---

## Part 2 — Unsupervised Learning: K-Means & K-Modes

### 2.1 Supervised vs unsupervised
- **Supervised** = labeled data (X **and** y). **Unsupervised** = only X, **no labels**. Most real-world data is unlabeled.
- Four unsupervised algorithms: **K-Means, Hierarchical clustering, DBSCAN, PCA** (also anomaly detection, ICA, Apriori).

### 2.2 Clustering vs classification
- **Clustering** groups similar instances into clusters (dissimilar across clusters) **without labels**; classification assigns to **predefined labeled** classes.
- Use cases: customer segmentation, document/news grouping, medical/financial grouping, crime/anomaly analysis.
- **Three categories:** partition-based (k-means), hierarchical, density-based (DBSCAN).

### 2.3 Centroid
- The **centroid** is the point whose each attribute equals the **mean** of that attribute over the cluster's points.
- It is **frequently an imaginary point not in the dataset**. Uses Euclidean distance (one-hot encode categoricals).

### 2.4 K-Means algorithm
1. Choose **K** and pick K initial centroids.
2. Build a distance matrix; assign each point to its **nearest centroid**.
3. Recompute each centroid = **mean** of its assigned points.
4. **Repeat until the centroids no longer move** (termination). K-means is **exclusive** — each point in exactly one cluster.

### 2.5 Worked one iteration (memorize the pattern)
Points **A=2, B=4, C=10, D=12**, K=2, initial centroids **c1=2, c2=4**.
- Assign: A→c1 (0<2); B→c2 (0<2); C→c2 (6<8); D→c2 (8<10). → C1={2}, C2={4,10,12}.
- Update: c1 = 2; c2 = (4+10+12)/3 = **26/3 ≈ 8.67**.
- Next pass would re-assign with c1=2, c2≈8.67 (now B=4 joins c1), and repeat until stable.

### 2.6 Choosing K — the Elbow method
- **WCSS** (Within-Cluster-Sum-of-Squares, a.k.a. **inertia**) = Σ squared distances of points to their centroid.
- Run k-means for K = 1…10, plot **WCSS vs K**, and pick K at the **"elbow"** — the sharp bend where extra clusters stop helping.

### 2.7 Evaluating clusters
- **External:** compare to ground truth (if available).
- **Internal:** average within-cluster distance, or average distance of points to their centroid (smaller = tighter).

### 2.8 K-Modes (categorical data)
- Euclidean distance/means don't fit categorical data. K-Modes uses **dissimilarity = number of mismatches** (match → 0, mismatch → 1) and updates leaders by the **mode** (most frequent value per feature).
- Loop: pick K leaders → assign each observation by **least dissimilarity** → recompute modes → repeat until no reassignment.

---

## Part 3 — Hierarchical Clustering & DBSCAN

### 3.1 Hierarchical clustering
- Builds a **hierarchy (tree) of clusters**, visualized with a **dendrogram**. **No preset K** — you cut the dendrogram at a threshold.
- **Agglomerative** (bottom-up: start with each point alone, merge closest pair — more popular) vs **Divisive** (top-down: split one big cluster).

### 3.2 Agglomerative steps
1. **Step 0:** build the **proximity matrix** (n×n, **diagonal = 0**, e.g. Euclidean distances).
2. Find the **smallest distance** and merge those two clusters.
3. Update the matrix (via a chosen linkage), repeat until one cluster remains.

### 3.3 Worked first merge
Distances d(1,2)=3, d(1,3)=7, d(2,3)=6.
- Smallest = 3 → merge {1,2}.
- **Complete linkage** (max): d({1,2},3) = max(7,6) = **7**. (Single linkage would take min = 6.)

### 3.4 Linkage types (distance between clusters)
- **Single** = minimum cross-cluster distance · **Complete** = maximum · **Average** = mean of all cross-cluster distances · **Centroid** = distance between centroids · **Ward** = merge causing the **smallest increase in within-cluster variance (SSE)**.

### 3.5 Dendrogram → number of clusters
- Taller vertical links = larger distance between merged clusters.
- Draw a horizontal **threshold** line; the number of clusters = the number of **vertical lines it crosses**.

### 3.6 Why DBSCAN? (K-means weaknesses)
- K-means **forces every point into a cluster** (including outliers), is sensitive to a single point / the choice of K, and fails on odd shapes.
- **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) finds **arbitrary-shaped** clusters, **separates outliers as noise**, and needs **no K**.

### 3.7 DBSCAN parameters & point types
- **eps (ε):** the neighborhood **radius**. Chosen via a **k-distance graph** elbow.
- **minPts:** minimum points to form a **dense** region. Rule of thumb ≥ D+1, often 2×dim, at least 3.
- **Core** = ≥ minPts within ε · **Border** = fewer than minPts but within ε of a core point · **Noise** = neither (an outlier).

---

## Part 4 — Natural Language Processing

### 4.1 Challenges of NLP
- Language is **ambiguous** (multiple meanings), **context-dependent**, full of **sarcasm, synonyms, slang, and spelling variation**, and differs across languages.

### 4.2 Applications
- Sentiment analysis, machine translation, chatbots, spam detection, NER, text summarization, speech recognition.

### 4.3 Bag of Words (BoW)
- Represents text as a vector of **word counts** over a vocabulary, ignoring grammar and order.
- Example: "I love NLP" + "NLP loves me" → vocab {I, love, NLP, loves, me}, per-sentence counts.
- **Limitations:** loses **word order & context** (so "dog bites man" = "man bites dog"), no semantics, large **sparse** vectors.
- Build it in Python with **scikit-learn's `CountVectorizer`** (or NLTK).

### 4.4 TF-IDF
- **Term Frequency – Inverse Document Frequency.** Weights each term by how often it appears in a document (**TF**) against how rare it is across all documents (**IDF**).
- Purpose: **down-weight common words**, **up-weight distinctive** ones, so informative terms dominate.

### 4.5 Preprocessing vocabulary
- **Tokenization** — split text into tokens/words.
- **Stop words** — common words (the, is, a), usually removed.
- **Stemming** — crude rule-based chop to a root; may not be a real word (`studies → studi`, `caring → car`).
- **Lemmatization** — reduce to a **valid dictionary base form / lemma** using vocabulary + morphology (`studies → study`, `better → good`). Works best when given the word's **part of speech (POS)**.
- **NER** — Named Entity Recognition: tag names, places, organizations.

### 4.6 Stemmer vs lemmatizer
- Stemmer: fast, rule-based, may produce non-words. Lemmatizer: returns a valid word but needs the **POS** to disambiguate.

### 4.7 Inflectional vs derivational affixes
- **Inflectional** change grammatical form (tense/number) **without changing word class**: cat→cats, walk→walked.
- **Derivational** create a **new word / meaning / class**: happy→happi**ness**, care→care**ful**.

---

## Part 5 — Neural Networks (ANN)

### 5.1 The perceptron
- A **perceptron** is a single neuron / single-layer network with **one output**: weighted sum + bias, then activation.
  - **z = b + Σ (xᵢ·wᵢ)   →   output = φ(z)**
- **Weights** = strength of each input; **bias b** shifts the activation up/down (and lets the neuron respond when inputs are 0). Adjusting weights = "learning."
- Stacking perceptrons in layers = a **multi-layer perceptron (basic ANN)**.

### 5.2 Layers
- **Input** = raw feature values. **Hidden** = between input and output, hard to interpret. **Output** = final estimate (can have multiple neurons).
- **Deep Neural Network** = **2 or more hidden layers** (non-deep = 0–1).

### 5.3 Activation functions
- **Purpose:** introduce **non-linearity** and **bound outputs** (e.g. 0–1 probabilities), enabling complex relationships.

| Function | Range | Notes |
|---|---|---|
| Binary step | {0, 1} | threshold; small input changes not reflected |
| Sigmoid (logistic) | (0, 1) | for probability; **vanishing gradient** for strong negatives → **output layer**, not hidden |
| Tanh | (−1, 1) | like sigmoid but better; strong negatives → negative outputs |
| **ReLU** = max(0, x) | [0, ∞) | **most used in hidden layers**; no vanishing gradient on positive side, cheap, sparse; negatives → 0 |

### 5.4 Multi-class outputs
- **Mutually exclusive** classes → **Softmax** (probabilities sum to 1).
- **Non-exclusive / multi-label** → **Sigmoid** per output node (independent probabilities).

### 5.5 Cost functions & backpropagation
- **Regression** → Mean Squared Error (MSE). **Classification** → **Cross-Entropy / log loss** (Binary Cross-Entropy for 2 classes).
- **C(W, B, Sr, Er):** W = weights, B = biases, **Sr** = input of a single training sample, **Er** = its desired output.
- **Backpropagation** = going back through the network to **adjust weights and biases** to reduce the loss (used with gradient descent / Adam).

### 5.6 Neural networks in sklearn
- `MLPClassifier` (classification), `MLPRegressor` (regression).
- Architecture via **`hidden_layer_sizes`** — e.g. `(100, 50)` = two hidden layers of 100 and 50 nodes (length = # hidden layers, each value = nodes).
- **Adam = Adaptive Moment Estimation**, an advanced adaptive version of SGD. Optimizers update weights via `New = Old − η·∇Loss` (η = learning rate).

---

## Part 6 — Model Evaluation & Data Prep (short free-response favorites)

These general ideas apply across the classification topics above and appear as short-answer scenarios.

### 6.1 The accuracy trap on imbalanced data
- If 96% of patients are healthy, a model that predicts "healthy" for **everyone** scores **96% accuracy** while catching **zero** sick patients. **Accuracy is misleading on imbalanced data.**
- Better metrics (from the **confusion matrix**): **Recall/Sensitivity** (of actual positives, how many caught), **Precision** (of flagged, how many are truly positive), **F1** (harmonic mean), **ROC-AUC / PR-AUC**.
- To catch **all** positives: lower the decision **threshold** / oversample / set `class_weight`. **Drawback:** recall → ~100% but **precision collapses** (flood of false positives) — the **precision–recall trade-off**.

### 6.2 Multiclass with a binary-only classifier
- **One-vs-Rest (OvR):** train **N** binary classifiers (each: "class k" vs "the rest"); predict the class whose classifier scores highest.
- **One-vs-One (OvO):** train **N(N−1)/2** classifiers (one per pair); the class with the most votes wins. (5 drugs → 5 OvR models or 10 OvO models.)

### 6.3 Data leakage
- Training on all rows then testing on **some of those same rows** is **data leakage** — the model has seen them, so the score is inflated. **Fix:** split **first**; train only on the training portion; evaluate on a held-out test set; use a validation set or **k-fold cross-validation** for tuning.

### 6.4 Stratified sampling
- With imbalanced classes (e.g. 900 vs 100), **simple random** splitting can leave few/no minority cases in the test set. **Stratified sampling** preserves class proportions in both train and test (`train_test_split(..., stratify=y)`).

---

## Part 7 — CNN (post-quiz reference — covered *after* Tuesday)

*Included for completeness; the quiz does not cover CNN.*

- **Why not a plain ANN on images?** Flattening (28×28 → 784) **destroys spatial information**; fully connected layers create a **huge number of parameters** (784→128 = 100K+); no feature reuse; **not translation invariant**. A color image has **3 channels** (RGB).
- **Convolution:** slide a **kernel/filter** over the image, **element-wise multiply** the overlap, **sum** → one value; the output is a **feature map** (each value = how strongly the filter matched). Weight sharing → fewer parameters, detection anywhere (**translation invariance**).
- **Stride** = step size (larger → smaller output). **Padding** = add zeros around the border so the kernel fits / to preserve size (`valid` = none, `same` = keep size).
- **Output-size math:**
  - Conv: **⌊(W − F + 2P)/S⌋ + 1**. Example: 6×6 input, 3×3 filter, stride 1, no pad → **4×4**; 32×32 with 5×5, stride 1 → 28×28.
  - Pool: **⌊(input − pool_size)/stride⌋ + 1**. Example: 28×28 with 2×2 stride 2 → 14×14.
- **Pooling:** down-samples the feature map (max = window maximum, average = mean); reduces computation, **reduces overfitting** (keeps dominant features), adds translation robustness.
- **Flatten:** turn feature maps into a **1D vector** for the Dense (fully connected) layer (ends in softmax).
- **Dropout:** randomly **turn off** a fraction p of neurons each training step (e.g. `Dropout(0.25)`) → less overfitting, more general patterns.

---

## Part 8 — High-value traps & distinctions

Where short MC/free-response questions live.

1. **Diversity is the point of an ensemble** — independent predictors make different errors that cancel.
2. **Hard voting = mode; soft voting = averaged probabilities** (needs `predict_proba`, SVC `probability=True`).
3. **Bagging = with replacement; pasting = without.** Only bagging repeats an instance within one predictor.
4. **Bagging reduces variance (parallel); boosting reduces bias (sequential).** Boosting can't fully parallelize.
5. **AdaBoost re-weights instances; Gradient Boosting fits residuals.**
6. **Unsupervised = no labels.** Clustering discovers groups; classification uses known labels.
7. **A centroid can be an imaginary point** not in the data.
8. **K-means termination = centroids stop moving.** WCSS + **elbow** picks K.
9. **K-modes = mismatches + mode** (for categorical data).
10. **Hierarchical needs no preset K** — cut the dendrogram; # clusters = vertical lines crossed.
11. **Single = min, Complete = max, Average = mean, Centroid = centroid gap, Ward = min SSE increase.**
12. **DBSCAN: eps + minPts; core/border/noise; arbitrary shapes, no K, labels outliers.**
13. **BoW ignores order/context** and is sparse; **TF-IDF** re-weights by rarity.
14. **Stemming = crude chop (maybe non-word); lemmatization = valid word, needs POS.**
15. **Inflectional keeps word class; derivational makes a new word/class.**
16. **Perceptron = z = b + Σxw → activation.** Deep = ≥ 2 hidden layers.
17. **Activation adds non-linearity.** Sigmoid vanishes for strong negatives → output layer; **ReLU** rules the hidden layers.
18. **Softmax = exclusive (sums to 1); sigmoid-per-node = multi-label.**
19. **Classification loss = cross-entropy; regression = MSE.** Backprop updates weights & biases.
20. **`MLPClassifier` layers via `hidden_layer_sizes`; Adam = Adaptive Moment Estimation.**
21. **Accuracy hides failure on imbalanced data** — use recall / precision / F1.
22. **Multiclass with a binary model → One-vs-Rest (N) or One-vs-One (N(N−1)/2).**
23. **Never test on training rows (data leakage); split first, use k-fold.**
24. **Stratified sampling preserves class ratios** for imbalanced splits.

---

## Part 9 — Self-test (closed-book; answers below)

1. What is the difference between bagging and pasting, and which one allows the same instance to be sampled twice for the same predictor?
2. You have three ~80%-accurate classifiers. How do you combine them, and what's the difference between hard and soft voting?
3. How does AdaBoost differ from Gradient Boosting during training?
4. Name the three categories of clustering algorithms and one example of each.
5. Points A=2, B=4, C=10, D=12, K=2, initial centroids c1=2, c2=4. Do one k-means iteration: give the clusters and the updated centroids.
6. In K-modes, what is the dissimilarity between two observations, and how are cluster leaders updated?
7. Distances d(1,2)=3, d(1,3)=7, d(2,3)=6. Do the first agglomerative merge and give the updated distance to the merged cluster under complete linkage.
8. List the two DBSCAN parameters and define core, border, and noise points.
9. What is the main limitation of Bag of Words, and what does TF-IDF do about common words?
10. Give one example each of stemming vs lemmatization, and state what extra info a lemmatizer needs.
11. Write the perceptron's computation, and say what a "deep" network means.
12. Which activation is used for hidden layers and why; which for a mutually-exclusive multi-class output?
13. A rare-disease model predicts "No Disease" for everyone and reports 96% accuracy. Why is accuracy misleading, and which metrics would you use instead?
14. You must predict one of 5 drugs using a binary-only classifier. Describe One-vs-Rest.
15. A team trains on all 1,000 rows then tests on 200 of them. What's wrong, and how do you evaluate correctly on an imbalanced dataset?

### Answers

1. **Bagging** samples **with** replacement, **pasting without**. Only **bagging** can draw the same instance twice for one predictor.
2. Combine them in a **voting classifier**; it can beat the best single model when they're diverse. **Hard** = majority class; **soft** = average predicted probabilities and pick the highest (needs `predict_proba`).
3. **AdaBoost** re-weights instances (up-weights the misclassified) each round; **Gradient Boosting** fits each new model to the **residual errors** of the previous.
4. **Partition-based** (k-means), **Hierarchical** (agglomerative/divisive), **Density-based** (DBSCAN).
5. C1={2}, C2={4,10,12}; updated c1 = **2**, c2 = 26/3 ≈ **8.67**.
6. Dissimilarity = **number of mismatched attributes** (match 0, differ 1); leaders are updated by taking each feature's **mode**.
7. Merge {1,2} (smallest = 3). Complete linkage: d({1,2},3) = max(7,6) = **7**.
8. **eps (ε)** = neighborhood radius; **minPts** = min points for a dense region. **Core** ≥ minPts within ε; **Border** < minPts but within ε of a core; **Noise** = neither.
9. BoW **loses word order and context** (and is sparse). **TF-IDF** down-weights words common across all documents and up-weights distinctive ones.
10. Stemming: `studies → studi` (crude). Lemmatization: `studies → study` / `better → good` (valid word). The lemmatizer needs the word's **part of speech (POS)**.
11. **z = b + Σ xᵢwᵢ, then output = φ(z).** "Deep" = **2 or more hidden layers**.
12. **ReLU** for hidden layers — no vanishing gradient on the positive side, cheap, sparse. **Softmax** for a mutually-exclusive multi-class output (probabilities sum to 1).
13. Because the data is **imbalanced**, predicting the majority scores high while catching **no** positives. Use **recall/sensitivity, precision, F1**, and the **confusion matrix** (PR-AUC/ROC-AUC).
14. **One-vs-Rest:** train **5 binary classifiers**, each "drug k vs the rest"; for a new patient run all 5 and pick the drug whose classifier gives the **highest score**. (OvO would train 10 pairwise classifiers and take the majority vote.)
15. Testing on training rows is **data leakage** — the score is inflated. **Fix:** split first, train only on the training portion, and evaluate on a held-out test set (use **k-fold cross-validation**); for the imbalance, use **stratified sampling** so both sets keep the class proportions.

---

## Part 10 — Last-pass plan

**If you have a few hours:** work every short-answer question in each module (Ensemble, Clustering, Hierarchical/DBSCAN, NLP, Neural Networks) with the answer covered, then do the interactive quiz sections and re-do any you miss.

**If you have one hour:** Part 0 cheat sheet → Part 8 traps → Part 9 self-test.

**If you have fifteen minutes:** memorize — bagging vs boosting (variance vs bias), hard vs soft voting, the k-means loop + elbow, the linkage list, DBSCAN's eps/minPts and core/border/noise, stemming vs lemmatization, the activation ranges (ReLU for hidden, softmax for exclusive multi-class), cross-entropy vs MSE, and the accuracy-on-imbalanced-data trap (use recall/precision/F1).
