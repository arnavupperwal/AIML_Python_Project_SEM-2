# How to Build PrismaTruth AI: A Step-by-Step Educational Guide

Welcome to the comprehensive, step-by-step engineering and design breakdown of **PrismaTruth AI**, a production-grade Fake News Detection System. This guide walks you through the entire project creation process, explaining the rationale behind every design decision, every line of code, and how all components interact.

---

## 🗺️ Project Architecture & Data Flow

PrismaTruth AI is divided into two distinct execution phases: the **Offline Training Pipeline** and the **Online Inference Server**.

```mermaid
graph TD
    %% Configuration
    subgraph Configuration Layer
        config["config.py<br/>(Hyperparameters & Paths)"]
        config_mgr["config_manager.py<br/>(Env-based Settings)"]
    end

    %% Data Input
    subgraph Data Layer
        true_csv[("data/True.csv")]
        fake_csv[("data/Fake.csv")]
    end

    %% Offline Pipeline
    subgraph Training & Evaluation Pipeline (Offline)
        pipeline["data_pipeline.py<br/>(Preprocessing & FeatureUnion)"]
        trainer["model_trainer.py<br/>(GridSearchCV & Cross-Validation)"]
        evaluator["evaluator.py<br/>(Metrics & Heatmaps)"]
        main_notebook["main_notebook.py<br/>(End-to-End Orchestrator)"]
        
        %% Pipeline Outputs
        best_model[("models/best_model.joblib")]
        tfidf_vectorizer[("models/tfidf_vectorizer.joblib")]
        reports[("reports/ Plots & Reports")]
    end

    %% Online / Serving
    subgraph Production Server & UI (Online)
        predictor["predictor.py<br/>(Inference Engine)"]
        api["api.py<br/>(FastAPI REST Endpoints)"]
        streamlit_app["app.py<br/>(Streamlit Console)"]
        frontend["static/ index.html, style.css, script.js"]
    end

    %% Offline Connections
    config --> pipeline
    config --> trainer
    config --> evaluator
    config --> main_notebook
    
    true_csv --> pipeline
    fake_csv --> pipeline
    pipeline -- Split & Vectorized Data --> main_notebook
    main_notebook -- X_train, y_train --> trainer
    trainer -- Best Grid Search Model --> main_notebook
    main_notebook -- Evaluate --> evaluator
    evaluator --> reports
    trainer -- Save Model --> best_model
    trainer -- Save Vectorizer --> tfidf_vectorizer

    %% Online Connections
    best_model --> predictor
    tfidf_vectorizer --> predictor
    predictor --> api
    predictor --> streamlit_app
    
    frontend -- POST /agent --> api
```

---

## 🛠️ Step 1: Setting Up the Dependencies (`requirements.txt`)

To ensure a robust, fast, and framework-independent environment, we compile a lean set of Python libraries in [requirements.txt](file:///c:/Users/Admin/Desktop/ML-project/requirements.txt):

1. **Pandas & NumPy**: For vectorized data frame manipulation and numeric arrays.
2. **scikit-learn**: Powers the machine learning lifecycle (TF-IDF vectorizer, FeatureUnion, GridSearchCV, MultinomialNB, LogisticRegression, RandomForestClassifier, and evaluation metrics).
3. **NLTK (Natural Language Toolkit)**: Automates the download of sentence and word tokenizer resources.
4. **joblib**: Designed to serialize large NumPy arrays and sklearn pipeline objects efficiently.
5. **Matplotlib & Seaborn**: Generates plots on a headless backend.
6. **FastAPI & Uvicorn**: Lightweight ASGI stack for serving inference with high concurrent request processing capability.
7. **Pydantic**: Validates input structure at runtime, throwing 400 Errors if fields are invalid or empty.
8. **python-dotenv**: Separates development configurations from production deployments.
9. **Streamlit**: Used as a fast, alternative dashboard prototype.

---

## ⚙️ Step 2: Configuration & The Twelve-Factor App Design

To prevent developers from hardcoding file directories, hyperparameters, or credentials, we separate configurations into two modules: static configurations (hyperparameters, vocabulary sizes, folds) and environment settings (ports, hosts, model paths).

### 1. Static Configuration Engine: [config.py](file:///c:/Users/Admin/Desktop/ML-project/config.py)
This module establishes the folder directories, model boundaries, and parameter search space.

* **Directory Auto-Creation**:
  ```python
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  DATA_DIR = os.path.join(BASE_DIR, "data")
  MODEL_DIR = os.path.join(BASE_DIR, "models")
  REPORT_DIR = os.path.join(BASE_DIR, "reports")

  for _dir in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
      os.makedirs(_dir, exist_ok=True)
  ```
  *Thought Process*: This guarantees that running any script automatically configures the necessary folders on any OS, preventing `FileNotFoundError`.
* **Hyperparameter Boundaries**:
  * `MAX_FEATURES_TFIDF = 12_000` limits word-level features to prevent overfitting on rare words (e.g. usernames, typos).
  * `CHAR_MAX_FEATURES_TFIDF = 18_000` limits character-level features.
  * `CHAR_NGRAM_RANGE = (3, 5)` defines that 3, 4, and 5-letter character slices are captured to make the classifier resilient against typos and slang.
* **Grid Search Parameter Configurations**: Sets the boundaries for parameter exploration, ensuring we search across regularization strength ($C$), solver penalties, and estimator sizes.

### 2. Environment Configuration Engine: [config_manager.py](file:///c:/Users/Admin/Desktop/ML-project/config_manager.py)
Implements **Twelve-Factor App** design patterns by loading runtime values from environment files (`.env`).
```python
load_dotenv()

class Config:
    APP_NAME = "Fake News Detector"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
    VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/tfidf_vectorizer.joblib")
    STATIC_DIR = os.getenv("STATIC_DIR", "static")

config = Config()
```
*Thought Process*: Keeps API secrets, hosting ports, and debugging modes configurable without modifying files.

### 3. Bootstrap Boot: [setup.py](file:///c:/Users/Admin/Desktop/ML-project/setup.py)
This script is executed once to confirm Python version compliance, download required NLTK corpuses (`stopwords`, `wordnet`, `punkt`), and inspect dataset files (`True.csv`, `Fake.csv`) inside `data/`.

---

## 🧬 Step 3: Engineering the Preprocessing & Feature Union Pipeline (`data_pipeline.py`)

A machine learning model is only as good as the features it receives. The file [data_pipeline.py](file:///c:/Users/Admin/Desktop/ML-project/data_pipeline.py) contains the core mathematical and NLP feature engineering.

### 1. Data Merging & Duplication Trick
```python
if "title" in df.columns and "text" in df.columns:
    combined_content = df["title"].fillna("").str.strip() + " " + df["text"].fillna("").str.strip()
    combined_df = pd.DataFrame({"content": combined_content, "label": df["label"]})

    title_only = df["title"].fillna("").str.strip()
    title_df = pd.DataFrame({"content": title_only, "label": df["label"]})
    df = pd.concat([combined_df, title_df], ignore_index=True)
```
> [!IMPORTANT]
> **Why duplicate the dataset as 'title+text' AND 'title only'?**
> A model trained purely on full articles will fail on short snippets (like headlines) during production because the distribution of word occurrences is completely different. By duplicating each training example—once as the full article text and once as the headline alone—we teach the vocabulary patterns to recognize markers in both long-form and short-form text.

### 2. Stripping Publisher Signatures
```python
def remove_publisher_prefix(text: str) -> str:
    match = re.search(r"^.*? - ", text)
    if match and match.end() < 80:
        return text[match.end():]
    return text
```
*Thought Process*: Real news datasets often contain signatures like `"WASHINGTON (Reuters) - "`. If we leave these signatures, the machine learning models will memorize the text `"Reuters"` or `"Associated Press"` as a 100% guarantee of "Real News". The model would learn to classify the publisher rather than analyzing the linguistic composition of the actual claims.

### 3. Preserving Sensor-Signals in Normalization
```python
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```
> [!TIP]
> **Why do we avoid standard stopword removal and lemmatization?**
> Clickbait and fake news rely heavily on sensationalist framing, punctuation groupings, and specific modal verbs (e.g., *"should", "must", "secretly", "you"*). Removing stopwords strips these patterns. Furthermore, lemmatizing words (e.g., converting *"shocking"* to *"shock"*) destroys emotional intensity indicators. Keeping normalization lightweight preserves these semantic features.

### 4. Seed Injection
```python
seed_df = pd.DataFrame(
    [{"content": text, "label": 1} for text in SEED_REAL_EXAMPLES]
    + [{"content": text, "label": 0} for text in SEED_FAKE_EXAMPLES]
)
```
*Thought Process*: If a user enters a very short text that is completely outside the dataset's vocabulary, prediction scores could collapse or default randomly. Adding representative seed templates guarantees that clean factual formats (real seeds) and hyper-sensational claims (fake seeds) anchor the edges of our vector space.

### 5. Hybrid Feature Extraction via FeatureUnion
Instead of extracting only word frequencies, we implement a parallel extraction pipeline that captures char-level and word-level relationships:
```python
word_tfidf = TfidfVectorizer(
    max_features=max_features,
    ngram_range=ngram_range,
    sublinear_tf=True,
    strip_accents="unicode",
    min_df=2,
    dtype=np.float32,
)
char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=config.CHAR_NGRAM_RANGE,
    max_features=config.CHAR_MAX_FEATURES_TFIDF,
    sublinear_tf=True,
    min_df=2,
    dtype=np.float32,
)
feature_extractor = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf),
])
```
* **Word-level TF-IDF**: Evaluates combinations of single words and word pairs (e.g. *"white house"*, *"secret weapon"*).
* **Character-level TF-IDF (`char_wb`)**: Extracts character substrings of size 3 to 5 within word boundaries.
  * *Thought Process*: Capturing char n-grams makes the model highly resilient to spelling variations, deliberate typos (e.g. *"C0VID"*), and structural prefixes/suffixes common in clickbait framing.
* **`sublinear_tf=True`**: Applies logarithmic scaling $1 + \log(\text{tf})$ to term frequencies. This prevents an article that repeats a word 50 times from dominating the feature weights over an article that mentions it 2 times.

---

## 🤖 Step 4: Model Training, Parameter Grids, & Cross-Validation (`model_trainer.py`)

The file [model_trainer.py](file:///c:/Users/Admin/Desktop/ML-project/model_trainer.py) manages model selection, parameter searches, and artifact serialization.

### 1. The Model Selection Strategy
We train three distinct algorithms, each representing a different mathematical classification approach:
1. **Multinomial Naive Bayes (MNB)**:
   * *How it works*: A probabilistic classifier based on Bayes' Theorem, calculating the joint probability of words given a category.
   * *Role*: A fast baseline with high accuracy on simple frequency distributions.
2. **Logistic Regression (LR)**:
   * *How it works*: Fits a logistic sigmoid function to a linear combination of features, learning positive/negative weights for every word/char n-gram.
   * *Role*: Typically the top performer for high-dimensional sparse text vectors.
3. **Random Forest Classifier (RF)**:
   * *How it works*: An ensemble of decision trees trained on random subsets of features and data.
   * *Role*: Captures non-linear feature interactions (e.g., word A and word B appearing together in a specific structural context).

### 2. GridSearchCV & Stratified K-Fold CV
To ensure we evaluate models accurately, we run grid-search cross-validation:
```python
grid = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    cv=config.CV_FOLDS,  # 5-fold Stratified CV
    scoring=config.SCORING_METRIC,  # Optimized for F1-score
    n_jobs=1,
)
grid.fit(X_train, y_train)
```
* **F1-Score optimization**: We prioritize the F1-score (harmonic mean of Precision and Recall) because accuracy can be misleading if class distributions shift.
* **Stratified Folds**: The training data is split into 5 chunks, ensuring that each chunk maintains the exact same ratio of Real-to-Fake news as the entire dataset. This prevents biased training folds.

---

## 📊 Step 5: Scientific Evaluation (`evaluator.py`)

A production system requires thorough evaluation. The file [evaluator.py](file:///c:/Users/Admin/Desktop/ML-project/evaluator.py) provides diagnostic outputs.

### 1. Headless Plotting
```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
```
> [!NOTE]
> **Why do we use the "Agg" backend?**
> By default, Matplotlib tries to open interactive GUI windows to render charts. In server environments (like Docker containers or headless servers), this causes the script to crash immediately because no display driver exists. Selecting `"Agg"` forces Matplotlib to render plots silently in system memory, allowing us to save figures directly to disk as PNG files.

### 2. Visualization Artefacts
* **Confusion Matrix**: Plots True Positives, False Positives, True Negatives, and False Negatives using a Seaborn heatmap.
* **Model Comparison Chart**: Renders a grouped bar chart comparing all three models side-by-side across Accuracy, Precision, Recall, and F1.

---

## 🔗 Step 6: End-to-End Orchestrator (`main_notebook.py`)

This file ties the entire offline system together. Running `python main_notebook.py` performs the following flow:

```
[Start]
   │
   ├── 1. run_pipeline() ──> Load, clean, split, and vectorize True/Fake CSVs
   │
   ├── 2. train_all_models() ──> Run GridSearch on Naive Bayes, LR, and Random Forest
   │
   ├── 3. evaluate_all() ──> Generate text classification reports & save heatmaps
   │
   ├── 4. plot_model_comparison() ──> Save the comparison bar chart to reports/
   │
   ├── 5. save_artifacts() ──> Persist the winner model & FeatureUnion vectorizer
   │
   └── 6. Demo test run on sample inputs
[End]
```

---

## ⚙️ Step 7: Prediction Engine & API Layer (`predictor.py` and `api.py`)

With our model saved, we build the interfaces to serve predictions.

### 1. Single & Batch Inference Engine: [predictor.py](file:///c:/Users/Admin/Desktop/ML-project/predictor.py)
This module acts as the model wrapper. It loads the persisted serialization files and handles predictions:
```python
def predict_news(text: str, model=None, tfidf=None) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    if model is None or tfidf is None:
        model, tfidf = _load_model_and_vectorizer()

    clean = preprocess_text(text)
    vector = tfidf.transform([clean])
    prediction = model.predict(vector)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        confidence = float(np.max(proba))

    label = "REAL News" if prediction == 1 else "FAKE News"

    return {
        "label": label,
        "confidence": confidence,
        "raw_prediction": int(prediction),
    }
```

### 2. High-Performance API: [api.py](file:///c:/Users/Admin/Desktop/ML-project/api.py)
Built on FastAPI, it loads model assets into memory on startup:
* **`/health`**: Allows monitoring tools to verify the server status and confirm that ML models are successfully loaded in memory.
* **`/predict`**: Returns the raw prediction payload.
* **`/agent`**: Wraps the raw results in the **Agent Response Layer**.

### 3. The Agent Response Layer
Raw numbers like `0.912` or `0.08` can be confusing for general users. The function `build_agent_response` translates raw scores into conversational text and guidelines:
```python
def build_agent_response(text: str, result: dict) -> dict:
    confidence = result.get("confidence")
    confidence_pct = round((confidence or 0.0) * 100, 1)
    is_real = result["raw_prediction"] == 1

    verdict = "likely reliable" if is_real else "likely misleading"
    summary = (
        "The article follows patterns that are common in factual reporting."
        if is_real
        else "The article shows patterns often associated with sensational or unreliable claims."
    )
    # Define custom verification steps depending on the outcome
    if is_real:
        next_steps = [
            "Treat this as a screening signal, not final proof.",
            "Check the publication, author, and date before sharing.",
            "Look for corroboration from multiple established outlets.",
        ]
    else:
        next_steps = [
            "Pause before sharing this claim.",
            "Search for matching reporting from trusted outlets.",
            "Check whether the headline is exaggerated compared with the body text.",
        ]

    excerpt = " ".join(text.split())
    excerpt = excerpt[:220] + ("..." if len(excerpt) > 220 else "")

    return {
        "message": f"I reviewed the article and my current assessment is {verdict}. Model confidence is {confidence_pct}%.",
        "verdict": result["label"],
        "confidence_percent": confidence_pct,
        "summary": summary,
        "excerpt": excerpt,
        "next_steps": next_steps,
    }
```
*Thought Process*: This layer turns a dry classification tool into an active assistant, providing clear context and actionable next steps.

---

## 🎨 Step 8: Frontend Interfaces (`app.py` & `static/`)

The application offers two frontend interfaces:

### 1. The Streamlit Dashboard: [app.py](file:///c:/Users/Admin/Desktop/ML-project/app.py)
A dashboard utilizing Streamlit's reactive UI engine to display predicted outcomes, raw probabilities, and preprocessed text fields.

### 2. Glassmorphic Web App Interface: `static/`
The primary interface is a pure HTML/CSS/JS frontend located in [static/index.html](file:///c:/Users/Admin/Desktop/ML-project/static/index.html).

#### ✨ Visual Design (`style.css`)
We design a clean glassmorphic container layer using modern CSS:
* **Theme & Colors**: Soft background tones (`#f4efe6`) combined with dark text variables (`#13262f`). Vibrant accents are set for warnings/success.
* **Glow Effects**: Fixed blurred background divs (`filter: blur(60px)`) produce soft ambient gradients that make the UI look modern and clean.
* **Glassmorphism**:
  ```css
  .hero-copy, .hero-panel, .workspace-card {
      background: rgba(255, 250, 243, 0.82);
      border: 1px solid rgba(13, 38, 47, 0.12);
      box-shadow: 0 30px 80px rgba(29, 35, 39, 0.12);
      backdrop-filter: blur(18px);
  }
  ```
  This creates a translucent glass effect over the background gradients.
* **Responsive Layout**: Utilizing CSS grid and media queries (`@media (max-width: 960px)`), the layout stacks elements neatly on mobile displays.

#### 🧠 Frontend Logic (`script.js`)
* **State Preservation**: Saves search history in the browser's `localStorage` so users can revisit their previous checks.
* **Quick Chips**: Interactive buttons let users test sample real and fake news snippets instantly.
* **FastAPI Integration**: Submits texts to the `/agent` endpoint via fetch, rendering progress bars and verdict lists dynamically.

---

## 🚀 How to Run the Completed System

Once your folders match this layout, follow these command steps to spin up the system:

1. **Verify or install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Bootstrap and download corpus files**:
   ```bash
   python setup.py
   ```
3. **Train models and generate verification charts**:
   ```bash
   python main_notebook.py
   ```
4. **Launch the FastAPI Server**:
   ```bash
   python api.py
   ```
   Now navigate your browser to `http://127.0.0.1:8000` to interact with your system.
