# Primary Topic Classification Application

A machine learning application for classifying the primary topic of tweets using both traditional ML and transformer-based approaches.

## Overview

This application provides a production-ready topic classifier that can predict the primary topic of tweets from a predefined set of categories. It uses the same `tweets_v1.csv` dataset as the sentiment classifier but focuses on topic classification instead.

## Features

- **Multiple Model Approaches**:
  - TF-IDF + Logistic Regression (fast, interpretable baseline)
  - DistilBERT transformer model (optional, higher accuracy)
- **Comprehensive Analysis**: Jupyter notebook with data exploration, training, and evaluation
- **Production-Ready API**: Python class interface for easy integration
- **Batch Processing**: Classify single tweets or batches efficiently
- **Probability Distributions**: Get confidence scores for all topics
- **Top-N Predictions**: Get the most likely topics for uncertain cases

## Project Structure

```
topic_classification/
├── .venv/                  # Virtual environment (created by uv)
├── pyproject.toml          # Project configuration and dependencies
├── .python-version         # Python version specification
├── notebooks/              # Jupyter notebooks
│   └── 01_topic_classification.ipynb  # Main training notebook
├── data/                   # Data storage (uses shared dataset)
├── models/                 # Trained model storage
│   ├── tfidf_vectorizer.pkl
│   ├── topic_classifier.pkl
│   └── topic_mappings.pkl
├── main.py                 # Production classifier API
└── README.md              # This file
```

## Setup

### Prerequisites
- Python 3.12+
- uv package manager

### Installation

```bash
# Navigate to the app directory
cd apps/topic_classification

# Install dependencies
uv sync
```

## Usage

### 1. Train the Model (First Time Only)

Before using the classifier, you need to train the model using the Jupyter notebook:

```bash
# Start Jupyter Lab
uv run jupyter lab

# Open and run: notebooks/01_topic_classification.ipynb
```

The notebook will:
1. Load and explore the tweets dataset
2. Analyze topic distribution
3. Train a TF-IDF + Logistic Regression model
4. Evaluate model performance
5. Save trained models to `models/` directory

### 2. Use the Classifier in Production

After training, you can use the classifier in your Python code:

```python
from main import TopicClassifier

# Initialize the classifier
classifier = TopicClassifier()

# Simple classification
topic = classifier.classify("Just had an amazing burger!")
print(topic)  # Output: 'food_drinks'

# Classification with probabilities
result = classifier.classify(
    "Traffic is terrible today",
    return_probabilities=True
)
print(result['predicted_topic'])  # 'transportation'
print(result['confidence'])       # 0.8542
print(result['probabilities'])    # {'transportation': 0.85, ...}

# Get top N most likely topics
top_topics = classifier.get_top_topics("Great food and music!", n=3)
# Returns: [('food_drinks', 0.65), ('entertainment', 0.20), ...]

# Batch classification
tweets = [
    "Great coffee this morning!",
    "Traffic jam on the highway",
    "My dog is so cute!"
]
topics = classifier.classify_batch(tweets)
# Returns: ['food_drinks', 'transportation', 'pets_animals']

# Get available topics
all_topics = classifier.get_available_topics()
print(f"The classifier can predict {len(all_topics)} different topics")

# Get model information
info = classifier.get_model_info()
print(info)
```

### 3. Run the Demo

```bash
# Run the demo script
uv run python main.py
```

This will demonstrate:
- Model information and available topics
- Single tweet classification
- Probability distributions
- Batch processing

## Dataset

The classifier uses `tweets_v1.csv` which contains:
- **tweet**: The tweet text
- **primary_topic**: The main topic (target variable)
- **secondary_topic**: Alternative topic classification
- Additional metadata: user info, engagement metrics, etc.

### Available Topics

The classifier can predict topics such as:
- food_drinks
- transportation
- entertainment
- nature_environment
- pets_animals
- shopping_retail
- work_business
- arts_culture
- personal_development
- community_social
- And more...

(Run the classifier to see the complete list of available topics)

## Model Performance

### TF-IDF + Logistic Regression
- **Training Time**: Fast (~1-2 seconds)
- **Inference Speed**: Very fast (~1ms per tweet)
- **Accuracy**: Good baseline performance
- **Interpretability**: High (can inspect feature weights)
- **Resource Requirements**: Low

### DistilBERT (Optional)
- **Training Time**: Slower (~5-10 minutes)
- **Inference Speed**: Moderate (~50ms per tweet)
- **Accuracy**: Higher than traditional ML
- **Interpretability**: Lower (black box)
- **Resource Requirements**: Higher (GPU recommended)

## API Reference

### TopicClassifier

#### `__init__(model_dir=None)`
Initialize the classifier with optional custom model directory.

#### `classify(tweet_text, return_probabilities=False)`
Classify a single tweet.
- **Args**:
  - `tweet_text` (str): The tweet to classify
  - `return_probabilities` (bool): Return probability distribution
- **Returns**: Topic string or dict with probabilities

#### `classify_batch(tweet_texts, return_probabilities=False)`
Classify multiple tweets at once.
- **Args**:
  - `tweet_texts` (List[str]): List of tweets
  - `return_probabilities` (bool): Return probabilities
- **Returns**: List of topics or list of dicts

#### `get_top_topics(tweet_text, n=5)`
Get the N most likely topics for a tweet.
- **Args**:
  - `tweet_text` (str): The tweet to classify
  - `n` (int): Number of top topics
- **Returns**: List of (topic, probability) tuples

#### `get_available_topics()`
Get all topics the classifier can predict.
- **Returns**: List of topic strings

#### `get_model_info()`
Get model metadata and statistics.
- **Returns**: Dict with model information

## Development

### Running Tests

```bash
# Run the main demo
uv run python main.py

# Or import in Python REPL
uv run python
>>> from main import TopicClassifier
>>> classifier = TopicClassifier()
>>> classifier.classify("Test tweet")
```

### Retraining the Model

If you want to retrain with different parameters:

1. Open `notebooks/01_topic_classification.ipynb`
2. Modify hyperparameters (e.g., `max_features`, `ngram_range`)
3. Run all cells to retrain
4. New models will be saved to `models/`

## Comparison with Sentiment Classifier

| Feature | Sentiment Classifier | Topic Classifier |
|---------|---------------------|------------------|
| **Target** | Sentiment (positive/negative/neutral) | Primary topic (food, transport, etc.) |
| **Classes** | 3 classes | 10+ classes (multi-class) |
| **Dataset** | tweets_v1.csv (sentiment_class) | tweets_v1.csv (primary_topic) |
| **Approach** | Same (TF-IDF + LR / DistilBERT) | Same (TF-IDF + LR / DistilBERT) |
| **Use Case** | Understanding sentiment | Content categorization |

## Next Steps

1. **Improve Accuracy**:
   - Experiment with different feature engineering
   - Try ensemble methods
   - Fine-tune transformer models

2. **Add Features**:
   - Secondary topic prediction
   - Multi-label classification (predict multiple topics)
   - Topic trend analysis over time

3. **Deployment**:
   - Build REST API with FastAPI
   - Create a web interface
   - Deploy to cloud (AWS/GCP/Azure)

4. **Integration**:
   - Combine with sentiment classifier for richer insights
   - Add to recommendation systems
   - Use for content filtering

## Dependencies

Key packages (managed by `uv`):
- **Jupyter ecosystem**: jupyter, ipykernel, jupyterlab
- **Data science**: numpy, pandas, scikit-learn
- **Visualization**: matplotlib, seaborn, wordcloud
- **NLP**: nltk, transformers, torch
- **ML frameworks**: transformers, datasets

All dependencies are specified in `pyproject.toml` and managed automatically by `uv`.

## Troubleshooting

### Models not found error
```
FileNotFoundError: Model files not found in models/
```
**Solution**: Run the Jupyter notebook first to train and save the models.

### Import errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Run `uv sync` to install dependencies.

### Low accuracy
- Check if you have enough training data
- Try the transformer model for better accuracy
- Experiment with different preprocessing steps

## License

This is an educational project for demonstration purposes.

## Authors

Built as part of the multi-agent todone system for parallel ML experimentation.
