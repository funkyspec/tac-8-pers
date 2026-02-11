"""
Primary Topic Classifier

A production-ready classifier for predicting the primary topic of tweets.
Uses a pre-trained TF-IDF + Logistic Regression model.

Usage:
    from main import TopicClassifier

    classifier = TopicClassifier()
    topic = classifier.classify("Just had an amazing coffee!")
    # Returns: 'food_drinks'

    # With probabilities
    result = classifier.classify("Traffic is terrible today", return_probabilities=True)
    # Returns: {'predicted_topic': 'transportation', 'probabilities': {...}}

    # Batch classification
    topics = classifier.classify_batch(["Tweet 1", "Tweet 2", "Tweet 3"])
"""

import pickle
from pathlib import Path
from typing import Union, List, Dict
import warnings

warnings.filterwarnings('ignore')


class TopicClassifier:
    """Primary topic classifier for tweets."""

    def __init__(self, model_dir: str = None):
        """
        Initialize the topic classifier.

        Args:
            model_dir: Path to directory containing model files.
                      Defaults to 'models/' in the same directory as this file.
        """
        if model_dir is None:
            model_dir = Path(__file__).parent / 'models'
        else:
            model_dir = Path(model_dir)

        self.model_dir = model_dir
        self._load_models()

    def _load_models(self):
        """Load the pre-trained models and vectorizer."""
        try:
            # Load vectorizer
            vectorizer_path = self.model_dir / 'tfidf_vectorizer.pkl'
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

            # Load classifier
            classifier_path = self.model_dir / 'topic_classifier.pkl'
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)

            # Load topic mappings
            mappings_path = self.model_dir / 'topic_mappings.pkl'
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.topic2id = mappings['topic2id']
                self.id2topic = mappings['id2topic']

            self.topics = self.classifier.classes_

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}. "
                "Please run the Jupyter notebook to train and save models first."
            ) from e

    def classify(
        self,
        tweet_text: str,
        return_probabilities: bool = False
    ) -> Union[str, Dict]:
        """
        Classify the primary topic of a tweet.

        Args:
            tweet_text: The tweet text to classify
            return_probabilities: If True, return probabilities for all topics

        Returns:
            If return_probabilities is False:
                str: The predicted primary topic
            If return_probabilities is True:
                dict: Dictionary with 'predicted_topic' and 'probabilities'

        Examples:
            >>> classifier = TopicClassifier()
            >>> classifier.classify("Great burger at the new restaurant!")
            'food_drinks'

            >>> result = classifier.classify("Traffic jam again!", return_probabilities=True)
            >>> result['predicted_topic']
            'transportation'
            >>> result['probabilities']['transportation']
            0.8542
        """
        if not isinstance(tweet_text, str) or not tweet_text.strip():
            raise ValueError("tweet_text must be a non-empty string")

        # Transform the text
        tweet_tfidf = self.vectorizer.transform([tweet_text])

        # Predict
        predicted_topic = self.classifier.predict(tweet_tfidf)[0]

        if return_probabilities:
            # Get probabilities for all topics
            probabilities = self.classifier.predict_proba(tweet_tfidf)[0]
            topic_probs = dict(zip(self.classifier.classes_, probabilities))
            # Sort by probability (descending)
            topic_probs = dict(
                sorted(topic_probs.items(), key=lambda x: x[1], reverse=True)
            )
            return {
                'predicted_topic': predicted_topic,
                'probabilities': topic_probs,
                'confidence': topic_probs[predicted_topic]
            }

        return predicted_topic

    def classify_batch(
        self,
        tweet_texts: List[str],
        return_probabilities: bool = False
    ) -> Union[List[str], List[Dict]]:
        """
        Classify topics for multiple tweets at once.

        Args:
            tweet_texts: List of tweet texts to classify
            return_probabilities: If True, return probabilities for each tweet

        Returns:
            If return_probabilities is False:
                List[str]: List of predicted topics
            If return_probabilities is True:
                List[dict]: List of dictionaries with predictions and probabilities

        Examples:
            >>> classifier = TopicClassifier()
            >>> tweets = ["Great coffee!", "Traffic jam!", "Cute puppy!"]
            >>> classifier.classify_batch(tweets)
            ['food_drinks', 'transportation', 'pets_animals']
        """
        if not tweet_texts:
            return []

        if not all(isinstance(t, str) and t.strip() for t in tweet_texts):
            raise ValueError("All tweet_texts must be non-empty strings")

        # Transform all texts
        tweets_tfidf = self.vectorizer.transform(tweet_texts)

        # Predict
        predictions = self.classifier.predict(tweets_tfidf)

        if return_probabilities:
            probabilities_matrix = self.classifier.predict_proba(tweets_tfidf)
            results = []
            for pred, probs in zip(predictions, probabilities_matrix):
                topic_probs = dict(zip(self.classifier.classes_, probs))
                topic_probs = dict(
                    sorted(topic_probs.items(), key=lambda x: x[1], reverse=True)
                )
                results.append({
                    'predicted_topic': pred,
                    'probabilities': topic_probs,
                    'confidence': topic_probs[pred]
                })
            return results

        return predictions.tolist()

    def get_top_topics(
        self,
        tweet_text: str,
        n: int = 5
    ) -> List[tuple]:
        """
        Get the top N most likely topics for a tweet.

        Args:
            tweet_text: The tweet text to classify
            n: Number of top topics to return (default: 5)

        Returns:
            List of (topic, probability) tuples, sorted by probability

        Examples:
            >>> classifier = TopicClassifier()
            >>> classifier.get_top_topics("Great food and music!", n=3)
            [('food_drinks', 0.65), ('entertainment', 0.20), ('arts_culture', 0.08)]
        """
        result = self.classify(tweet_text, return_probabilities=True)
        top_topics = list(result['probabilities'].items())[:n]
        return top_topics

    def get_available_topics(self) -> List[str]:
        """
        Get list of all available topics the classifier can predict.

        Returns:
            List of topic names
        """
        return self.topics.tolist()

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        return {
            'num_topics': len(self.topics),
            'topics': self.get_available_topics(),
            'model_type': 'TF-IDF + Logistic Regression',
            'vectorizer_features': len(self.vectorizer.get_feature_names_out()),
        }


def main():
    """Demo usage of the TopicClassifier."""
    print("=" * 80)
    print("Primary Topic Classifier - Demo")
    print("=" * 80)

    # Initialize classifier
    try:
        classifier = TopicClassifier()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the Jupyter notebook first to train and save the models:")
        print("  cd apps/topic_classification")
        print("  uv run jupyter lab")
        print("  # Then run: notebooks/01_topic_classification.ipynb")
        return

    # Show model info
    info = classifier.get_model_info()
    print(f"\nModel Information:")
    print(f"  Type: {info['model_type']}")
    print(f"  Number of topics: {info['num_topics']}")
    print(f"  TF-IDF features: {info['vectorizer_features']}")

    print(f"\n\nAvailable Topics ({info['num_topics']} total):")
    for i, topic in enumerate(info['topics'], 1):
        print(f"  {i:2d}. {topic}")

    # Test predictions
    print("\n" + "=" * 80)
    print("Example Predictions")
    print("=" * 80)

    test_tweets = [
        "Just had the best pizza ever at this new Italian place!",
        "The highway is completely blocked due to an accident",
        "My cat just did the cutest thing!",
        "This new movie is absolutely amazing, must watch!",
        "We need to do more about climate change urgently",
        "Great workout at the gym today, feeling energized!",
        "The new smartphone has incredible features and battery life",
    ]

    for i, tweet in enumerate(test_tweets, 1):
        print(f"\n{i}. Tweet: {tweet}")

        # Simple prediction
        topic = classifier.classify(tweet)
        print(f"   Predicted Topic: {topic}")

        # Top 3 likely topics
        top_3 = classifier.get_top_topics(tweet, n=3)
        print(f"   Top 3 Topics:")
        for topic_name, prob in top_3:
            print(f"     - {topic_name}: {prob:.4f}")

    # Batch prediction
    print("\n" + "=" * 80)
    print("Batch Prediction")
    print("=" * 80)

    batch_results = classifier.classify_batch(test_tweets)
    print(f"\nClassified {len(test_tweets)} tweets:")
    for tweet, topic in zip(test_tweets, batch_results):
        print(f"  [{topic}] {tweet[:50]}...")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
