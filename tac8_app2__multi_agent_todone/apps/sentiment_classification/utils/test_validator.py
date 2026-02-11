#!/usr/bin/env python3
"""
Test script for tweet data validator.

This script demonstrates the usage of the TweetDataValidator
and validates the tweets_v1.csv dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from utils.validate_tweets import validate_tweet_data, TweetDataValidator


def test_validator_with_real_data():
    """Test validator with the actual tweets_v1.csv dataset."""
    print("=" * 60)
    print("Testing Tweet Data Validator")
    print("=" * 60)

    # Load the data
    data_path = Path(__file__).parent.parent / "data" / "tweets_v1.csv"

    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        return False

    print(f"\n📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Run validation
    print("\n" + "=" * 60)
    print("Running Validation...")
    print("=" * 60)

    result = validate_tweet_data(df, verbose=True)

    # Additional statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    print(f"\nSentiment Class Distribution:")
    sentiment_counts = df["sentiment_class"].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")

    print(f"\nTweet Length Statistics:")
    print(f"  Min length: {df['tweet'].str.len().min()}")
    print(f"  Max length: {df['tweet'].str.len().max()}")
    print(f"  Mean length: {df['tweet'].str.len().mean():.1f}")
    print(f"  Median length: {df['tweet'].str.len().median():.1f}")

    print(f"\nMissing Values:")
    missing_summary = df.isna().sum()
    if missing_summary.sum() == 0:
        print("  ✅ No missing values detected")
    else:
        for col, count in missing_summary[missing_summary > 0].items():
            print(f"  {col}: {count}")

    return result.is_valid


def test_validator_with_bad_data():
    """Test validator with intentionally bad data to verify error detection."""
    print("\n" + "=" * 60)
    print("Testing Error Detection with Bad Data")
    print("=" * 60)

    # Create sample bad data
    bad_data = pd.DataFrame({
        "tweet": [
            "Good tweet",
            "",  # Too short
            "A" * 300,  # Too long
            None,  # Missing
            "Another good tweet"
        ],
        "sentiment_class": [
            "positive",
            "negative",
            "invalid_sentiment",  # Invalid
            "neutral",
            None  # Missing
        ],
        "user": ["@user1", "@user2", "@user3", "@user4", "@user5"],
        "sex": ["male", "female", "male", "female", "male"],
        "hour_of_day": [10, 11, 12, 13, 14],
        "total_comments": [5, 10, 15, 20, 25],
        "total_likes": [50, 100, 150, 200, 250],
        "total_impressions": [500, 1000, 1500, 2000, 2500],
        "tweet_length": [10, 0, 300, 0, 18],
        "verified_user": [True, False, True, False, True],
        "reply_to_verified": [False, True, False, True, False],
        "primary_topic": ["topic1", "topic2", "topic3", "topic4", "topic5"],
        "secondary_topic": ["topic6", "topic7", "topic8", "topic9", "topic10"]
    })

    print(f"\n📊 Testing with {len(bad_data)} rows of intentionally bad data")

    result = validate_tweet_data(bad_data, verbose=True)

    print("\n✅ Error detection test completed!")
    print(f"   Expected failures: DETECTED" if not result.is_valid else "   ⚠️  WARNING: Should have failed!")

    return not result.is_valid  # Should fail for bad data


def test_get_valid_rows():
    """Test filtering to get only valid rows."""
    print("\n" + "=" * 60)
    print("Testing Valid Row Filtering")
    print("=" * 60)

    # Load the data
    data_path = Path(__file__).parent.parent / "data" / "tweets_v1.csv"
    df = pd.read_csv(data_path)

    validator = TweetDataValidator()

    # Get valid and invalid rows
    valid_df = validator.get_valid_rows(df)
    invalid_df = validator.get_invalid_rows(df)

    print(f"\n📊 Total rows: {len(df)}")
    print(f"   Valid rows: {len(valid_df)} ({len(valid_df)/len(df)*100:.1f}%)")
    print(f"   Invalid rows: {len(invalid_df)} ({len(invalid_df)/len(df)*100:.1f}%)")

    if len(invalid_df) > 0:
        print(f"\n⚠️  Invalid rows detected:")
        print(invalid_df[["tweet", "sentiment_class"]].head())
    else:
        print(f"\n✅ All rows are valid!")

    return True


def main():
    """Run all tests."""
    print("\n🚀 Starting Tweet Data Validator Tests\n")

    tests = [
        ("Real Data Validation", test_validator_with_real_data),
        ("Error Detection", test_validator_with_bad_data),
        ("Valid Row Filtering", test_get_valid_rows)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' raised an exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")

    all_passed = all(success for _, success in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
