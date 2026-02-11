#!/usr/bin/env python3
"""
Filter tweets_v1.csv to create a dataset containing only positive sentiment tweets.
Output: data/tweets_positive_only.csv
"""

import csv
from pathlib import Path


def filter_positive_tweets():
    """Filter tweets to include only those with positive sentiment."""
    # Define paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "data" / "tweets_v1.csv"
    output_file = script_dir / "data" / "tweets_positive_only.csv"

    # Ensure input file exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read and filter
    positive_tweets = []
    total_rows = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        for row in reader:
            total_rows += 1
            if row['sentiment_class'] == 'positive':
                positive_tweets.append(row)

    # Write filtered data
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(positive_tweets)

    # Print summary
    print(f"Filtering complete!")
    print(f"Total tweets processed: {total_rows}")
    print(f"Positive tweets found: {len(positive_tweets)}")
    print(f"Output file created: {output_file}")
    print(f"Percentage positive: {len(positive_tweets) / total_rows * 100:.1f}%")


if __name__ == "__main__":
    filter_positive_tweets()
