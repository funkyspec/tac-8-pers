#!/usr/bin/env python3
"""
Filter tweets dataset to include only technology and entertainment topics.

This script reads tweets_v1.csv and creates a filtered dataset containing only
tweets where the primary_topic is either 'technology' or 'entertainment'.
"""

import csv
from pathlib import Path


def filter_tweets_by_topics(
    input_file: Path,
    output_file: Path,
    topics: list[str]
) -> dict:
    """
    Filter tweets CSV by primary_topic.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        topics: List of topic values to include

    Returns:
        Dictionary with statistics about the filtering operation
    """
    total_rows = 0
    filtered_rows = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

        # Write header
        writer.writeheader()

        # Filter rows
        for row in reader:
            total_rows += 1
            if row['primary_topic'] in topics:
                writer.writerow(row)
                filtered_rows += 1

    return {
        'total_rows': total_rows,
        'filtered_rows': filtered_rows,
        'topics': topics,
        'input_file': str(input_file),
        'output_file': str(output_file)
    }


def main():
    # Define paths
    data_dir = Path(__file__).parent / 'data'
    input_file = data_dir / 'tweets_v1.csv'
    output_file = data_dir / 'tweets_tech_topics.csv'

    # Define topics to filter
    topics = ['technology', 'entertainment']

    # Perform filtering
    print(f"Filtering tweets from {input_file.name}...")
    print(f"Including topics: {', '.join(topics)}")

    stats = filter_tweets_by_topics(input_file, output_file, topics)

    # Print results
    print(f"\n✅ Filtering complete!")
    print(f"   Total rows processed: {stats['total_rows']}")
    print(f"   Rows matching filter: {stats['filtered_rows']}")
    print(f"   Output written to: {output_file}")
    print(f"   Retention rate: {stats['filtered_rows'] / stats['total_rows'] * 100:.1f}%")


if __name__ == '__main__':
    main()
