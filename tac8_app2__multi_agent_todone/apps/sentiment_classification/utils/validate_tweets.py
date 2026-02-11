"""
Data quality validator for tweet sentiment classification dataset.

This module provides validation functions to ensure data quality by checking:
- Missing values in critical columns
- Valid sentiment classes
- Text length constraints
"""

from dataclasses import dataclass
from typing import List, Set
import pandas as pd


@dataclass
class ValidationResult:
    """Result of data validation with detailed error information."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    total_rows: int
    valid_rows: int

    def __str__(self) -> str:
        """Format validation result as readable string."""
        status = "✅ PASSED" if self.is_valid else "❌ FAILED"
        result = [
            f"\nValidation Result: {status}",
            f"Total rows: {self.total_rows}",
            f"Valid rows: {self.valid_rows}",
        ]

        if self.errors:
            result.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                result.append(f"  - {error}")

        if self.warnings:
            result.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                result.append(f"  - {warning}")

        return "\n".join(result)


class TweetDataValidator:
    """Validator for tweet sentiment classification datasets."""

    # Valid sentiment classes
    VALID_SENTIMENTS = {"positive", "negative", "neutral"}

    # Text length constraints
    MIN_TWEET_LENGTH = 1
    MAX_TWEET_LENGTH = 280  # Twitter's character limit

    # Required columns
    REQUIRED_COLUMNS = {
        "tweet",
        "sentiment_class",
        "user",
        "sex",
        "hour_of_day",
        "total_comments",
        "total_likes",
        "total_impressions",
        "tweet_length",
        "verified_user",
        "reply_to_verified",
        "primary_topic",
        "secondary_topic"
    }

    # Columns that should not have missing values
    CRITICAL_COLUMNS = {"tweet", "sentiment_class"}

    def __init__(
        self,
        valid_sentiments: Set[str] = None,
        min_tweet_length: int = None,
        max_tweet_length: int = None,
        strict_mode: bool = False
    ):
        """
        Initialize validator with custom parameters.

        Args:
            valid_sentiments: Set of valid sentiment class labels
            min_tweet_length: Minimum allowed tweet length
            max_tweet_length: Maximum allowed tweet length
            strict_mode: If True, treat warnings as errors
        """
        self.valid_sentiments = valid_sentiments or self.VALID_SENTIMENTS
        self.min_tweet_length = min_tweet_length or self.MIN_TWEET_LENGTH
        self.max_tweet_length = max_tweet_length or self.MAX_TWEET_LENGTH
        self.strict_mode = strict_mode

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate the entire dataset.

        Args:
            df: DataFrame containing tweet data

        Returns:
            ValidationResult with detailed error and warning information
        """
        errors = []
        warnings = []

        # Check required columns
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(sorted(missing_cols))}")
            # Return early if required columns are missing
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                total_rows=len(df),
                valid_rows=0
            )

        # Check for missing values in critical columns
        missing_results = self._check_missing_values(df)
        errors.extend(missing_results["errors"])
        warnings.extend(missing_results["warnings"])

        # Check sentiment classes
        sentiment_results = self._check_sentiment_classes(df)
        errors.extend(sentiment_results["errors"])
        warnings.extend(sentiment_results["warnings"])

        # Check text length constraints
        length_results = self._check_text_lengths(df)
        errors.extend(length_results["errors"])
        warnings.extend(length_results["warnings"])

        # Calculate valid rows (rows without critical issues)
        valid_mask = self._get_valid_rows_mask(df)
        valid_rows = valid_mask.sum()

        # In strict mode, warnings become errors
        if self.strict_mode and warnings:
            errors.extend(warnings)
            warnings = []

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            total_rows=len(df),
            valid_rows=valid_rows
        )

    def _check_missing_values(self, df: pd.DataFrame) -> dict:
        """Check for missing values in critical columns."""
        errors = []
        warnings = []

        for col in self.CRITICAL_COLUMNS:
            if col not in df.columns:
                continue

            missing_count = df[col].isna().sum()
            if missing_count > 0:
                errors.append(
                    f"Column '{col}' has {missing_count} missing values "
                    f"({missing_count/len(df)*100:.1f}%)"
                )

        # Check for missing values in non-critical columns
        non_critical_cols = set(df.columns) - self.CRITICAL_COLUMNS
        for col in non_critical_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                warnings.append(
                    f"Column '{col}' has {missing_count} missing values "
                    f"({missing_count/len(df)*100:.1f}%)"
                )

        return {"errors": errors, "warnings": warnings}

    def _check_sentiment_classes(self, df: pd.DataFrame) -> dict:
        """Check that all sentiment classes are valid."""
        errors = []
        warnings = []

        if "sentiment_class" not in df.columns:
            return {"errors": errors, "warnings": warnings}

        # Get unique sentiment values (excluding NaN)
        unique_sentiments = set(df["sentiment_class"].dropna().unique())

        # Check for invalid sentiment classes
        invalid_sentiments = unique_sentiments - self.valid_sentiments
        if invalid_sentiments:
            invalid_count = df["sentiment_class"].isin(invalid_sentiments).sum()
            errors.append(
                f"Found {invalid_count} rows with invalid sentiment classes: "
                f"{', '.join(sorted(invalid_sentiments))}"
            )

        # Check sentiment class distribution
        sentiment_counts = df["sentiment_class"].value_counts()
        if len(sentiment_counts) > 0:
            min_count = sentiment_counts.min()
            max_count = sentiment_counts.max()
            if max_count > 0 and min_count / max_count < 0.1:
                warnings.append(
                    f"Imbalanced sentiment classes detected. "
                    f"Distribution: {sentiment_counts.to_dict()}"
                )

        return {"errors": errors, "warnings": warnings}

    def _check_text_lengths(self, df: pd.DataFrame) -> dict:
        """Check that tweet text lengths are within constraints."""
        errors = []
        warnings = []

        if "tweet" not in df.columns:
            return {"errors": errors, "warnings": warnings}

        # Calculate actual text lengths for non-null tweets
        text_lengths = df["tweet"].dropna().str.len()

        # Check for tweets that are too short
        too_short = (text_lengths < self.min_tweet_length).sum()
        if too_short > 0:
            errors.append(
                f"Found {too_short} tweets shorter than {self.min_tweet_length} characters"
            )

        # Check for tweets that are too long
        too_long = (text_lengths > self.max_tweet_length).sum()
        if too_long > 0:
            errors.append(
                f"Found {too_long} tweets longer than {self.max_tweet_length} characters"
            )

        # Check if tweet_length column matches actual lengths
        if "tweet_length" in df.columns:
            df_with_text = df[df["tweet"].notna()].copy()
            actual_lengths = df_with_text["tweet"].str.len()
            recorded_lengths = df_with_text["tweet_length"]

            mismatches = (actual_lengths != recorded_lengths).sum()
            if mismatches > 0:
                warnings.append(
                    f"Found {mismatches} rows where 'tweet_length' column "
                    f"doesn't match actual tweet length"
                )

        return {"errors": errors, "warnings": warnings}

    def _get_valid_rows_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get boolean mask of rows that pass all critical validations."""
        mask = pd.Series([True] * len(df), index=df.index)

        # Must have tweet text
        if "tweet" in df.columns:
            mask &= df["tweet"].notna()
            mask &= df["tweet"].str.len() >= self.min_tweet_length
            mask &= df["tweet"].str.len() <= self.max_tweet_length

        # Must have valid sentiment class
        if "sentiment_class" in df.columns:
            mask &= df["sentiment_class"].notna()
            mask &= df["sentiment_class"].isin(self.valid_sentiments)

        return mask

    def get_valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only the rows that pass all critical validations.

        Args:
            df: DataFrame containing tweet data

        Returns:
            Filtered DataFrame with only valid rows
        """
        mask = self._get_valid_rows_mask(df)
        return df[mask].copy()

    def get_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only the rows that fail critical validations.

        Args:
            df: DataFrame containing tweet data

        Returns:
            Filtered DataFrame with only invalid rows
        """
        mask = self._get_valid_rows_mask(df)
        return df[~mask].copy()


def validate_tweet_data(
    df: pd.DataFrame,
    strict_mode: bool = False,
    verbose: bool = True
) -> ValidationResult:
    """
    Convenience function to validate tweet data with default settings.

    Args:
        df: DataFrame containing tweet data
        strict_mode: If True, treat warnings as errors
        verbose: If True, print validation result

    Returns:
        ValidationResult with detailed error and warning information

    Example:
        >>> import pandas as pd
        >>> from validate_tweets import validate_tweet_data
        >>> df = pd.read_csv("tweets_v1.csv")
        >>> result = validate_tweet_data(df)
        >>> if not result.is_valid:
        >>>     print("Validation failed!")
    """
    validator = TweetDataValidator(strict_mode=strict_mode)
    result = validator.validate(df)

    if verbose:
        print(result)

    return result
