"""
Dataset Ingestion and Contract Enforcement Module

Enforces the standardized dataset schema, validates data integrity,
and provides adapters for external and synthetic research corpora.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.data.sanitization import TextSanitizer


EXPECTED_COLUMNS = ["id", "text", "label", "source", "split"]
VALID_LABELS = {0, 1, 2, 3}
VALID_SPLITS = {"train", "val", "test"}


class DatasetValidationError(Exception):
    """Raised when a dataset violates the formal schema or data integrity rules."""
    pass


def validate_dataset(
    df: pd.DataFrame,
    require_split: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validates a DataFrame against the formal SUBTEXT data contract.
    
    Args:
        df: Pandas DataFrame to validate.
        require_split: If True, requires the 'split' column to be populated with 'train'/'val'/'test'.
        
    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    # 1. Column existence
    required = EXPECTED_COLUMNS if require_split else [c for c in EXPECTED_COLUMNS if c != "split"]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors  # Cannot continue further checks without required columns

    # 2. Check for empty dataframe
    if len(df) == 0:
        errors.append("Dataset contains 0 rows.")
        return False, errors

    # 3. Check for Null / NaN values
    for col in required:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' contains {null_count} null/NaN values.")

    # 4. Check 'id' uniqueness
    dup_ids = df["id"].duplicated().sum()
    if dup_ids > 0:
        errors.append(f"Found {dup_ids} duplicate IDs.")

    # 5. Check 'text' validity (must be non-empty string after strip)
    empty_texts = df["text"].astype(str).str.strip().eq("").sum()
    if empty_texts > 0:
        errors.append(f"Found {empty_texts} empty or whitespace-only text entries.")

    # 6. Check 'label' range
    invalid_labels = ~df["label"].isin(VALID_LABELS)
    if invalid_labels.any():
        bad_values = df.loc[invalid_labels, "label"].unique().tolist()
        errors.append(f"Found invalid labels: {bad_values}. Allowed: {sorted(list(VALID_LABELS))}")

    # 7. Check 'split' values if required
    if require_split and "split" in df.columns:
        invalid_splits = ~df["split"].isin(VALID_SPLITS)
        if invalid_splits.any():
            bad_splits = df.loc[invalid_splits, "split"].unique().tolist()
            errors.append(f"Found invalid split tags: {bad_splits}. Allowed: {sorted(list(VALID_SPLITS))}")

    return len(errors) == 0, errors


def load_raw_dataset(
    file_path: Union[str, Path],
    column_mapping: Optional[Dict[str, str]] = None,
    source_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads raw tabular dataset from CSV, TSV, Parquet, or JSON Lines,
    normalizing columns into the standard schema.
    
    Args:
        file_path: Path to dataset file.
        column_mapping: Optional dict mapping {raw_column_name: target_column_name}.
        source_name: Fallback corpus name if 'source' column is absent.
        
    Returns:
        Standardized pandas DataFrame conforming to schema (split may be unassigned).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    elif suffix in [".csv", ".tsv"]:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    elif suffix in [".json", ".jsonl"]:
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: .parquet, .csv, .tsv, .jsonl")

    # Apply column mapping
    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Ensure text column exists
    if "text" not in df.columns:
        raise DatasetValidationError("Dataset must contain a 'text' column.")

    # Ensure label column exists
    if "label" not in df.columns:
        raise DatasetValidationError("Dataset must contain a 'label' column.")

    # Generate deterministic IDs if not present
    if "id" not in df.columns:
        df["id"] = df["text"].apply(
            lambda t: hashlib.sha256(str(t).encode("utf-8")).hexdigest()[:16]
        )

    # Assign source if not present
    if "source" not in df.columns:
        df["source"] = source_name or path.stem

    # Ensure labels are integers
    df["label"] = df["label"].astype(int)

    # Clean text to string
    df["text"] = df["text"].astype(str)

    return df


def generate_synthetic_benchmark(
    num_samples: int = 400,
    seed: int = 42,
    inject_pii: bool = True,
) -> pd.DataFrame:
    """
    Generates a balanced, highly realistic synthetic research corpus for testing
    the end-to-end pipeline without requiring external data access.
    
    Specifically constructs samples that test:
    - Class 0 (Control): Hyperbolic figurative frustration, sarcasm, gaming/sports banter,
      colloquial usage of "dying", "killing me", with zero crisis intent.
    - Class 1 (Low Stress): Everyday academic/work burnout, fatigue, deadlines.
    - Class 2 (Moderate Distress): Depressive dysphoria, feeling numb, isolated, purposeless.
    - Class 3 (Severe Crisis): High-risk active suicide ideation, farewell statements, explicit intent.
    
    Args:
        num_samples: Total number of samples to generate (distributed across classes).
        seed: Random seed for reproducibility.
        inject_pii: If True, injects synthetic usernames, emails, URLs, and phone numbers.
        
    Returns:
        DataFrame conforming to the standard schema with unassigned splits.
    """
    rng = np.random.default_rng(seed)
    samples_per_class = num_samples // 4

    # Combinatorial linguistic pools per class to ensure every sample is unique
    prefixes = {
        0: ["Honestly, ", "OMG, ", "Just realized that ", "Not gonna lie, ", "Update: ", "Lol, ", "", "Seriously, ", "Man, ", "Bruh, "],
        1: ["Feeling so ", "Currently ", "Lately I have been ", "Really ", "So ", "", "Genuinely ", "Quite ", "Starting to feel "],
        2: ["Every single day, ", "For weeks now, ", "It feels like ", "Truthfully, ", "I feel like ", "", "Deep down, ", "Lately, "],
        3: ["This is my final note. ", "I have made my choice. ", "I cannot keep going. ", "Tonight is the end. ", "Goodbye everyone. ", "", "I have decided. "],
    }

    core_phrases = {
        0: [
            "this chemistry midterm is literally killing me with these equations",
            "I'm actually dying of laughter right now watching this absurd cat video",
            "the highway traffic this morning made me want to scream into my steering wheel",
            "my soccer team conceded in stoppage time, I swear I want to throw my TV",
            "just ran 8 miles in the blazing heat and my legs are completely dead",
            "the campus Wi-Fi dropped during my video call, someone just shoot me now haha",
            "that new horror movie almost gave me a literal heart attack in the theater",
            "finally got a decent night of sleep after spending the weekend playing games",
            "this rain will not stop and I'm trapped inside drinking too much coffee",
            "this extra spicy vindaloo is murdering my tongue but it's delicious",
            "had a great debate in seminar today about NLP interpretability and ethics",
            "sitting through a three hour corporate seminar that really could have been a memo",
            "my laptop battery died right before I saved my draft, total disaster lol",
            "spent two hours assembling this IKEA bookshelf and my back is destroyed",
            "forgot my umbrella at home so now I look like a drowned rat, classic Monday",
            "cannot believe the season finale ended on that cliffhanger, I am losing my mind",
        ],
        1: [
            "overwhelmed with multiple project deadlines this week and running on 4 hours of sleep",
            "super stressed about the comprehensive exams next week and trying to memorize everything",
            "drained from balancing a 30-hour work shift on top of full-time undergraduate courses",
            "had a tense misunderstanding with my roommate about rent and groceries",
            "feeling heavy pressure from my family regarding upcoming summer internship interviews",
            "very anxious about presenting our group quarterly report in front of the VP tomorrow",
            "chores and laundry have been piling up all week because there's simply no free time",
            "struggling with mild tension headaches and needing to cut back on energy drinks",
            "worried about car repair costs this month after an unexpected alternator breakdown",
            "my work schedule got reshuffled and it's making my sleep routine totally erratic",
            "feeling slightly burned out from continuous back-to-back assignments this semester",
            "trying hard to manage my daily time better but work notifications never pause",
        ],
        2: [
            "feeling completely hollow inside for months and nothing seems to spark joy anymore",
            "trapped in a relentless cycle of numbness and detachment from everyone who cares",
            "convinced that I'm just a persistent burden to my family and friends around me",
            "crying in my car after work every evening because the loneliness is overwhelming",
            "social isolation is exhausting me, but I lack the mental energy to answer basic texts",
            "putting on a functional mask at my job and then immediately collapsing in bed at night",
            "it feels like I am slowly suffocating while everyone else is breathing effortlessly",
            "unable to care about my career, goals, or hobbies due to constant depressive fog",
            "haunted by persistent feelings of worthlessness regardless of what tasks I complete",
            "waking up each morning with a heavy dread wishing the day was already over",
            "carrying this deep emotional ache that never seems to lighten no matter what I do",
            "withdrawing further from my closest friends because explaining this sadness feels impossible",
        ],
        3: [
            "cannot bear this agony another minute, tonight is my last night and I am ready",
            "finished writing my goodbye letters and gave away my remaining possessions, finally at peace",
            "have the lethal dose ready on my nightstand, I will not be waking up tomorrow morning",
            "the unbearable pain has won and there are no other exits left for me, this is goodbye",
            "standing on the high bridge right now looking down, there is nothing left holding me",
            "made my final decision to end my life tonight, please forgive me for doing this",
            "everything is planned and set for tonight, nobody can convince me to stay anymore",
            "stepping into the permanent dark tonight, I just want the relentless suffering to end",
            "saying my final farewell to this world, thank you to those who showed kindness",
            "the means are right in front of me and my resolve is absolute, peace at last tonight",
            "there is no hope left for me and my time has run out, tonight I am ending it all",
            "leaving tonight forever, cannot survive another sunrise of this unbearable torment",
        ],
    }

    suffixes = {
        0: [" lol.", " haha.", " send help!", " honestly so ridiculous.", " but we move on.", " anyway, have a good one.", ""],
        1: [" Need to get through this week.", " Hopefully things settle down soon.", " Just taking it hour by hour.", " Need some proper sleep.", ""],
        2: [" I just want the emptiness to fade.", " Feeling so invisible.", " I don't know how to fix this.", " It's just so heavy.", ""],
        3: [" Goodbye.", " Farewell forever.", " Please don't blame yourselves.", " Finally at rest.", ""],
    }

    synthetic_pii_snippets = [
        " Email me at user_{i}@fakemail.com if you want.",
        " Follow my account @social_user_{i} for updates.",
        " Check out the full story at https://myblog.example.com/post/{i}.",
        " Reach out to u/helper_person_{i} on Reddit.",
        " Call my direct number 555-01{i:02d} anytime.",
    ]

    records = []
    seen_texts = set()
    sample_id = 0

    for label in [0, 1, 2, 3]:
        count = 0
        pref_pool = prefixes[label]
        core_pool = core_phrases[label]
        suff_pool = suffixes[label]

        # Combinatorially sample unique sentences
        attempts = 0
        while count < samples_per_class and attempts < samples_per_class * 20:
            attempts += 1
            sample_id += 1
            p = rng.choice(pref_pool)
            c = rng.choice(core_pool)
            s = rng.choice(suff_pool)
            base_text = f"{p}{c}{s}".strip()

            # Ensure uniqueness
            if base_text in seen_texts:
                # Add unique identifier variation if needed
                base_text = f"{base_text} (post #{sample_id})"

            seen_texts.add(base_text)

            # Randomly inject synthetic PII to verify sanitization
            if inject_pii and rng.random() < 0.35:
                pii_addon = rng.choice(synthetic_pii_snippets).format(i=sample_id)
                full_text = f"{base_text}{pii_addon}"
            else:
                full_text = base_text

            rec_id = hashlib.sha256(f"{label}_{sample_id}_{full_text}".encode("utf-8")).hexdigest()[:16]

            records.append({
                "id": rec_id,
                "text": full_text,
                "label": int(label),
                "source": "synthetic_benchmark",
                "split": "unassigned",
            })
            count += 1

    # Shuffle dataset
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df
