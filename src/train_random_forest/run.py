#!/usr/bin/env python

"""
This script trains a Random Forest
"""

import argparse
import json
import logging
import os
import shutil

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import wandb

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
)


def delta_date_feature(dates):
    """
    Given a 2d array containing dates
    (in any format recognized by pd.to_datetime),
    return the delta in days between each date
    and the most recent date in its column.
    """

    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)

    return date_sanitized.apply(
        lambda d: (d.max() - d).dt.days,
        axis=0
    ).to_numpy()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)

logger = logging.getLogger()


def go(args):

    # -------------------------------------------------------
    # Initialize W&B run
    # -------------------------------------------------------
    run = wandb.init(job_type="train_random_forest")

    run.config.update(args)

    # -------------------------------------------------------
    # Load Random Forest configuration
    # -------------------------------------------------------
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)

    run.config.update(rf_config)

    # Reproducibility
    rf_config["random_state"] = args.random_seed

    # -------------------------------------------------------
    # Download train + validation artifact
    # -------------------------------------------------------
    logger.info(
        "Downloading training artifact %s",
        args.trainval_artifact
    )

    trainval_local_path = run.use_artifact(
        args.trainval_artifact
    ).file()

    # -------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------
    X = pd.read_csv(trainval_local_path)

    # Remove price from X and use it as target
    y = X.pop("price")

    logger.info(
        "Minimum price: %s, Maximum price: %s",
        y.min(),
        y.max()
    )

    # -------------------------------------------------------
    # Train-validation split
    # -------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=X[args.stratify_by],
        random_state=args.random_seed,
    )

    # -------------------------------------------------------
    # Prepare sklearn pipeline
    # -------------------------------------------------------
    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(
        rf_config,
        args.max_tfidf_features
    )

    # -------------------------------------------------------
    # Train
    # -------------------------------------------------------
    logger.info("Fitting")

    sk_pipe.fit(
        X_train,
        y_train
    )

    # -------------------------------------------------------
    # Validation metrics
    # -------------------------------------------------------
    logger.info("Scoring")

    r_squared = sk_pipe.score(
        X_val,
        y_val
    )

    y_pred = sk_pipe.predict(
        X_val
    )

    mae = mean_absolute_error(
        y_val,
        y_pred
    )

    logger.info(
        "Score: %s",
        r_squared
    )

    logger.info(
        "MAE: %s",
        mae
    )

    # -------------------------------------------------------
    # Export model
    # -------------------------------------------------------
    logger.info("Exporting model")

    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    mlflow.sklearn.save_model(
        sk_pipe,
        "random_forest_dir"
    )

    # -------------------------------------------------------
    # Upload model artifact to W&B
    # -------------------------------------------------------
    model_artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Random Forest model export",
        metadata=rf_config,
    )

    model_artifact.add_dir(
        "random_forest_dir"
    )

    run.log_artifact(
        model_artifact
    )

    # -------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------
    fig_feat_imp = plot_feature_importance(
        sk_pipe,
        processed_features
    )

    # -------------------------------------------------------
    # Save metrics to W&B
    # -------------------------------------------------------
    run.summary["r2"] = r_squared
    run.summary["mae"] = mae

    # -------------------------------------------------------
    # Upload feature importance visualization
    # -------------------------------------------------------
    run.log(
        {
            "feature_importance": wandb.Image(
                fig_feat_imp
            )
        }
    )

    plt.close(fig_feat_imp)

    run.finish()


def plot_feature_importance(pipe, feat_names):

    # Collect feature importance for all non-NLP features
    feat_imp = pipe[
        "random_forest"
    ].feature_importances_[
        :len(feat_names) - 1
    ]

    # Sum all TF-IDF dimensions into one NLP importance
    nlp_importance = sum(
        pipe[
            "random_forest"
        ].feature_importances_[
            len(feat_names) - 1:
        ]
    )

    feat_imp = np.asarray(
        np.append(
            feat_imp,
            nlp_importance
        )
    )

    fig_feat_imp, sub_feat_imp = plt.subplots(
        figsize=(10, 10),
        layout="constrained"
    )

    sub_feat_imp.bar(
        np.arange(
            feat_imp.shape[0]
        ),
        feat_imp,
        color="r",
        align="center"
    )

    sub_feat_imp.set_xticks(
        np.arange(
            feat_imp.shape[0]
        )
    )

    sub_feat_imp.set_xticklabels(
        feat_names,
        rotation=90
    )

    return fig_feat_imp


def get_inference_pipeline(
    rf_config,
    max_tfidf_features
):

    # -------------------------------------------------------
    # Categorical features
    # -------------------------------------------------------

    ordinal_categorical = [
        "room_type"
    ]

    non_ordinal_categorical = [
        "neighbourhood_group"
    ]

    # Room type is ordinal
    ordinal_categorical_preproc = OrdinalEncoder()

    # Non-ordinal categorical preprocessing:
    # 1. Impute missing values
    # 2. One-hot encode
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(
            strategy="most_frequent"
        ),
        OneHotEncoder()
    )

    # -------------------------------------------------------
    # Numerical columns
    # -------------------------------------------------------
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude",
    ]

    zero_imputer = SimpleImputer(
        strategy="constant",
        fill_value=0
    )

    # -------------------------------------------------------
    # Date feature engineering
    # -------------------------------------------------------
    date_imputer = make_pipeline(
        SimpleImputer(
            strategy="constant",
            fill_value="2010-01-01"
        ),
        FunctionTransformer(
            delta_date_feature,
            check_inverse=False,
            validate=False
        ),
    )

    # -------------------------------------------------------
    # Text/NLP preprocessing
    # -------------------------------------------------------
    name_tfidf = make_pipeline(
        SimpleImputer(
            strategy="constant",
            fill_value=""
        ),
        FunctionTransformer(
            lambda x: x.squeeze(),
            validate=False,
            feature_names_out="one-to-one",
        ),
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words="english",
        ),
    )

    # -------------------------------------------------------
    # Combine preprocessing
    # -------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal_cat",
                ordinal_categorical_preproc,
                ordinal_categorical
            ),
            (
                "non_ordinal_cat",
                non_ordinal_categorical_preproc,
                non_ordinal_categorical
            ),
            (
                "impute_zero",
                zero_imputer,
                zero_imputed
            ),
            (
                "transform_date",
                date_imputer,
                ["last_review"]
            ),
            (
                "transform_name",
                name_tfidf,
                ["name"]
            ),
        ],
        remainder="drop",
    )

    processed_features = (
        ordinal_categorical
        + non_ordinal_categorical
        + zero_imputed
        + ["last_review", "name"]
    )

    # -------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------
    random_forest = RandomForestRegressor(
        **rf_config
    )

    # -------------------------------------------------------
    # Full inference pipeline
    # -------------------------------------------------------
    sk_pipe = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "random_forest",
                random_forest
            ),
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Random Forest model"
    )

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help=(
            "Artifact containing the training dataset. "
            "It will be split into train and validation"
        ),
        required=True,
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help=(
            "Size of the validation split. "
            "Fraction of the dataset, or number of items"
        ),
        required=True,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="neighbourhood_group",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help=(
            "Random forest configuration. "
            "A JSON dict that will be passed to "
            "RandomForestRegressor."
        ),
        required=True,
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words to consider for TF-IDF",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)