import json
import os
import tempfile

import hydra
import mlflow
from omegaconf import DictConfig


_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # Do NOT include test_regression_model here.
    # It must be run explicitly after assigning the W&B "prod" alias.
]


@hydra.main(
    version_base=None,
    config_name="config",
    config_path=".",
)
def go(config: DictConfig):

    # ---------------------------------------------------------
    # W&B configuration
    # ---------------------------------------------------------
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # ---------------------------------------------------------
    # Select pipeline steps
    # ---------------------------------------------------------
    steps_par = config["main"]["steps"]

    active_steps = (
        steps_par.split(",")
        if steps_par != "all"
        else _steps
    )

    # Hydra changes the current working directory.
    # This gives us the original repository directory.
    original_cwd = hydra.utils.get_original_cwd()

    with tempfile.TemporaryDirectory() as tmp_dir:

        # =====================================================
        # STEP 1 - DOWNLOAD DATA
        # =====================================================
        if "download" in active_steps:

            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description":
                        "Raw file as downloaded",
                },
            )

        # =====================================================
        # STEP 2 - BASIC CLEANING
        # =====================================================
        if "basic_cleaning" in active_steps:

            _ = mlflow.run(
                os.path.join(
                    original_cwd,
                    "src",
                    "basic_cleaning",
                ),
                "main",
                env_manager="conda",
                parameters={
                    "input_artifact":
                        "sample.csv:latest",

                    "output_artifact":
                        "clean_sample.csv",

                    "output_type":
                        "clean_sample",

                    "output_description":
                        "Data with outliers and null values removed",

                    "min_price":
                        config["etl"]["min_price"],

                    "max_price":
                        config["etl"]["max_price"],
                },
            )

        # =====================================================
        # STEP 3 - DATA CHECK
        # =====================================================
        if "data_check" in active_steps:

            _ = mlflow.run(
                os.path.join(
                    original_cwd,
                    "src",
                    "data_check",
                ),
                "main",
                env_manager="conda",
                parameters={
                    "csv":
                        "clean_sample.csv:latest",

                    "ref":
                        "clean_sample.csv:reference",

                    "kl_threshold":
                        config["data_check"]["kl_threshold"],

                    "min_price":
                        config["etl"]["min_price"],

                    "max_price":
                        config["etl"]["max_price"],
                },
            )

        # =====================================================
        # STEP 4 - TRAIN / VALIDATION / TEST SPLIT
        # =====================================================
        if "data_split" in active_steps:

            _ = mlflow.run(
                f"{config['main']['components_repository']}/"
                "train_val_test_split",
                "main",
                env_manager="conda",
                parameters={
                    "input":
                        "clean_sample.csv:latest",

                    "test_size":
                        config["modeling"]["test_size"],

                    "random_seed":
                        config["modeling"]["random_seed"],

                    "stratify_by":
                        config["modeling"]["stratify_by"],
                },
            )

        # =====================================================
        # STEP 5 - TRAIN RANDOM FOREST
        # =====================================================
        if "train_random_forest" in active_steps:

            # Serialize Random Forest parameters to JSON.
            # DO NOT change this logic.
            rf_config = os.path.abspath("rf_config.json")

            with open(rf_config, "w+") as fp:
                json.dump(
                    dict(
                        config[
                            "modeling"
                        ][
                            "random_forest"
                        ].items()
                    ),
                    fp,
                )

            _ = mlflow.run(
                os.path.join(
                    original_cwd,
                    "src",
                    "train_random_forest",
                ),
                "main",
                env_manager="conda",
                parameters={
                    "trainval_artifact":
                        "trainval_data.csv:latest",

                    "val_size":
                        config["modeling"]["val_size"],

                    "random_seed":
                        config["modeling"]["random_seed"],

                    "stratify_by":
                        config["modeling"]["stratify_by"],

                    "rf_config":
                        rf_config,

                    "max_tfidf_features":
                        config[
                            "modeling"
                        ][
                            "max_tfidf_features"
                        ],

                    "output_artifact":
                        "random_forest_export",
                },
            )

        # =====================================================
        # STEP 6 - TEST PRODUCTION MODEL
        # =====================================================
        if "test_regression_model" in active_steps:

            _ = mlflow.run(
                f"{config['main']['components_repository']}/"
                "test_regression_model",
                "main",
                env_manager="conda",
                parameters={
                    "mlflow_model":
                        "random_forest_export:prod",

                    "test_dataset":
                        "test_data.csv:latest",
                },
            )


if __name__ == "__main__":
    go()