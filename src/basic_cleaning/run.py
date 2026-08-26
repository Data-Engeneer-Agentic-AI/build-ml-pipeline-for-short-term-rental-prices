import argparse
import logging

import pandas as pd
import wandb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")

    logger.info("Downloading artifact %s", args.input_artifact)

    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    logger.info(
        "Original data has %s rows and %s columns",
        df.shape[0],
        df.shape[1],
    )

    df = df.drop_duplicates()
    df = df.dropna(subset=["price"])
    df = df[df["price"].between(args.min_price, args.max_price)]

    logger.info(
        "Cleaned data has %s rows and %s columns",
        df.shape[0],
        df.shape[1],
    )

    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)

    args = parser.parse_args()

    go(args)