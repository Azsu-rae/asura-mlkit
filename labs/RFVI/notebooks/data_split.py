
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


SEED = 42

CLASS_NAMES = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}


def scan_dataset(data_dir):

    data_dir = Path(data_dir)

    records = []

    for label, class_name in enumerate(CLASS_NAMES):

        class_dir = data_dir / class_name

        for path in class_dir.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            records.append({
                "filepath": str(path),
                "filename": path.name,
                "class_name": class_name,
                "label": label
            })

    return pd.DataFrame(records)


def make_split(df):

    train_df, rest_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        rest_df,
        test_size=0.50,
        random_state=SEED,
        stratify=rest_df["label"]
    )

    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test")
        ],
        ignore_index=True
    )

    return split_df


def verify_split(sdf):

    assert set(
        sdf["split"].unique()
    ) == {
        "train", "val", "test"
    }

    path_sets = {
        s: set(
            sdf.loc[
                sdf["split"] == s,
                "filepath"
            ]
        )
        for s in ["train", "val", "test"]
    }

    assert path_sets["train"].isdisjoint(
        path_sets["val"]
    )

    assert path_sets["train"].isdisjoint(
        path_sets["test"]
    )

    assert path_sets["val"].isdisjoint(
        path_sets["test"]
    )

    for split_name in [
        "train", "val", "test"
    ]:

        classes = set(
            sdf.loc[
                sdf["split"] == split_name,
                "class_name"
            ]
        )

        assert classes == set(CLASS_NAMES)

    return True


def save_split(split_df, output_dir):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    split_df.to_csv(
        output_dir / "split.csv",
        index=False
    )

    for split_name in [
        "train", "val", "test"
    ]:

        split_df[
            split_df["split"] == split_name
        ].to_csv(
            output_dir /
            f"{split_name}.csv",
            index=False
        )


def load_split(output_dir):

    output_dir = Path(output_dir)

    split_df = pd.read_csv(
        output_dir / "split.csv"
    )

    assert verify_split(split_df)

    return split_df
