from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (
    cross_validate,
    KFold,
)
import pandas as pd
from dvclive import Live
from dvc.api import params_show
from rich import print

RANDOM_STATE = 42


def get_model(model_params: dict):
    pipeline = Pipeline(
        [
            (
                "column_transformer",
                ColumnTransformer(
                    [
                        (
                            "Pclass-onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["Pclass"],
                        ),
                        ("Sex-onehot", OneHotEncoder(handle_unknown="ignore"), ["Sex"]),
                        (
                            "Cabin-onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["Cabin"],
                        ),
                        (
                            "Embarked-onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["Embarked"],
                        ),
                    ]
                ),
            ),
            (
                "estimator",
                RandomForestClassifier(**model_params, random_state=RANDOM_STATE),
            ),
        ]
    )
    return pipeline


def main(df_path: str = "data/train.csv"):
    params = params_show()
    print(params)
    model = get_model(params["model"])
    titanic_data = pd.read_csv(df_path)
    X = titanic_data.drop(["Survived", "Name", "Ticket"], axis="columns")
    Y = titanic_data["Survived"]
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    score = cross_validate(
        model,
        X,
        Y,
        cv=cv,
        scoring={"acc": "accuracy", "presision": "precision", "rec": "recall"},
    )
    with Live(save_dvc_exp=True) as live:
        for k, v in score.items():
            live.log_metric(f"{k}_mean", v.mean())
            live.log_metric(f"{k}_std", v.std())


if __name__ == "__main__":
    main()
