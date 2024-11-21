# cortex/core/models/schema.py
from __future__ import annotations

import datetime
from typing import Annotated, Any, Literal, Optional

from architecture import BaseModel, Meta, field

from ..infra.constants import AlgorithmType


class ColumnInfo(BaseModel):
    """
    Represents information about a dataset column.

    Attributes:
        name (str): Name of the column.
        dtype (Literal["int64", "float64", "object", "bool", "datetime64"]): Data type of the column.
        sample_value (Any): A sample value from the column.
    """

    name: Annotated[
        str,
        Meta(
            title="Column Name",
            description="Name of the column.",
            examples=["age", "gender", "income"],
        ),
    ]

    dtype: Annotated[
        Literal["int64", "float64", "object", "bool", "datetime64"],
        Meta(
            title="Data Type",
            description="Data type of the column.",
            examples=["int64", "float64", "object"],
        ),
    ]

    sample_value: Annotated[
        Any,
        Meta(
            title="Sample Value",
            description="A sample value from the column.",
            examples=[25, "Male", 50000.0],
        ),
    ] = field(default=None)


class ForgedModel(BaseModel):
    """
    Represents a trained machine learning model along with its metadata and artifacts.

    Attributes:
        uid (str): Unique identifier for the model.
        name (Optional[str]): Name of the model.
        description (Optional[str]): Description of the model.
        algorithm (AlgorithmType): Algorithm used for training.
        hyperparameters (dict[str, Any]): Hyperparameters used during training.
        metrics (dict[str, float]): Evaluation metrics of the trained model.
        created_at (str): Timestamp of model creation.
        updated_at (Optional[str]): Timestamp of last model update.
        artifacts (list[str]): Paths to model artifacts (e.g., model file, encoder, scaler).
        feature_names (list[str]): List of feature names used during training.
        target_name (str): Name of the target variable.
        categorical_columns (list[CategoricalColumn]): List of categorical columns and their encoders.
        scaler (Optional[str]): Name of the scaler used for feature scaling.
        columns_info (list[ColumnInfo]): Information about each column in the dataset.
    """

    uid: Annotated[
        str, Meta(title="UID", description="Unique Identifier for the Forged Model")
    ]

    target_name: Annotated[
        str,
        Meta(title="Target Variable", description="Name of the target variable."),
    ]

    name: Annotated[
        Optional[str],
        Meta(title="Model Name", description="The name of the model."),
    ] = field(default=None)

    description: Annotated[
        Optional[str],
        Meta(title="Model Description", description="A description of the model."),
    ] = field(default=None)

    algorithm: Annotated[
        AlgorithmType,
        Meta(title="Algorithm", description="The algorithm used to train the model."),
    ] = field(default=AlgorithmType.RANDOM_FOREST)

    hyperparameters: Annotated[
        dict[str, Any],
        Meta(
            title="Hyperparameters", description="Hyperparameters used during training."
        ),
    ] = field(default_factory=dict)

    metrics: Annotated[
        dict[str, float],
        Meta(
            title="Evaluation Metrics", description="Evaluation metrics of the model."
        ),
    ] = field(default_factory=dict)

    created_at: Annotated[
        str,
        Meta(
            title="Creation Time", description="Timestamp when the model was created."
        ),
    ] = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

    updated_at: Annotated[
        Optional[str],
        Meta(
            title="Last Updated Time",
            description="Timestamp when the model was last updated.",
        ),
    ] = field(default=None)

    artifacts: Annotated[
        list[str],
        Meta(title="Artifacts", description="List of file paths to model artifacts."),
    ] = field(default_factory=list)

    feature_names: Annotated[
        list[str],
        Meta(
            title="Feature Names",
            description="List of feature names used during training.",
        ),
    ] = field(default_factory=list)

    categorical_columns: Annotated[
        list[CategoricalColumn],
        Meta(
            title="Categorical Columns",
            description="List of categorical columns and their encoders.",
        ),
    ] = field(default_factory=list)

    scaler: Annotated[
        Optional[str],
        Meta(
            title="Scaler", description="Name of the scaler used for feature scaling."
        ),
    ] = field(default=None)

    columns_info: Annotated[
        list[ColumnInfo],
        Meta(
            title="Columns Information",
            description="Information about each column in the dataset.",
        ),
    ] = field(default_factory=list)


class CategoricalColumn(BaseModel):
    """
    Represents a categorical column and its encoding strategy.

    Attributes:
        name (str): Name of the categorical column.
        encoder (Literal["OneHotEncoder", "LabelEncoder", "OrdinalEncoder"]): Encoder to use.
    """

    name: Annotated[
        str,
        Meta(
            title="Column Name",
            description="Name of the categorical column.",
            examples=["gender"],
        ),
    ]

    encoder: Annotated[
        Literal["OneHotEncoder", "LabelEncoder", "OrdinalEncoder"],
        Meta(
            title="Encoder",
            description="Encoder to use for this categorical column.",
            examples=["OneHotEncoder"],
        ),
    ]

    type: Annotated[
        Literal["column", "target"],
        Meta(
            title="Type",
            description="Type of the column.",
            examples=["column"],
        ),
    ] = field(default="column")


class TrainingConfig(BaseModel):
    """
    Configuration for training a machine learning model.

    Attributes:
        gen_ai_assist (bool): If True, use AI assistance for training configuration.
        algorithm (AlgorithmType): The algorithm to be used for training.
        hyperparameters (dict[str, Union[str, int, float, bool]]): Hyperparameters for the model.
        test_size (float): Proportion of the dataset to be used for testing.
        random_state (int): Seed used by the random number generator.
        evaluation_metrics (list[Literal["accuracy", "f1_score", "precision", "recall", "mean_squared_error"]]):
            List of metrics to evaluate the model's performance.
        cross_validation (Optional[int]): Number of folds for cross-validation.
        feature_selection (bool): Flag to perform feature selection.
        imbalance_handling (Optional[Literal["SMOTE", "class_weight"]]): Strategy to handle imbalanced classes.
        target_column (str): Name of the target variable in the dataset.
        categorical_columns (list[CategoricalColumn]): List of categorical columns and their encoders.
        scaler (Optional[Literal["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"]]):
            Name of the scaler to use for feature scaling.
    """

    gen_ai_assist: Annotated[
        bool,
        Meta(
            title="Generative AI Assistance",
            description=(
                "If True, the AI will provide assistance during the training process, "
                "such as choosing the algorithm, hyperparameters, categorical encoding, etc."
            ),
            examples=[True],
        ),
    ] = field(default=False)

    algorithm: Annotated[
        AlgorithmType,
        Meta(
            title="Algorithm",
            description="The algorithm to be used for training.",
            examples=["RandomForestClassifier"],
        ),
    ] = field(default=AlgorithmType.RANDOM_FOREST)

    hyperparameters: Annotated[
        dict[str, Any],
        Meta(
            title="Hyperparameters",
            description="Hyperparameters for the model.",
            examples=[{"n_estimators": 100, "max_depth": 5}],
        ),
    ] = field(default_factory=dict)

    test_size: Annotated[
        float,
        Meta(
            title="Test Size",
            description="Proportion of the dataset to be used for testing.",
            le=1.0,
            gt=0.0,
            examples=[0.2],
        ),
    ] = field(default=0.2)

    random_state: Annotated[
        int,
        Meta(
            title="Random State",
            description="Seed used by the random number generator.",
            examples=[42],
        ),
    ] = field(default=42)

    evaluation_metrics: Annotated[
        list[
            Literal["accuracy", "f1_score", "precision", "recall", "mean_squared_error"]
        ],
        Meta(
            title="Evaluation Metrics",
            description="List of metrics to evaluate the model's performance.",
            examples=[["accuracy", "f1_score"]],
        ),
    ] = field(default_factory=lambda: ["accuracy"])

    cross_validation: Annotated[
        Optional[int],
        Meta(
            title="Cross Validation",
            description="Number of folds for cross-validation.",
            examples=[5],
        ),
    ] = field(default=None)

    feature_selection: Annotated[
        bool,
        Meta(
            title="Feature Selection",
            description="Flag to perform feature selection.",
            examples=[True],
        ),
    ] = field(default=False)

    imbalance_handling: Annotated[
        Optional[Literal["SMOTE", "class_weight"]],
        Meta(
            title="Imbalance Handling",
            description="Strategy to handle imbalanced classes.",
            examples=["SMOTE", "class_weight"],
        ),
    ] = field(default=None)

    target_column: Annotated[
        str,
        Meta(
            title="Target Column",
            description="Name of the target variable in the dataset.",
            examples=["label"],
        ),
    ] = field(default="target")

    categorical_columns: Annotated[
        list[CategoricalColumn],
        Meta(
            title="Categorical Columns",
            description="List of categorical columns and their encoders.",
            examples=[
                [
                    {
                        "name": "gender",
                        "encoder": "OneHotEncoder",
                    },
                    {
                        "name": "country",
                        "encoder": "LabelEncoder",
                    },
                ]
            ],
        ),
    ] = field(default_factory=list)

    scaler: Annotated[
        Optional[
            Literal["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"]
        ],
        Meta(
            title="Scaler",
            description=(
                "Name of the scaler to use for feature scaling. "
                "Options include 'StandardScaler', 'MinMaxScaler', etc."
            ),
            examples=["StandardScaler"],
        ),
    ] = field(default=None)


class TrainingResult(BaseModel):
    """
    Result of the training process.

    Attributes:
        model_uid (str): Unique identifier of the trained model.
        metrics (dict[str, float]): Evaluation metrics of the trained model.
        artifacts (list[str]): Paths to saved model artifacts.
    """

    model_uid: Annotated[
        str,
        Meta(
            title="Model UID",
            description="Unique identifier of the trained model.",
            examples=["model_123456"],
        ),
    ]

    metrics: Annotated[
        dict[str, float],
        Meta(
            title="Evaluation Metrics",
            description="Evaluation metrics of the trained model.",
            examples=[{"accuracy": 0.95, "f1_score": 0.93}],
        ),
    ] = field(default_factory=dict)

    artifacts: Annotated[
        list[str],
        Meta(
            title="Artifacts",
            description="Paths to saved model artifacts.",
            examples=["/path/to/model.joblib", "/path/to/encoder.joblib"],
        ),
    ] = field(default_factory=list)
