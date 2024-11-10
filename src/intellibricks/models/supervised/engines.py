import abc
import base64
import binascii
import datetime
import io
import json
import os
from typing import Annotated, Any, Optional, Sequence

import magic
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from typing_extensions import Doc
from weavearc import ReadAllResult, ReadResult
from weavearc.data import AsyncRepository
from weavearc.logging import logger
from weavearc.utils.creators import DynamicInstanceCreator

from intellibricks.llms import CompletionEngineProtocol

from ..data.entities import ForgedModel
from ..data.schema import (
    TrainingConfig,
    TrainingResult,
)
from ..data.value_objects import ColumnInfo
from ..infra.constants import AlgorithmType
from ..infra.exceptions import (
    InvalidBase64Exception,
    InvalidFileException,
    MissingColumnsException,
    TargetColumnNotFoundException,
)


class SupervisedLearningEngine(abc.ABC):
    """
    Protocol for SKLearnSupervisedLearningEngine classes.

    This protocol defines the contract for training and using supervised machine learning models.
    Any class implementing this protocol should provide concrete implementations for the defined methods.
    """

    completion_engine: Annotated[
        CompletionEngineProtocol,
        Doc("Completion Engine instance to use when AI assistance is activated."),
    ]

    repository: Annotated[
        AsyncRepository[ForgedModel],
        Doc("Repository to store the trained models."),
    ]

    @abc.abstractmethod
    async def train(
        self,
        *,
        b64_file: str,
        uid: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingResult:
        """
        Train a machine learning model based on the provided configuration and dataset.

        Args:
            b64_file (str): Base64-encoded dataset file.
            uid (str): Unique identifier for the model.
            name (Optional[str]): Name of the model.
            description (Optional[str]): Description of the model.
            config (Optional[TrainingConfig]): Configuration for training.

        Returns:
            TrainingResult: Result of the training process.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        ...

    @abc.abstractmethod
    async def get_model(self, model_id: str) -> ForgedModel: ...

    @abc.abstractmethod
    async def get_models(self) -> Sequence[ForgedModel]: ...

    @abc.abstractmethod
    async def predict(
        self,
        uid: str,
        input_data: dict[str, Any],
    ) -> np.ndarray:
        """
        Load a model by UID and make predictions on new data.

        Args:
            uid (str): The unique identifier of the model.
            input_data (dict[str, Any]): Input data as a dictionary.

        Returns:
            np.ndarray: Predictions made by the model.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        ...


class SKLearnSupervisedLearningEngine(SupervisedLearningEngine):
    """
    The Forge class provides an interface to train and use machine learning models dynamically.

    Example:
        forge = Forge(completion_engine=ObservableCompletionEngine(), repository=LocalSupervisedModelRepository())
        training_result = await forge.train(
            b64_file=base64_encoded_dataset,
            uid="model_uid",
            name="My Model",
            description="A model to predict something",
            config=TrainingConfig(
                algorithm=AlgorithmType.RANDOM_FOREST,
                hyperparameters={"n_estimators": 100, "max_depth": 5},
                target_column="target",
                categorical_columns=[
                    CategoricalColumn(name="category", encoder="OneHotEncoder")
                ],
                scaler="StandardScaler",
            ),
        )
    """

    async def train(
        self,
        *,
        b64_file: str,
        uid: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingResult:
        """
        Train a machine learning model based on the provided configuration and dataset.

        Args:
            b64_file (str): Base64-encoded dataset file.
            uid (str): Unique identifier for the model.
            name (Optional[str]): Name of the model.
            description (Optional[str]): Description of the model.
            config (Optional[TrainingConfig]): Configuration for training.

        Returns:
            TrainingResult: Result of the training process.

        Raises:
            ValueError: If TrainingConfig is not provided.
        """
        df: pd.DataFrame = await self._get_df(b64_file)

        if config is None:
            raise ValueError("TrainingConfig must be provided.")

        if config.gen_ai_assist:
            config = await self._get_ai_assisted_config(df, config)

        try:
            X = df.drop(columns=[config.target_column])
        except KeyError:
            try:
                df = await self._get_df(b64_file, sep=";")
                X = df.drop(columns=[config.target_column])
            except KeyError as e:
                raise TargetColumnNotFoundException(
                    f'Target column "{config.target_column}" not found in dataset. Available columns: {[col for col in df.columns]}'
                ) from e

        y = df[config.target_column]

        logger.debug("Preparing preprocessing pipeline.")
        preprocessor = self._build_preprocessor(config, X)

        logger.debug("Building model pipeline.")
        model_pipeline = self._build_model_pipeline(config, preprocessor)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )

        logger.debug("Training model.")
        model_pipeline.fit(X_train, y_train)

        logger.debug("Evaluating model.")
        metrics = self._evaluate_model(
            model_pipeline, X_test, y_test, config.evaluation_metrics, config.algorithm
        )

        # Collect dataset information
        columns_info = []
        for col in df.columns:
            sample_value = df[col].iloc[0] if not df[col].empty else None
            dtype = str(df[col].dtype)
            # Restrict dtype to allowed Literals
            dtype_literal = (
                dtype
                if dtype in ["int64", "float64", "object", "bool", "datetime64"]
                else "object"
            )
            columns_info.append(
                ColumnInfo(name=col, dtype=dtype_literal, sample_value=sample_value)
            )

        forged_model = ForgedModel(
            uid=uid,
            name=name,
            description=description,
            algorithm=config.algorithm,
            hyperparameters=config.hyperparameters,
            metrics=metrics,
            created_at=datetime.datetime.utcnow().isoformat(),
            artifacts=[],
            feature_names=list(X.columns),
            target_name=config.target_column,
            categorical_columns=config.categorical_columns,
            scaler=config.scaler,
            columns_info=columns_info,
        )

        logger.debug("Saving model artifacts.")
        artifacts = self._save_model_artifacts(uid, model_pipeline, forged_model)
        forged_model.artifacts = artifacts

        logger.debug("Storing model metadata.")
        await self.repository.create(forged_model)

        return TrainingResult(
            model_uid=uid,
            metrics=metrics,
            artifacts=artifacts,
        )

    async def get_model(self, model_id: str) -> ForgedModel:
        result: ReadResult[ForgedModel] = await self.repository.read(q=model_id)
        return result.entity

    async def get_models(self) -> Sequence[ForgedModel]:
        result: ReadAllResult[ForgedModel] = await self.repository.read_all()
        return result.entities

    async def predict(
        self,
        uid: str,
        input_data: dict[str, Any],
    ) -> np.ndarray:
        """
        Load a model by UID and make predictions on new data.

        Args:
            uid (str): The unique identifier of the model.
            input_data (dict[str, Any]): Input data as a dictionary.

        Returns:
            np.ndarray: Predictions made by the model.

        Raises:
            ValueError: If model is not found or cannot be loaded.
        """
        result = await self.repository.read(uid)

        model: ForgedModel = result.entity  # model is a ForgedModel instance

        model_dir = os.path.join("cortex", "core", "models", "store", uid)
        model_path = os.path.join(model_dir, "pipe.joblib")
        import joblib

        try:
            model_pipeline: Pipeline = joblib.load(model_path)
            input_df = pd.DataFrame([input_data])

            # Ensure columns match
            missing_cols = set(model.feature_names) - set(input_df.columns)
            if missing_cols:
                raise MissingColumnsException(f"{missing_cols}")

            # Reorder columns to match training data
            input_df = input_df[model.feature_names]

            predictions = model_pipeline.predict(input_df)
            return predictions
        except (FileNotFoundError, NotFittedError) as e:
            raise ValueError(
                f"Model artifacts not found or model not fitted: {str(e)}"
            ) from e

    async def _get_df(self, b64_file: str, sep: Optional[str] = None) -> pd.DataFrame:
        """
        Decode base64-encoded data and load it into a pandas DataFrame.

        Args:
            b64_file (str): Base64-encoded data.

        Returns:
            pd.DataFrame: Loaded DataFrame.

        Raises:
            InvalidBase64Exception: If the base64 encoding is invalid.
            InvalidFileException: If the file type is unsupported or cannot be loaded.
        """
        try:
            file_bytes: bytes = base64.b64decode(b64_file)
        except binascii.Error as e:
            raise InvalidBase64Exception("Invalid base64 encoding") from e

        # Guess the MIME type
        f = magic.Magic(mime=True)
        mime = f.from_buffer(file_bytes)

        logger.debug("mime: {mime}", mime=mime)

        # Load the data into pandas based on MIME type
        df_loaders = {
            "text/csv": lambda: pd.read_csv(io.BytesIO(file_bytes), sep=sep),
            "application/json": lambda: pd.read_json(io.BytesIO(file_bytes), sep=sep),
            "text/plain": lambda: pd.read_csv(io.BytesIO(file_bytes), sep=sep),
            "application/vnd.ms-excel": lambda: pd.read_excel(
                io.BytesIO(file_bytes), sep=sep
            ),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": lambda: pd.read_excel(
                io.BytesIO(file_bytes), sep=sep
            ),
            "application/parquet": lambda: pd.read_parquet(
                io.BytesIO(file_bytes), sep=sep
            ),
            "application/x-hdf": lambda: pd.read_hdf(io.BytesIO(file_bytes), sep=sep),
        }

        df_loader = df_loaders.get(mime)
        if not df_loader:
            raise InvalidFileException(f"Unsupported MIME type: {mime}")

        try:
            df: pd.DataFrame = df_loader()
        except Exception as e:
            raise InvalidFileException(f"Error loading data: {str(e)}") from e

        return df

    async def _get_ai_assisted_config(
        self, df: pd.DataFrame, config: TrainingConfig
    ) -> TrainingConfig:
        """
        Use the ObservableCompletionEngine to get AI-assisted training configuration.

        Args:
            df (pd.DataFrame): The dataset as a pandas DataFrame.
            config (TrainingConfig): The initial training configuration.

        Returns:
            TrainingConfig: Updated training configuration.

        Raises:
            ValueError: If AI-assisted configuration cannot be obtained.
        """
        prompt = (
            "You are a data science assistant. Analyze the following dataset and suggest the best algorithm, hyperparameters, "
            "and preprocessing steps. Provide a suggested configuration for the training process based on the dataset."
            "Keep in mind that the library beein used is scikit-learn. If you see that the column has a numeric value,"
            "there is no need to provide an encoder for it."
        )
        data_preview = df.head().to_json()

        ai_input = f"{prompt}\nDataset Preview:\n{data_preview}"

        response = await self.completion_engine.complete_async(
            prompt=ai_input,
            system_prompt="""
            You are an AI assistant designed to provide detailed, step-by-step responses. Your outputs should follow this structure:

            1. Begin with a <thinking> section.
            2. Inside the thinking section:
            a. Briefly analyze the question and outline your approach.
            b. Present a clear plan of steps to solve the problem.
            c. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps.
            3. Include a <reflection> section for each idea where you:
            a. Review your reasoning.
            b. Check for potential errors or oversights.
            c. Confirm or adjust your conclusion if necessary.
            4. Be sure to close all reflection sections.
            5. Close the thinking section with </thinking>.
            6. Provide your final answer in an <output> section.
            7. Close the final answer with </output>.

            Always use these tags in your responses. Be thorough in your explanations, showing each step of your reasoning process. Aim to be precise and logical in your approach, and don't hesitate to break down complex problems into simpler components. Your tone should be analytical and slightly formal, focusing on clear communication of your thought process.

            Remember: <thinking>, <reflection> and <output> MUST be tags and must be closed at their conclusion

            Make sure all <tags> are on separate lines with no other text. Do not include other text on a line containing a tag.
            """,
            response_format=TrainingConfig,
        )

        ai_config: Optional[TrainingConfig] = response.get_parsed()
        if ai_config is not None:
            logger.debug(
                "AI-assisted config received: {ai_config}", ai_config=ai_config
            )
            config = ai_config
        else:
            logger.error("Failed to get AI-assisted config")
            raise ValueError(
                "Failed to get AI-assisted config from ObservableCompletionEngine"
            )

        return config

    def _build_preprocessor(
        self, config: TrainingConfig, X: pd.DataFrame
    ) -> ColumnTransformer:
        """
        Build the preprocessing pipeline based on the training configuration.

        Args:
            config (TrainingConfig): Training configuration.
            X (pd.DataFrame): Feature DataFrame.

        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """
        transformers = []

        # Handle categorical columns
        if config.categorical_columns:
            categorical_transformers = []
            for cat_col in config.categorical_columns:
                encoder_cls = DynamicInstanceCreator.create_from_module(
                    "sklearn.preprocessing", cat_col.encoder, return_class_only=True
                )
                categorical_transformers.append(
                    (cat_col.name, encoder_cls(), [cat_col.name])
                )
            if categorical_transformers:
                transformers.append(
                    (
                        "categorical",
                        Pipeline(steps=categorical_transformers),
                        [col.name for col in config.categorical_columns],
                    )
                )

        # Handle numerical columns
        numerical_columns = [
            col
            for col in X.columns
            if col not in [cat_col.name for cat_col in config.categorical_columns]
        ]
        if config.scaler and numerical_columns:
            scaler_cls = DynamicInstanceCreator.create_from_module(
                "sklearn.preprocessing", config.scaler, return_class_only=True
            )
            transformers.append(("numerical", scaler_cls(), numerical_columns))

        preprocessor = ColumnTransformer(
            transformers=transformers, remainder="passthrough"
        )

        return preprocessor

    def _build_model_pipeline(
        self, config: TrainingConfig, preprocessor: ColumnTransformer
    ) -> Pipeline:
        """
        Build the machine learning model pipeline.

        Args:
            config (TrainingConfig): Training configuration.
            preprocessor (ColumnTransformer): Preprocessing pipeline.

        Returns:
            Pipeline: Machine learning model pipeline.
        """
        # Map AlgorithmType to appropriate module
        algorithm_module_map = {
            AlgorithmType.RANDOM_FOREST: "sklearn.ensemble",
            AlgorithmType.SVM: "sklearn.svm",
            AlgorithmType.LOGISTIC_REGRESSION: "sklearn.linear_model",
            AlgorithmType.LINEAR_REGRESSION: "sklearn.linear_model",
            AlgorithmType.GRADIENT_BOOSTING: "sklearn.ensemble",
            AlgorithmType.KNN: "sklearn.neighbors",
            AlgorithmType.DECISION_TREE: "sklearn.tree",
        }

        module_name = algorithm_module_map.get(config.algorithm)
        if not module_name:
            raise ValueError(f"Algorithm {config.algorithm.value} is not supported.")

        algorithm_cls = DynamicInstanceCreator.create_from_module(
            module_name, config.algorithm.value, return_class_only=True
        )

        model_instance = algorithm_cls(**config.hyperparameters)

        steps = [
            ("preprocessor", preprocessor),
            ("model", model_instance),
        ]

        pipeline = Pipeline(steps=steps)

        return pipeline

    def _evaluate_model(
        self,
        model_pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics_list: Sequence[str],
        algorithm: AlgorithmType,
    ) -> dict[str, float]:
        """
        Evaluate the trained model using the specified metrics.

        Args:
            model_pipeline (Pipeline): Trained model pipeline.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test labels.
            metrics_list (Sequence[str]): List of metrics to compute.
            algorithm (AlgorithmType): Algorithm used.

        Returns:
            dict[str, float]: Computed metrics.
        """
        y_pred = model_pipeline.predict(X_test)

        metrics: dict[str, float] = {}
        for metric_name in metrics_list:
            if metric_name == "accuracy":
                metrics["accuracy"] = accuracy_score(y_test, y_pred)
            elif metric_name == "f1_score":
                metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")
            elif metric_name == "mean_squared_error":
                metrics["mean_squared_error"] = mean_squared_error(y_test, y_pred)
            elif metric_name == "precision":
                metrics["precision"] = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            elif metric_name == "recall":
                metrics["recall"] = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            # Add more metrics as needed

        return metrics

    def _save_model_artifacts(
        self,
        uid: str,
        model_pipeline: Pipeline,
        forged_model: ForgedModel,
    ) -> list[str]:
        """
        Save model artifacts to disk.

        Args:
            uid (str): Unique identifier for the model.
            model_pipeline (Pipeline): Trained model pipeline.
            forged_model (ForgedModel): Model information and metadata.

        Returns:
            list[str]: Paths to saved artifacts.
        """
        model_dir = os.path.join("cortex", "core", "models", "supervised", "store", uid)
        os.makedirs(model_dir, exist_ok=True)

        artifacts: list[str] = []

        # Save the entire pipeline
        model_path = os.path.join(model_dir, "pipe.joblib")
        import joblib

        joblib.dump(model_pipeline, model_path)
        artifacts.append(model_path)

        # Save model info
        model_info_path = os.path.join(model_dir, "model_info.json")
        with open(model_info_path, "w") as f:
            json.dump(forged_model.as_dict(), f, indent=4)
        artifacts.append(model_info_path)

        return artifacts
