"""
The `models` module provides a robust interface for training and using machine learning models dynamically.

This module allows users to:

- Train machine learning models using various algorithms with customizable configurations.
- Utilize Generative AI Assistance to automatically determine the best algorithm and hyperparameters.
- Save trained models along with preprocessing steps for future predictions.
- Load and use existing models to make predictions on new data.

Key Classes:

- `Forge`: The main class providing methods to train and predict using machine learning models.
- `TrainingConfig`: Configuration class specifying training parameters.
- `TrainingResult`: Result class containing information about the trained model.
- `CategoricalColumn`: Class representing a categorical column and its encoding strategy.

Examples:
    Training a model:
    ```python
    from intellibricks.models.forge import Forge
    from intellibricks.models.repositories.impl import LocalSupervisedModelRepository
    from intellibricks.models.schema import TrainingConfig, CategoricalColumn
    from intellibricks.llms import CompletionEngine

    # Initialize Forge
    forge = Forge(
        completion_engine=CompletionEngine(),
        repository=LocalSupervisedModelRepository(),
    )

    # Base64 encode your dataset
    with open('dataset.csv', 'rb') as f:
        b64_file = base64.b64encode(f.read()).decode('utf-8')

    # Define training configuration
    config = TrainingConfig(
        gen_ai_assist=False,
        algorithm=AlgorithmType.RANDOM_FOREST,
        hyperparameters={'n_estimators': 100, 'max_depth': 5},
        target_column='target',
        categorical_columns=[CategoricalColumn(name='category', encoder='OneHotEncoder')],
        scaler='StandardScaler',
        evaluation_metrics=['accuracy', 'f1_score'],
    )

    # Train the model
    training_result = await forge.train(
        b64_file=b64_file,
        uid='model_uid_123',
        name='My Random Forest Model',
        description='A model to predict something',
        config=config,
    )
    ```

    Making predictions with a trained model:
    ```python
    # Prepare input data for prediction
    input_data = {
        'feature1': 10,
        'category': 'A',
        'feature3': 5.5,
        # ... other features
    }

    # Make predictions with the trained model
    predictions = await forge.predict(
        uid='model_uid_123',
        input_data=input_data,  # Input data as a dictionary
    )

    print(predictions)
    ```

The `Forge` class handles the entire pipeline, including data preprocessing, model training, evaluation, and saving artifacts for later use. The models are stored locally in a structured directory with all relevant artifacts and metadata, facilitating easy retrieval and use for predictions.

Dependencies:

- scikit-learn: For machine learning algorithms and preprocessing.
- pandas: For data manipulation.
- msgspec: For data validation and serialization.
- joblib: For saving and loading model artifacts.
- CompletionEngine: For AI-assisted configuration when enabled.

Note:

- Ensure that the necessary exception handling and validation are in place.
- The `CompletionEngine` should be properly configured and integrated if Generative AI Assistance is used.

"""

from .constants import AlgorithmType
from .exceptions import InvalidBase64Exception, InvalidFileException
from .engines import SKLearnSupervisedLearningEngine, SupervisedLearningEngine
from .repositories import LocalSupervisedModelRepository
from .schema import CategoricalColumn, TrainingConfig, TrainingResult, ForgedModel

__all__: list[str] = [
    "SKLearnSupervisedLearningEngine",
    "TrainingConfig",
    "TrainingResult",
    "CategoricalColumn",
    "AlgorithmType",
    "InvalidBase64Exception",
    "InvalidFileException",
    "LocalSupervisedModelRepository",
    "ForgedModel",
    "SupervisedLearningEngine",
]
