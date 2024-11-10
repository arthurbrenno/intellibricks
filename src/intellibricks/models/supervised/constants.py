from enum import Enum


class AlgorithmType(str, Enum):
    RANDOM_FOREST = "RandomForestClassifier"
    SVM = "SVM"
    LOGISTIC_REGRESSION = "LogisticRegression"
    LINEAR_REGRESSION = "LinearRegression"
    GRADIENT_BOOSTING = "GradientBoosting"
    KNN = "KNN"
    DECISION_TREE = "DecisionTree"
