import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def sample_data():
    """Fixture to generate synthetic data for testing."""
    df = pd.DataFrame({
        "age": [25, 40, 30, 45],
        "workclass": ["Private", "Self-emp", "Private", "Private"],
        "education": ["Bachelors", "Masters", "Bachelors", "HS-grad"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"]
    })
    return df


def test_process_data_returns_correct_shape(sample_data):
    """
    Test that process_data returns arrays with the correct shape and expected data types.
    Verifies that X is a numpy array with the same number of rows as input data,
    and that y contains only binary values (0s and 1s).
    """
    cat_features = ["workclass", "education"]
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Verify X is a numpy array with expected number of rows
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == sample_data.shape[0]
    # Verify y is binarized (should contain only 0s and 1s)
    assert len(np.unique(y)) <= 2


def test_train_model_returns_random_forest(sample_data):
    """
    Test that train_model returns the expected algorithm type (RandomForestClassifier).
    Ensures the model is properly instantiated and trained.
    """
    cat_features = ["workclass", "education"]
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    model = train_model(X, y)

    # Verify the model is trained and is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)


def test_compute_metrics_returns_valid_values():
    """
    Test that compute_model_metrics returns values within the valid range [0, 1].
    Verifies that precision, recall, and fbeta are floats between 0 and 1.
    """
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Metrics should be floats between 0 and 1
    assert isinstance(precision, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_inference_returns_predictions(sample_data):
    """
    Test that inference returns predictions with correct shape and type.
    Verifies that predictions are a numpy array with the same number of rows
    as the input data and contains only binary values (0 or 1).
    """
    cat_features = ["workclass", "education"]
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)

    # Verify predictions shape and type
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    # Predictions should be binary (0 or 1)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_metrics_exact_values():
    """
    Test that compute_model_metrics returns mathematically correct values.
    With y_true=[0,1,0,1] and y_pred=[0,1,1,1], we have:
    - TP=2 (correctly predicted 1), FP=1 (predicted 1 but was 0), FN=0
    - Precision = TP/(TP+FP) = 2/3 = 0.6667
    - Recall = TP/(TP+FN) = 2/2 = 1.0
    - F1 = 2*(P*R)/(P+R) = 0.8
    """
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Verify exact expected values with tolerance for floating point
    assert abs(precision - 0.6667) < 0.001, f"Expected precision ~0.6667, got {precision}"
    assert recall == 1.0, f"Expected recall 1.0, got {recall}"
    assert 0.79 < fbeta < 0.81, f"Expected F1 ~0.8, got {fbeta}"


def test_process_data_returns_all_artifacts(sample_data):
    """
    Test that process_data returns all required artifacts with correct types.
    Verifies that encoder is OneHotEncoder and lb is LabelBinarizer,
    which are essential for inference in production.
    """
    cat_features = ["workclass", "education"]
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Verify all return types
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert isinstance(encoder, OneHotEncoder), "encoder should be OneHotEncoder"
    assert isinstance(lb, LabelBinarizer), "lb should be LabelBinarizer"
