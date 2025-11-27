import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


# =====================================================================
# 1. DATA LOADING & PREPARATION
# =====================================================================

def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the heart disease dataset from a CSV file.

    We also add a simple 'patient_id' that mirrors the row number (1..N),
    so we can later look up patients by their line number in the file.
    """
    df = pd.read_csv(csv_path)

    # Make a copy to be safe (avoid editing the original in place)
    df = df.copy()

    # Create IDs 1, 2, 3, ..., number of rows
    df["patient_id"] = range(1, len(df) + 1)

    print(f"Loaded dataset from {csv_path} with shape {df.shape}")
    return df


def add_target_column(df: pd.DataFrame, target_col: str = "has_heart_disease") -> pd.DataFrame:
    """
    Add a new binary target column representing the *presence* of heart disease.

    In the UCI heart disease dataset, the column 'num' encodes:
        0 = no heart disease
        1..4 = varying levels of disease severity

    For this research question, we simplify to a binary classification:
        has_heart_disease = 1 if num > 0 else 0
    """
    df = df.copy()
    df[target_col] = (df["num"] > 0).astype(int)
    return df


def prepare_features_and_target(df: pd.DataFrame, target_col: str):
    """
    Choose which columns are used as inputs (features) and which is the target.

    Target:
        - 'has_heart_disease' (0 = no disease, 1 = disease)

    Features:
        Routine clinical measurements and basic patient information.
    """
    # Categorical features (text / boolean-like fields)
    categorical_features = [
        "sex",
        "dataset",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "thal",
    ]

    # Numeric features (continuous measurements)
    numeric_features = [
        "age",
        "trestbps",
        "chol",
        "thalch",
        "oldpeak",
        "ca",
    ]

    feature_cols = categorical_features + numeric_features

    X = df[feature_cols]         # input features
    y = df[target_col]           # values we want to predict (0 or 1)

    return X, y, categorical_features, numeric_features


# =====================================================================
# 2. MODEL BUILDING
# =====================================================================

def build_decision_tree_model(categorical_features, numeric_features):
    """
    Build a simple ML pipeline for heart disease prediction.

    - OneHotEncoder: turns categorical variables (e.g. sex, chest pain type) into
      numeric indicators the model can use.
    - 'passthrough' for numeric features: decision trees handle raw numeric
      values reasonably well without scaling.
    - DecisionTreeClassifier: learns to predict whether a patient has heart
      disease based on these features.

    This model is relatively interpretable (we can inspect feature importances),
    but may not be as accurate as more complex models (e.g. random forests).
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    # Decision tree for classification:
    # - max_depth controls how "deep" the tree can grow.
    #   Lower depth = simpler model, more interpretable, less chance of overfitting.
    tree = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )

    # Combine preprocessing and model into one pipeline
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", tree),
        ]
    )

    return model


# =====================================================================
# 3. TRAINING & EVALUATION
# =====================================================================

def train_and_evaluate_model(X, y, categorical_features, numeric_features):
    """
    Train the decision tree classifier and show how well it does overall.

    Steps:
      - split data into train and test sets (with stratification)
      - fit the model on the training data
      - test it on the test set
      - print common classification metrics

    These metrics help us assess how effectively the model predicts the presence
    of heart disease, and give us a baseline to discuss accuracy vs interpretability.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,   # keep class balance similar in train/test
    )

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    model = build_decision_tree_model(categorical_features, numeric_features)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_test = model.predict(X_test)

    # For ROC-AUC we need predicted probabilities for the positive class (1)
    if hasattr(model, "predict_proba"):
        y_proba_test = model.predict_proba(X_test)[:, 1]
    else:
        y_proba_test = None

    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba_test) if y_proba_test is not None else float("nan")

    print("\n=== Global Model Performance (Decision Tree Classifier) ===")
    print(f"Accuracy:      {accuracy:.3f}")
    print(f"Precision:     {precision:.3f}")
    print(f"Recall:        {recall:.3f}")
    print(f"F1-score:      {f1:.3f}")
    print(f"ROC-AUC:       {roc_auc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred_test, target_names=["No disease", "Disease"]))

    # Return the trained model so other parts of the program can use it
    return model


# =====================================================================
# 4. FEATURE IMPORTANCE (MODEL "WEIGHTING")
# =====================================================================

def compute_feature_weightings(model, categorical_features, numeric_features) -> pd.DataFrame:
    """
    Show how important each feature is to the decision tree.

    Decision trees do not have "weights" in the same sense as linear models.
    Instead, they expose feature_importances_:

        "How much did this feature help reduce classification error in the tree?"

    Higher importance values mean the feature contributed more to splitting
    the data into correct classes. This is one way to discuss interpretability.
    """
    # Get the internal bits of the pipeline
    preprocess: ColumnTransformer = model.named_steps["preprocess"]
    tree: DecisionTreeClassifier = model.named_steps["model"]

    # Get the generated one-hot feature names
    ohe: OneHotEncoder = preprocess.named_transformers_["cat"]
    encoded_cat_names = ohe.get_feature_names_out(categorical_features)

    # Numeric features are passed through unchanged, and come after the encoded cats
    all_feature_names = list(encoded_cat_names) + list(numeric_features)

    importances = tree.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": all_feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    print("\n=== Learned Feature Importances (Decision Tree) ===")
    for _, row in importance_df.iterrows():
        print(f"{row['feature']:<45} {row['importance']:.4f}")

    return importance_df


# =====================================================================
# 5. SIMPLE INTERACTIVE LOOP
# =====================================================================

def search_and_predict_loop(df: pd.DataFrame, model, categorical_features, numeric_features):
    """
    Simple text interface so we can "look up" a patient by their line number.

    You type a patient number (which is just the row number in the CSV, stored
    in 'patient_id'), and the code shows:
      - the actual diagnosis (has_heart_disease: 0/1)
      - the prediction from the decision tree
      - the model's estimated probability of heart disease

    This helps you see individual-level predictions and think about
    interpretability vs clinical usefulness.
    """
    feature_cols = categorical_features + numeric_features

    print("\n==============================================")
    print(" Heart Disease Prediction Interface (Decision Tree)")
    print(" Type a patient number (line number) to view their predicted label.")
    print(f" Valid range: 1 to {len(df)}")
    print(" Type 'quit' to exit.")
    print("==============================================")

    while True:
        query = input("\nEnter patient number (or 'quit'): ").strip()
        if query.lower() == "quit":
            print("Exiting. Goodbye!")
            break

        if not query:
            print("Please type a number or 'quit'.")
            continue

        # Try to turn the input into an integer
        try:
            patient_num = int(query)
        except ValueError:
            print("Please enter a whole number (integer) for the patient.")
            continue

        # Check that the number is within the correct range
        if not (1 <= patient_num <= len(df)):
            print(f"Patient number out of range. Please enter a number between 1 and {len(df)}.")
            continue

        # Grab the row for this patient_id
        row = df[df["patient_id"] == patient_num].iloc[0]

        actual_label = row["has_heart_disease"]
        X_patient = row[feature_cols].to_frame().T

        predicted_label = model.predict(X_patient)[0]

        if hasattr(model, "predict_proba"):
            prob_heart_disease = model.predict_proba(X_patient)[0, 1]
        else:
            prob_heart_disease = float("nan")

        print("----------------------------------------------")
        print(f"Patient number (line):          {patient_num}")
        print(f"Actual has_heart_disease:       {actual_label}  (0 = no, 1 = yes)")
        print(f"Predicted has_heart_disease:    {predicted_label}  (0 = no, 1 = yes)")
        print(f"Predicted probability (disease):{prob_heart_disease:.3f}")


# =====================================================================
# 6. MAIN SCRIPT
# =====================================================================

def main():
    """
    - Load the data from heart_disease_uci.csv.
    - Create a binary 'has_heart_disease' target we want to predict.
    - Train a Decision Tree classification model (machine learning).
    - Show which features the model found most important (its "weighting").
    - Let the user type a patient number and see that patient's predicted label.

    This supports the research question:
    "How effectively can machine learning algorithms predict the presence of
     heart disease from routine clinical measurements, and what trade-offs
     exist between accuracy and model interpretability?"
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "heart_disease_uci.csv"

    # Load and prepare the data
    df = load_data(csv_path)
    df = add_target_column(df, target_col="has_heart_disease")
    X, y, cat_features, num_features = prepare_features_and_target(df, target_col="has_heart_disease")

    # Train the ML model
    model = train_and_evaluate_model(X, y, cat_features, num_features)

    # Show how the model is weighting the input features
    _ = compute_feature_weightings(model, cat_features, num_features)

    # Interactive search by patient number / line
    search_and_predict_loop(df, model, cat_features, num_features)


if __name__ == "__main__":
    main()
