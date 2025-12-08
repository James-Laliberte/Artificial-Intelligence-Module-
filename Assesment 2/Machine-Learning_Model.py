# Import pandas to work with tables of data
import pandas as pd
# Import Path to handle file paths in a tidy way
from pathlib import Path
# Import tools from scikit learn to split data and build the model
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

# pandas https://pandas.pydata.org/
# pathlib https://docs.python.org/3/library/pathlib.html
# scikit-learn https://scikit-learn.org/stable/


def load_data(csv_path: Path) -> pd.DataFrame:
    # Read the csv file into a DataFrame
    df = pd.read_csv(csv_path)
    # Make a copy so we do not change the original data by mistake
    df = df.copy()
    # Add a simple id so each patient has a number we can look up later
    df["patient_id"] = range(1, len(df) + 1)
    # Print the shape so we can check how many rows and columns we have
    print(f"Loaded dataset from {csv_path} with shape {df.shape}")
    return df


def add_target_column(df: pd.DataFrame, target_col: str = "has_heart_disease") -> pd.DataFrame:
    # Work on a copy again to keep the input safe
    df = df.copy()
    # Create the target column where 1 means some heart disease and 0 means none
    df[target_col] = (df["num"] > 0).astype(int)
    return df


def prepare_features_and_target(df: pd.DataFrame, target_col: str):
    # List of columns that should be treated as categories
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

    # List of columns that should be treated as numbers
    numeric_features = [
        "age",
        "trestbps",
        "chol",
        "thalch",
        "oldpeak",
        "ca",
    ]

    # Put the columns together in the order they will be used for the model
    feature_cols = categorical_features + numeric_features

    # X holds the input features and y holds the target values
    X = df[feature_cols]
    y = df[target_col]

    # Return everything we need for later steps
    return X, y, categorical_features, numeric_features


def build_decision_tree_model(categorical_features, numeric_features):
    # Set up preprocessing so that category columns are one hot encoded
    # and numeric columns are passed through as they are
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    # Create a decision tree with a fixed depth and minimum leaf size
    tree = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        class_weight={0: 1.0, 1: 2.0},
        random_state=42,
    )

    # Combine preprocessing and the tree into one pipeline
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", tree),
        ]
    )

    return model


def train_and_evaluate_model(X, y, categorical_features, numeric_features):
    # Split the data into train and test parts
    # Stratify keeps the same balance of classes in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Show how many rows ended up in each split
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Build the pipeline with preprocessing and the decision tree
    model = build_decision_tree_model(categorical_features, numeric_features)
    # Fit the model on the training data
    model.fit(X_train, y_train)

    # If the model can give probabilities then use them with a custom threshold
    if hasattr(model, "predict_proba"):
        # Get predicted probabilities for the test set
        y_proba_test = model.predict_proba(X_test)[:, 1]
        # Set the decision threshold lower than the default 0.5
        # This makes the model more sensitive to disease cases
        threshold = 0.40
        y_pred_test = (y_proba_test >= threshold).astype(int)
        print(f"\nUsing custom decision threshold of {threshold:.2f} for 'Disease'.")
    else:
        # If probabilities are not available fall back to standard predictions
        y_pred_test = model.predict(X_test)
        y_proba_test = None

    # Work out the main performance scores on the test set
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba_test) if y_proba_test is not None else float("nan")

    # Print the metrics so we can interpret how the model is doing
    print("\nGlobal Model Performance:")
    print(f"Accuracy:      {accuracy:.3f}")
    print(f"Precision:     {precision:.3f}")
    print(f"Recall:        {recall:.3f}")
    print(f"F1-score:      {f1:.3f}")
    print(f"ROC-AUC:       {roc_auc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred_test, target_names=["No disease", "Disease"]))

    # Return the trained model so it can be used later
    return model


def compute_feature_weightings(model, categorical_features, numeric_features) -> pd.DataFrame:
    # Get the preprocessing part from the pipeline
    preprocess: ColumnTransformer = model.named_steps["preprocess"]
    # Get the fitted decision tree from the pipeline
    tree: DecisionTreeClassifier = model.named_steps["model"]

    # Get the names of all one hot encoded category columns
    ohe: OneHotEncoder = preprocess.named_transformers_["cat"]
    encoded_cat_names = ohe.get_feature_names_out(categorical_features)

    # Join the encoded names and the numeric column names
    all_feature_names = list(encoded_cat_names) + list(numeric_features)

    # Pull out the feature importance scores from the tree
    importances = tree.feature_importances_

    # Put feature names and scores into a DataFrame and sort by importance
    importance_df = pd.DataFrame(
        {
            "feature": all_feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    # Print the feature importances to help with interpretation
    print("\nLearned Feature Importances (Decision Tree):")
    for _, row in importance_df.iterrows():
        print(f"{row['feature']:<45} {row['importance']:.4f}")

    return importance_df


def probability_to_risk_category(prob: float) -> str:
    # Turn a probability into a simple risk group label
    if prob < 0.25:
        return "low risk"
    elif prob < 0.50:
        return "mild risk"
    elif prob < 0.75:
        return "high risk"
    else:
        return "likely has heart disease"


def search_and_predict_loop(df: pd.DataFrame, model, categorical_features, numeric_features):
    # Use the same feature columns as in training
    feature_cols = categorical_features + numeric_features

    # Simple text menu for the user so they know how to look up patients
    print("\n------------------------------------------------")
    print(" Search interface")
    print(" Type a patient number (line number) to view their predicted risk.")
    print(f" Valid range: 1 to {len(df)}")
    print(" Type 'quit' to exit.")
    print("------------------------------------------------")

    while True:
        # Ask the user which patient they want to inspect
        query = input("\nEnter patient number (or 'quit'): ").strip()
        if query.lower() == "quit":
            # Leave the loop if the user is finished
            break

        if not query:
            # Handle empty input and remind the user what to do
            print("Please type a number or 'quit'.")
            continue

        try:
            # Try to turn the input into an integer
            patient_num = int(query)
        except ValueError:
            # If this fails tell the user and ask again
            print("Please enter a whole number (integer) for the patient.")
            continue

        # Check that the number is in the valid range
        if not (1 <= patient_num <= len(df)):
            print(f"Patient number out of range. Please enter a number between 1 and {len(df)}.")
            continue

        # Select the row for the chosen patient
        row = df[df["patient_id"] == patient_num].iloc[0]

        # Get the true label from the data
        actual_label = row["has_heart_disease"]
        # Build a one row DataFrame with the same feature columns used in training
        X_patient = row[feature_cols].to_frame().T

        # Use the model to get the predicted probability for this patient
        if hasattr(model, "predict_proba"):
            prob_heart_disease = model.predict_proba(X_patient)[0, 1]
        else:
            prob_heart_disease = float("nan")

        # Map the probability to a risk category label
        risk_category = probability_to_risk_category(prob_heart_disease)

        # Print all the key information for this patient
        print("----------------------------------------------")
        print(f"Patient number (line): {patient_num}")
        print(f"Actual has_heart_disease (0/1): {actual_label}")
        print(f"Predicted probability (disease): {prob_heart_disease:.3f}")
        print(f"Predicted risk category: {risk_category}")


def main():
    # Work out the folder where this script lives
    base_dir = Path(__file__).resolve().parent
    # Build the full path to the csv file
    csv_path = base_dir / "heart_disease_uci.csv"

    # Load the data from the csv file
    df = load_data(csv_path)
    # Add the binary target column to the DataFrame
    df = add_target_column(df, target_col="has_heart_disease")
    # Prepare features and target and record which columns are categorical or numeric
    X, y, cat_features, num_features = prepare_features_and_target(df, target_col="has_heart_disease")

    # Train the model and show the evaluation results
    model = train_and_evaluate_model(X, y, cat_features, num_features)

    # Work out and display how important each feature is to the tree
    _ = compute_feature_weightings(model, cat_features, num_features)

    # Start the interactive search loop so a user can inspect individual patients
    search_and_predict_loop(df, model, cat_features, num_features)


if __name__ == "__main__":
    # Only run main when this file is run directly
    main()