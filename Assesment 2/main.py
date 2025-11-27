import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the student performance dataset from a CSV file.
    Also add a simple 'student_id' that matches the row number (1..N),
    so we can later look up students by their line number.
    """
    df = pd.read_csv(csv_path)

    # Make a copy to be safe (avoid editing the original in place)
    df = df.copy()

    # Create IDs 1, 2, 3, ..., number of rows
    df["student_id"] = range(1, len(df) + 1)

    print(f"Loaded dataset from {csv_path} with shape {df.shape}")
    return df


def add_target_column(df: pd.DataFrame, target_col: str = "final_grade") -> pd.DataFrame:
    """
    Add a new column that represents the 'final grade' we want to predict.

    Here we keep it simple:
    final_grade = average of math, reading and writing scores.
    """
    df = df.copy()
    df[target_col] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    return df


def prepare_features_and_target(df: pd.DataFrame, target_col: str):
    """
    Choose which columns are used as inputs (features) and which is the target.

    We only use background / preparation info as inputs:
    - gender
    - race/ethnicity
    - parental education
    - lunch type
    - test preparation course

    The target is the final_grade we just created.
    """
    feature_cols = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ]

    X = df[feature_cols]   # input features
    y = df[target_col]     # values we want to predict (final grade)

    return X, y, feature_cols


def build_decision_tree_model(categorical_features):
    """
    Build a small machine learning pipeline:

    1. OneHotEncoder turns text categories (e.g. 'male', 'female')
       into numbers the model can use.
    2. DecisionTreeRegressor is the actual ML model that learns
       how features relate to the final grade.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Decision tree for regression:
    # - max_depth controls how "deep" the tree can grow.
    #   Lower depth = simpler model, less chance of overfitting.
    tree = DecisionTreeRegressor(
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


def train_and_evaluate_model(X, y, feature_cols):
    """
    Train the decision tree and show how well it does overall.

    We:
      - split data into train and test sets
      - fit the model on the training data
      - test it on the test set
      - print a few common error metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    model = build_decision_tree_model(feature_cols)
    model.fit(X_train, y_train)

    # Make predictions on the test set (data the model hasn't seen before)
    y_pred_test = model.predict(X_test)

    # Calculate simple accuracy scores
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred_test)

    print("\n=== Global Model Performance (Decision Tree) ===")
    print(f"Test MAE   (average absolute error in marks): {mae:.2f}")
    print(f"Test RMSE  (root mean squared error):        {rmse:.2f}")
    print(f"Test R²    (variance explained):             {r2:.3f}")

    # Return the trained model so other parts of the program can use it
    return model


def compute_feature_weightings(model, feature_cols) -> pd.DataFrame:
    """
    Show how important each feature is to the decision tree.

    Decision trees don't have "weights" like linear models.
    Instead, they have feature importances that tell us:

        "How much did this feature help reduce prediction error in the tree?"

    Higher importance numbers mean the feature mattered more to the model.
    """
    # Get the internal bits of the pipeline
    preprocess: ColumnTransformer = model.named_steps["preprocess"]
    tree: DecisionTreeRegressor = model.named_steps["model"]

    # Get the generated one-hot feature names
    ohe: OneHotEncoder = preprocess.named_transformers_["cat"]
    encoded_feature_names = ohe.get_feature_names_out(feature_cols)

    importances = tree.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": encoded_feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    print("\n=== Learned Feature Weightings (Decision Tree Importances) ===")
    for _, row in importance_df.iterrows():
        print(f"{row['feature']:<45} {row['importance']:.4f}")

    return importance_df



def search_and_predict_loop(df: pd.DataFrame, model, feature_cols):
    """
    Simple text interface so we can "look up" a student.

    You type a student number (which is just the row number in the CSV),
    and the code shows:
      - the actual final grade (based on the three scores)
      - the grade predicted by the decision tree
    """
    print("\n==============================================")
    print(" Student Grade Prediction Interface (Decision Tree)")
    print(" Type a student number (line number) to view their predicted grade.")
    print(f" Valid range: 1 to {len(df)}")
    print(" Type 'quit' to exit.")
    print("==============================================")

    while True:
        query = input("\nEnter student number (or 'quit'): ").strip()
        if query.lower() == "quit":
            print("Exiting. Goodbye!")
            break

        if not query:
            print("Please type a number or 'quit'.")
            continue

        # Try to turn the input into an integer
        try:
            student_num = int(query)
        except ValueError:
            print("Please enter a whole number (integer) for the student.")
            continue

        # Check that the number is within the correct range
        if not (1 <= student_num <= len(df)):
            print(f"Student number out of range. Please enter a number between 1 and {len(df)}.")
            continue

        # Grab the row for this student_id
        row = df[df["student_id"] == student_num].iloc[0]

        actual_grade = row["final_grade"]
        # Build a one-row DataFrame with just the features the model expects
        X_student = row[feature_cols].to_frame().T
        predicted_grade = model.predict(X_student)[0]

        print("----------------------------------------------")
        print(f"Student number (line): {student_num}")
        print(f"Actual final grade:     {actual_grade:.2f}")
        print(f"Predicted final grade:  {predicted_grade:.2f}")


def main():
    """
    Main "story" of the program:

    1. Load the data from StudentsPerformance.csv.
    2. Create a final_grade column we want to predict.
    3. Train a Decision Tree regression model (machine learning).
    4. Show which features the model found most important (its "weighting").
    5. Let the user type a student number and see that student's predicted grade.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "StudentsPerformance.csv"

    # Step 1–2: Load and prepare the data
    df = load_data(csv_path)
    df = add_target_column(df, target_col="final_grade")
    X, y, feature_cols = prepare_features_and_target(df, target_col="final_grade")

    # Step 3: Train the ML model
    model = train_and_evaluate_model(X, y, feature_cols)

    # Step 4: Show how the model is "weighting" the input features
    _ = compute_feature_weightings(model, feature_cols)

    # Step 5: Interactive search by student number / line
    search_and_predict_loop(df, model, feature_cols)


if __name__ == "__main__":
    main()
