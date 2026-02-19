from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors


# read two data: policy and claims from /content/drive/MyDrive/Colab Notebooks/Allianz Data/Motor vehicle insurance data.csv and  /content/drive/MyDrive/Colab Notebooks/Allianz Data/sample type claim.csv
import pandas as pd
# Increase pandas option max_for columns
pd.set_option('display.max_columns', None)
df_policies = pd.read_csv('data/Motor vehicle insurance data.csv', sep=";")
df_claims  = pd.read_csv('data/sample type claim.csv' , sep = ';')
# Convert dates to datetime objects
date_cols = ['Date_start_contract', 'Date_last_renewal', 'Date_next_renewal', 'Date_birth', 'Date_driving_licence']
for col in date_cols:
    df_policies[col] = pd.to_datetime(df_policies[col], format='%d/%m/%Y', errors='coerce')


selected_numerical_features_for_pred = [
      'Seniority', 'Policies_in_force', 'Max_policies', 'Max_products',
      'Year_matriculation', 'Power', 'Cylinder_capacity', 'Value_vehicle',
      'Length', 'Weight', 'Age', 'Driving_Experience' , 'Cost_claims_year',
      'N_claims_year', 'N_claims_history', 'R_Claims_history'
  ]
selected_categorical_features_for_pred = [
      'Distribution_channel', 'Payment', 'Type_risk', 'Area', 'Second_driver', 'N_doors'
  ]

selected_numerical_features = selected_numerical_features_for_pred
selected_categorical_features = selected_categorical_features_for_pred
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selected_numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), selected_categorical_features)
    ],
    remainder='passthrough'
)

preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

print("Preprocessing pipeline created successfully.")


df_preprocessed_for_pipeline = df_policies.copy()
reference_year = 2019  # Define reference_year as used in training

# Calculate 'Age'
df_preprocessed_for_pipeline['Age'] = reference_year - df_preprocessed_for_pipeline['Date_birth'].dt.year

# Calculate 'Driving_Experience'
df_preprocessed_for_pipeline['Driving_Experience'] = reference_year - df_preprocessed_for_pipeline['Date_driving_licence'].dt.year

# Impute 'Length' with its median
median_length = df_policies['Length'].median() # Ensure median_length is available from training context
df_preprocessed_for_pipeline['Length'] = df_preprocessed_for_pipeline['Length'].fillna(median_length)

print("Created df_preprocessed_for_pipeline with engineered features and imputed 'Length'.")
print(df_preprocessed_for_pipeline[['Age', 'Driving_Experience', 'Length']].head())


def predict_premium_with_pipeline(new_policy_data, preprocessing_pipeline, knn_pipeline_model,
                                  selected_numerical_features, selected_categorical_features,
                                  transformed_feature_names, reference_year, median_length, target_variable):
    """
    Predicts the premium for a new policy using a preprocessing pipeline and a K-Nearest Neighbors model.

    Args:
        new_policy_data (pd.Series): Data for the new policy.
        preprocessing_pipeline (Pipeline): The fitted sklearn preprocessing pipeline.
        knn_pipeline_model (NearestNeighbors): The trained NearestNeighbors model.
        selected_numerical_features (list): List of numerical feature names.
        selected_categorical_features (list): List of categorical feature names.
        transformed_feature_names (list): Column names of the data after pipeline transformation.
        reference_year (int): Year used for 'Age' and 'Driving_Experience' calculation.
        median_length (float): Median 'Length' for imputation.
        target_variable (str): The name of the target variable (e.g., 'Premium').

    Returns:
        float: Predicted premium for the new policy.
        pd.DataFrame: Top k similar policies used for prediction.
    """
    # 1. Create a copy to avoid modifying the original Series
    policy_to_predict = new_policy_data.copy()

    # 2. Apply feature engineering steps
    policy_to_predict['Age'] = reference_year - policy_to_predict['Date_birth'].year
    policy_to_predict['Driving_Experience'] = reference_year - policy_to_predict['Date_driving_licence'].year

    # 3. Impute 'Length' if it's missing
    if pd.isna(policy_to_predict['Length']):
        policy_to_predict['Length'] = median_length

    # 4. Convert the preprocessed Series into a DataFrame with a single row
    # and select only the relevant features for the pipeline
    policy_df_for_pipeline = pd.DataFrame([policy_to_predict[selected_numerical_features + selected_categorical_features]])

    # 5. Use the preprocessing_pipeline.transform() method
    policy_transformed = preprocessing_pipeline.transform(policy_df_for_pipeline)

    # Convert the transformed numpy array back to DataFrame for consistency with previous debugging steps
    # and to potentially view it if needed, though knn_pipeline_model expects a numpy array.
    policy_transformed_df = pd.DataFrame(policy_transformed, columns=transformed_feature_names)

    # 6. Find Nearest Neighbors
    distances, indices = knn_pipeline_model.kneighbors(policy_transformed)
    print(f"Distances to neighbors: {distances}")
    print(f"Indices of neighbors: {indices}")

    # 7. Predict Premium by averaging the premiums of the similar policies
    # Use the original df_model_encoded for premium lookup, as it contains the Premium column.
    # Make sure to access the Premium column from df_model_encoded, not from the transformed data.
    similar_policies_premiums = df_policies.iloc[indices[0]][target_variable]
    predicted_premium = similar_policies_premiums.mean()

    top_k_similar_rows = df_policies.iloc[indices[0]]

    print(f"\nPredicted Premium: {predicted_premium:.2f}")
    print(f"Actual Premium for new policy (if available): {new_policy_data[target_variable]:.2f}")

    print(policy_df_for_pipeline)
    print(top_k_similar_rows)

    return predicted_premium, top_k_similar_rows

print("Simplified premium prediction function `predict_premium_with_pipeline` defined.")


