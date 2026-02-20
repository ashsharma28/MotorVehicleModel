from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import logging
import logger_setup

log = logging.getLogger(__name__)

def get_knn_pipeline_model(df_policies ):
    
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

    log.debug("Preprocessing pipeline created successfully.")


    df_preprocessed_for_pipeline = df_policies.copy()
    reference_year = 2019

    # Calculate 'Age'
    df_preprocessed_for_pipeline['Age'] = reference_year - df_preprocessed_for_pipeline['Date_birth'].dt.year

    # Calculate 'Driving_Experience'
    df_preprocessed_for_pipeline['Driving_Experience'] = reference_year - df_preprocessed_for_pipeline['Date_driving_licence'].dt.year

    # Impute 'Length' with its median
    median_length = df_policies['Length'].median() # Ensure median_length is available from training context
    df_preprocessed_for_pipeline['Length'] = df_preprocessed_for_pipeline['Length'].fillna(median_length)

    log.debug("Created df_preprocessed_for_pipeline with engineered features and imputed 'Length'.")
    log.debug(df_preprocessed_for_pipeline[['Age', 'Driving_Experience', 'Length']].head().to_string())

    X_pipeline_input = df_preprocessed_for_pipeline[selected_numerical_features + selected_categorical_features]

    # 5. Fit and transform the data using the preprocessing_pipeline
    X_transformed = preprocessing_pipeline.fit_transform(X_pipeline_input)

    # 6. Retrieve the feature names after transformation
    transformed_feature_names = preprocessing_pipeline.named_steps['preprocessor'].get_feature_names_out()

    log.debug("Data transformed by preprocessing pipeline.")
    log.debug(f"Shape of transformed data: {X_transformed.shape}")
    log.debug(f"Number of transformed features: {len(transformed_feature_names)}")

    # 7. Initialize a NearestNeighbors model
    knn_pipeline_model = NearestNeighbors(n_neighbors=3, algorithm='brute')

    # 8. Fit the NearestNeighbors model to the X_transformed data
    knn_pipeline_model.fit(X_transformed)

    log.debug("NearestNeighbors model trained successfully with pipeline-transformed data.")
    return preprocessing_pipeline, knn_pipeline_model



def get_pickle_file_model():
    import pickle
    with open(r'knn_pipeline_model.pkl', 'rb') as file:
        loaded_knn_pipeline_model = pickle.load(file)
    log.debug("KNN pipeline model successfully loaded from 'knn_pipeline_model.pkl'.")
    return loaded_knn_pipeline_model

def get_pickle_file_preprocessor():
    import pickle
    with open(r'preprocessing_pipeline.pkl', 'rb') as file:
        loaded_preprocessing_pipeline = pickle.load(file)
    log.debug("Preprocessing pipeline successfully loaded from 'preprocessing_pipeline.pkl'.")
    return loaded_preprocessing_pipeline



def predict_premium_with_pipeline(df_policies, new_policy_data, preprocessing_pipeline, knn_pipeline_model):
    """
    Predicts the premium for a new policy using a preprocessing pipeline and a K-Nearest Neighbors model.

    Args:
        df_policies (pd.DataFrame): The original DataFrame of policies used for training.
        new_policy_data (pd.Series): Data for the new policy.
        preprocessing_pipeline (Pipeline): The fitted sklearn preprocessing pipeline.
        knn_pipeline_model (NearestNeighbors): The trained NearestNeighbors model.
        reference_year (int): Year used for 'Age' and 'Driving_Experience' calculation.
        median_length (float): Median 'Length' for imputation.

    Returns:
        float: Predicted premium for the new policy.
        pd.DataFrame: Top k similar policies used for prediction.
    """
    target_variable = "Premium"
    reference_year, median_length = 2019, df_policies['Length'].median() # Ensure these are consistent with training context
    selected_numerical_features = [
        'Seniority', 'Policies_in_force', 'Max_policies', 'Max_products',
        'Year_matriculation', 'Power', 'Cylinder_capacity', 'Value_vehicle',
        'Length', 'Weight', 'Age', 'Driving_Experience' , 'Cost_claims_year',
        'N_claims_year', 'N_claims_history', 'R_Claims_history'
    ]
    selected_categorical_features = [
        'Distribution_channel', 'Payment', 'Type_risk', 'Area', 'Second_driver', 'N_doors'
    ]

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

    # 6. Find Nearest Neighbors
    distances, indices = knn_pipeline_model.kneighbors(policy_transformed)
    log.debug(f"Distances to neighbors: {distances}")
    log.debug(f"Indices of neighbors: {indices}")

    # 7. Predict Premium by averaging the premiums of the similar policies
    # Use the original df_model_encoded for premium lookup, as it contains the Premium column.
    # Make sure to access the Premium column from df_model_encoded, not from the transformed data.
    similar_policies_premiums = df_policies.iloc[indices[0]][target_variable]
    predicted_premium = similar_policies_premiums.mean()

    top_k_similar_rows = df_policies.iloc[indices[0]]

    log.debug(f"\nPredicted Premium: {predicted_premium:.2f}")
    log.debug(f"Actual Premium for new policy (if available): {new_policy_data[target_variable]:.2f}")

    log.debug(policy_df_for_pipeline.to_string())
    log.debug(top_k_similar_rows.to_string())
    return {f"Predicted_premium" : predicted_premium ,
          "Top_3_Similar_Policies" : top_k_similar_rows}