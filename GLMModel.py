import pickle
import pandas as pd

# Load the GLM model from the pickle file
with open(r'glmm_model.pkl', 'rb') as file:
    loaded_glmm_model = pickle.load(file)

print("GLM model successfully loaded from 'glmm_model.pkl'.")
data_For_Shap_list = []
# --- Prepare a sample for prediction with all preprocessing steps ---

def get_prediction_from_GLM(i , df_policies):
  """
  Predicts the premium for a new policy using a GLM model.

  Args:
  i (int): Index of the policy to predict.

  Returns:
  None
  """

  # Simulate new policy data (e.g., taking the first row from original df_policies)
  # In a real scenario, 'new_policy_raw_data' would come from an external source
  new_policy_raw_data = df_policies.iloc[[i]].copy()

  # 1. Feature Engineering (Age, Driving_Experience)
  reference_year = 2019  # Define reference_year as used in training
  new_policy_raw_data['Age'] = reference_year - new_policy_raw_data['Date_birth'].dt.year
  new_policy_raw_data['Driving_Experience'] = reference_year - new_policy_raw_data['Date_driving_licence'].dt.year

  # 2. Missing Value Imputation (Length)
  median_length = df_policies['Length'].median() # Ensure median_length is available from training context
  new_policy_raw_data['Length'] = new_policy_raw_data['Length'].fillna(median_length)

  # 3. Select relevant numerical and categorical features
  # These lists should match the ones used during model training
  selected_numerical_features_for_pred = [
      'Seniority', 'Policies_in_force', 'Max_policies', 'Max_products',
      'Year_matriculation', 'Power', 'Cylinder_capacity', 'Value_vehicle',
      'Length', 'Weight', 'Age', 'Driving_Experience' , 'Cost_claims_year',
      'N_claims_year', 'N_claims_history', 'R_Claims_history'
  ]
  selected_categorical_features_for_pred = [
      'Distribution_channel', 'Payment', 'Type_risk', 'Area', 'Second_driver', 'N_doors'
  ]

  # Combine all features to be processed
  all_features_for_pred = selected_numerical_features_for_pred + selected_categorical_features_for_pred
  processed_policy_data = new_policy_raw_data[all_features_for_pred]

  # 4. One-Hot Encoding for categorical variables
  # Ensure consistency with the training data's one-hot encoded columns
  # We need the list of columns that the model was trained on (existing_selected_features)
  # The loaded_glmm_model.params.index can give us the feature names it expects.
  # We exclude the 'Intercept' from loaded_glmm_model.params.index.
  model_expected_features = loaded_glmm_model.params.index.drop('Intercept').tolist()

  new_policy_encoded = pd.get_dummies(
      processed_policy_data,
      columns=selected_categorical_features_for_pred,
      drop_first=True
  )

  # Reindex to match the columns of the training data and fill missing with 0
  # This handles cases where new_policy_data might not have all categories present in training data
  # or if 'drop_first=True' removed the only existing category
  final_policy_data_for_pred = new_policy_encoded.reindex(columns=model_expected_features, fill_value=0)

  # Convert boolean columns to int if any were created during reindexing
  for col in final_policy_data_for_pred.columns:
      if final_policy_data_for_pred[col].dtype == bool:
          final_policy_data_for_pred[col] = final_policy_data_for_pred[col].astype(int)

  # Predict the premium for the sample policy using the loaded model
  data_For_Shap_list.append(final_policy_data_for_pred)
  predicted_premium_glmm = loaded_glmm_model.predict(final_policy_data_for_pred)

  # The predict method for GLM with log link returns the predicted mean on the response scale
  # so no need to manually exp() here for statsmodels GLM.
  weights = loaded_glmm_model.params
  Feature_Importance = final_policy_data_for_pred * weights
  sorted_Feature_Importance = Feature_Importance.sort_values(by=Feature_Importance.index[0], axis=1, ascending=False)
  Top_3_Important_Factors = sorted_Feature_Importance.loc[: , list(sorted_Feature_Importance.columns[:3])]
  # Add the intercept (const) to see the full picture
  print(Top_3_Important_Factors)

  print(f"Predicted premium : {predicted_premium_glmm.iloc[0]:.2f}"  , f"Actual premium : {new_policy_raw_data['Premium'].values[0]:.2f}")
  return {f"Predicted premium" : predicted_premium_glmm.iloc[0] ,
          "Top_3_Important_Factors" : Top_3_Important_Factors}



if __name__ == "__main__":
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

    for i in range(2):
        get_prediction_from_GLM(i, df_policies)