from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool, create_swarm
import os
# Set your Hugging Face API token as an environment variable before running the code
os.environ['HF_TOKEN'] = '<ADD_YOUR_HF_TOKEN_HERE>'
import pandas as pd

# Increase pandas option max_for columns
pd.set_option('display.max_columns', None)
df_policies = pd.read_csv('data/Motor vehicle insurance data.csv', sep=";")
df_claims  = pd.read_csv('data/Claims.csv' , sep = ';')
# Convert dates to datetime objects
date_cols = ['Date_start_contract', 'Date_last_renewal', 'Date_next_renewal', 'Date_birth', 'Date_driving_licence']
for col in date_cols:
    df_policies[col] = pd.to_datetime(df_policies[col], format='%d/%m/%Y', errors='coerce')


model = ChatOpenAI(model="meta-llama/Llama-3.1-8B-Instruct" ,
                   base_url="https://router.huggingface.co/v1",
                   api_key=os.environ["HF_TOKEN"])


from GLMModel import *
from KNN import *
def get_Policy_Data(ID):
    """Retrieves the policy history for the given policy ID.
    Args:    ID (int): ID of the policy to retrieve.
    Returns: str: A DataFrame containing the policy history."""
    # get the index of the policy in df_policies based on the given ID
    policy_data = df_policies.loc[df_policies['ID'] == ID]
    if policy_data.empty:
        return f"No policy found with ID {ID}."
    # Make Premium of the last policy history row as NaN to avoid data leakage in prediction
    policy_data.loc[policy_data.index[-1], 'Premium'] = pd.NA

    return policy_data.to_string()

knn_pipeline_model = get_pickle_file_model()
preprocessing_pipeline = get_pickle_file_preprocessor()

def get_prediction_from_KNN(ID):
    """Predicts the premium for a new policy along with the top 3 similar policies using a KNN model.
    Args:    ID (int): ID of the policy to predict.
    Returns: Dict: A dictionary containing the predicted premium and the top 3 similar policies."""
    # get the index of the policy in df_policies based on the given ID
    if df_policies.loc[df_policies['ID'] == ID].empty:
        return f"No policy found with ID {ID}."
    i = df_policies.index[df_policies['ID'] == ID].tolist()[-1]
    new_policy_data = df_policies.iloc[i]  # Example: using the first policy for prediction
    predicted_premium_and_similar_policies = predict_premium_with_pipeline(df_policies, new_policy_data, preprocessing_pipeline, knn_pipeline_model)
    return predicted_premium_and_similar_policies

def get_prediction_from_GLM(ID):
    """Predicts the premium for a new policy along with the top 3 contributing factors using a GLM model.
    Args:    ID (int): ID of the policy to predict.
    Returns: Dict: A dictionary containing the predicted premium and the top 3 contributing factors."""
    # get the index of the policy in df_policies based on the given ID
    if df_policies.loc[df_policies['ID'] == ID].empty:
        return f"No policy found with ID {ID}."
    i = df_policies.index[df_policies['ID'] == ID].tolist()[-1]  
    predicted_premium_glmm_and_Top_3_Contributing_Factors = get_prediction_from_GLM_withData(i, df_policies)
    return predicted_premium_glmm_and_Top_3_Contributing_Factors

# 1. Grounding with Retrieval-Augmented Generation (RAG) 
def get_Models_Prediction(ID):
    """Predicts the premium for a new policy using both GLM and KNN models.
    Args:    ID (int): ID of the policy to predict.
    Returns: Dict: A dictionary containing the predicted premiums and insights from both models."""
    glm_prediction = get_prediction_from_GLM(ID)
    knn_prediction = get_prediction_from_KNN(ID)

    return f"""
    GLM Prediction: {glm_prediction}
    KNN Prediction: {knn_prediction}
    """
Ally = create_agent(
    model,
    name="Ally",
    tools=[
        get_Policy_Data,
        get_Models_Prediction],
        # 2. Prompt Engineering and Guardrails
    system_prompt="""
You are Ally, an helpful assistant to the Actuarial professionals for Nexus Insurance.
Nexus Insurance, a subsidiary of the Nexus Group, is a multinational non-life insurer with a large motor vehicle insurance portfolio. The company records detailed transaction-level data for each policy, including renewals, premium changes, vehicle characteristics, and claims information.
The Actuarial professionals are working on pricing a new motor vehicle insurance policy, your task is to help them with structured, explainable predictions and reasonings which are suitable for rapid review by analysts.

***Task Instructions:***
First, Ask User to give a policy ID, DO NOT ASSUME on your own.

Once User provides the ID, you should fist fetch the policy data using the get_Policy_Data tool, then use both the GLM and KNN models to predict the premium for that policy. Think this answer through step-by-step before writing a response, finally You should provide a structured response in markdown format, that includes:
1. *The predicted premium* from the GLM model along with the top 3 contributing factors.
2. *Similar Policies*: The top 3 similar policies and their premiums.
3. *Reasoning*: A brief explanation of the predictions, highlighting what the policy data point values are for given ID, and why they have a impact on the results.
4. Conclusion on the expected premium for the policy based on the insights from both models.

***Important Notes:***
You have access to two predictive models: a Generalized Linear Model (GLM) and a K-Nearest Neighbors (KNN) model. You will always call the tool get_Models_Prediction with the policy ID to get the predictions from both models.
The GLM model provides a predicted premium for a given policy and identifies the top 3 contributing factors to that prediction. The KNN model identifies the top 3 most similar policies in the historical data and provides their premiums.

Statistically, below are found to be the most impacting factors on the premium of a policy:
Column                    Importance     Meaning
Type_risk_2               65.034536     Type of risk associated with the policy – Vans (Type_risk = 2)
Type_risk_3               63.427346     Type of risk associated with the policy – Passenger cars (Type_risk = 3)
Payment_1                 61.927136     Last payment method – Half-yearly administrative process (Payment = 1)
Value_vehicle             60.740985     Market value of the vehicle on 31/12/2019
Year_matriculation        59.989753     Year of registration of the vehicle (YYYY)
Intercept                 46.159973     Model intercept (baseline constant term)
Second_driver_1           44.786042     Multiple regular drivers declared (Second_driver = 1)
Type_risk_4               28.700191     Type of risk associated with the policy – Agricultural vehicles (Type_risk = 4)
R_Claims_history          27.497455     Ratio of number of claims to total policy duration (claims frequency history)
N_claims_history          21.700103     Total number of claims filed throughout the entire duration of the policy
Area_1                    17.429705     Urban area (more than 30,000 inhabitants) – Area = 1
Distribution_channel_1    14.433484     Contracted through Insurance brokers (Distribution_channel = 1)
Max_products              12.828811     Maximum number of products simultaneously held at any time
Cylinder_capacity         12.599687     Cylinder capacity of the vehicle
Policies_in_force         12.029689     Total number of policies currently held by the insured
Driving_Experience        10.817501     Years of driving experience (derived from Date_driving_licence)
Power                     10.368187     Vehicle power measured in horsepower
N_claims_year              9.925667     Total number of claims incurred during the current year
Length                     9.781918     Length of the vehicle in meters
Max_policies               9.436313     Maximum number of policies ever held in force
Weight                     6.696493     Weight of the vehicle in kilograms
Cost_claims_year           6.642014     Total cost of claims during the current year
N_doors_4                  6.372515     Vehicle has 4 doors (N_doors = 4)
Seniority                  6.029820     Total number of years associated with the insurance entity
Age                        5.523550     Age of the insured (derived from Date_birth)
N_doors_3                  1.082978     Vehicle has 3 doors (N_doors = 3)
N_doors_5                  0.551247     Vehicle has 5 doors (N_doors = 5)
    """    
)

checkpointer = InMemorySaver()
workflow = create_swarm(
    [Ally],
    default_active_agent="Ally"
)
app = workflow.compile(checkpointer=checkpointer)
