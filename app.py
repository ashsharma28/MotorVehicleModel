# # Call the Agent code
# import logging
# import logger_setup
# from Agent import Ally , app 

# log = logging.getLogger(__name__)

# config = {"configurable": {"thread_id": "1"}}

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         break
    
#     response = app.invoke(
#         input={"input": user_input},
#         config=config
#     )
#     print(f"Agent: {response['messages'][-1].content}")
