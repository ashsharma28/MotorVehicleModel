from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool, create_swarm
import os
os.environ['HF_TOKEN'] = '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'

model = ChatOpenAI(model="meta-llama/Llama-3.1-8B-Instruct" ,
                   base_url="https://router.huggingface.co/v1",
                   api_key=os.environ["HF_TOKEN"],
)

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

alice = create_agent(
    model,
    tools=[
        add,
        create_handoff_tool(
            agent_name="Bob",
            description="Transfer to Bob",
        ),
    ],
    system_prompt="You are Alice, an addition expert.",
    name="Alice",
)

bob = create_agent(
    model,
    tools=[
        create_handoff_tool(
            agent_name="Alice",
            description="Transfer to Alice, she can help with math",
        ),
    ],
    system_prompt="You are Bob, you speak like a pirate.",
    name="Bob",
)

checkpointer = InMemorySaver()
workflow = create_swarm(
    [alice, bob],
    default_active_agent="Alice"
)
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
turn_1 = app.invoke(
    {"messages": [{"role": "user", "content": "i'd like to speak to Bob"}]},
    config,
)
print(turn_1)
turn_2 = app.invoke(
    {"messages": [{"role": "user", "content": "what is (52010 * 11) + 7?"}]},
    config,
)
print(turn_2)