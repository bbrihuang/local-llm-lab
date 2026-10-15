"""smolagents (Hugging Face): its tool-calling agent."""
from smolagents import OpenAIServerModel, ToolCallingAgent, tool
from tasks import API_KEY, BASE_URL, INSTRUCTIONS, MAX_STEPS, MODEL, TOOLS

model = OpenAIServerModel(model_id=MODEL, api_base=BASE_URL, api_key=API_KEY)


def run(question):
    # A new agent per question, so no memory carries over between questions.
    agent = ToolCallingAgent(tools=[tool(f) for f in TOOLS], model=model, max_steps=MAX_STEPS,
                             instructions=INSTRUCTIONS, verbosity_level=0)
    return str(agent.run(question))
