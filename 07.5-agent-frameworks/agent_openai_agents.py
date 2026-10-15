"""OpenAI Agents SDK, pointed at the local server instead of OpenAI."""
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from openai import AsyncOpenAI
from tasks import API_KEY, BASE_URL, INSTRUCTIONS, MAX_STEPS, MODEL, TOOLS

set_tracing_disabled(True)  # otherwise it tries to upload traces to OpenAI
agent = Agent(name="assistant", instructions=INSTRUCTIONS,
              tools=[function_tool(f) for f in TOOLS],
              model=OpenAIChatCompletionsModel(model=MODEL,
                                               openai_client=AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)))


def run(question):
    return Runner.run_sync(agent, question, max_turns=MAX_STEPS).final_output
