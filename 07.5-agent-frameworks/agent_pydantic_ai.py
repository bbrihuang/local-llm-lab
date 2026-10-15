"""PydanticAI: its agent with an OpenAI-compatible model."""
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
from tasks import API_KEY, BASE_URL, INSTRUCTIONS, MAX_STEPS, MODEL, TOOLS

agent = Agent(OpenAIChatModel(MODEL, provider=OpenAIProvider(base_url=BASE_URL, api_key=API_KEY)),
              instructions=INSTRUCTIONS, tools=TOOLS)


def run(question):
    return agent.run_sync(question, usage_limits=UsageLimits(request_limit=MAX_STEPS)).output
