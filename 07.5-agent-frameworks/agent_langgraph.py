"""LangGraph: its ready-made ReAct agent with a LangChain chat model."""
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tasks import API_KEY, BASE_URL, INSTRUCTIONS, MAX_STEPS, MODEL, TOOLS

agent = create_react_agent(ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY),
                           tools=TOOLS, prompt=INSTRUCTIONS)


def run(question):
    result = agent.invoke({"messages": [("user", question)]},
                          {"recursion_limit": 2 * MAX_STEPS + 1})
    return result["messages"][-1].content
