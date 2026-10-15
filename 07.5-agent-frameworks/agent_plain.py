"""No framework: the agent loop written by hand with the openai client."""
import json
from openai import OpenAI
from tasks import API_KEY, BASE_URL, INSTRUCTIONS, MAX_STEPS, MODEL, calculator, list_files, read_file

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
FUNCTIONS = {f.__name__: f for f in (calculator, list_files, read_file)}
SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "required": ["expression"],
                       "properties": {"expression": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "list_files", "description": "List the files in the data folder.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read one file from the data folder.",
        "parameters": {"type": "object", "required": ["name"],
                       "properties": {"name": {"type": "string"}}}}},
]


def run(question):
    messages = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": question}]
    for _ in range(MAX_STEPS):
        msg = client.chat.completions.create(model=MODEL, messages=messages, tools=SCHEMAS).choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = FUNCTIONS[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    raise RuntimeError("no answer within the step limit")
