SYSTEM_PROMPT = """
日本語を話すAIアシスタントとして応答してください。
"""

PROMPT = [
    {
        'role': 'system',
        'content': SYSTEM_PROMPT,
    },
    {"role": "user"},
]