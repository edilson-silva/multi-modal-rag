SYSTEM_PROMPT = (
    'Answer the user question exclusively based on the context excerpts '
    'provided below. If the answer cannot be found in the context, '
    'say so clearly — do not fabricate information.'
)

USER_PROMPT_TEMPLATE = """\
Context:
{context}

Question: {query}"""
