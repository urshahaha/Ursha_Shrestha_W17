import json
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .schemas import AssistantAnswer
from .tools import TOOL_DEFINITIONS, calculator, list_documents


SYSTEM_PROMPT = """You are a careful RAG assistant.
Use retrieved context for knowledge-base questions.
If context is insufficient, say so clearly.
Never invent a source.
You may call calculator for arithmetic and list_documents for available files.
Return the final answer as valid JSON matching the requested schema.
"""

ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "chunk_id": {"type": "integer"},
                        },
                        "required": ["filename", "chunk_id"],
                        "additionalProperties": False,
                    },
                },
                "used_tool": {"type": ["string", "null"]},
                "grounded": {"type": "boolean"},
            },
            "required": ["answer", "sources", "used_tool", "grounded"],
            "additionalProperties": False,
        },
    },
}


def _client_and_model():
    provider = settings.llm_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing."
            )

        return (
            OpenAI(
                api_key=settings.openai_api_key
            ),
            settings.openai_model,
            True,
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing."
            )

        return (
            OpenAI(
                api_key=settings.gemini_api_key,
                base_url=(
                    "https://generativelanguage.googleapis.com/"
                    "v1beta/openai/"
                ),
            ),
            settings.gemini_model,
            True,
        )

    if provider == "vllm":
        return (
            OpenAI(
                api_key="EMPTY",
                base_url=settings.vllm_base_url,
            ),
            settings.vllm_model,
            False,
        )

    raise ValueError(
        "LLM_PROVIDER must be "
        "'openai', 'gemini', or 'vllm'."
    )

def _context(rows: list[dict]) -> str:
    if not rows:
        return "(No relevant document context was retrieved.)"
    return "\n\n".join(
        f"[Source: {row['filename']} | chunk {row['chunk_id']}]\n{row['text']}"
        for row in rows
    )


def _tool(name: str, args: dict):
    if name == "calculator":
        return calculator(args["expression"])
    if name == "list_documents":
        return list_documents(settings.upload_dir)
    raise ValueError(f"Unknown tool: {name}")


def _json(text: str | None) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("The model did not return JSON.")
        return json.loads(match.group(0))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=6,
    ),
    reraise=True,
)
def answer_question(
    question: str,
    retrieved: list[dict],
) -> AssistantAnswer:

    client, model, _ = _client_and_model()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Retrieved context:\n{_context(retrieved)}\n\n"
                f"User question:\n{question}"
            ),
        },
    ]

    common = {
        "model": model,
        "messages": messages,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
    }

    provider = settings.llm_provider.lower()

    # --------------------------------------------------------
    # Only make tools available when the question actually
    # needs them.
    # --------------------------------------------------------

    q = question.lower()

    wants_calculator = (
        "calculate" in q
        or "compute" in q
        or bool(
            re.search(
                r"\d+\s*[\+\-\*/]\s*\d+",
                q,
            )
        )
    )

    wants_document_list = any(
        phrase in q
        for phrase in [
            "list documents",
            "what documents",
            "which documents",
            "list files",
            "what files",
            "which files",
        ]
    )

    selected_tools = []

    if wants_calculator:
        selected_tools = [
            TOOL_DEFINITIONS[0]
        ]

    elif wants_document_list:
        selected_tools = [
            TOOL_DEFINITIONS[1]
        ]


    # --------------------------------------------------------
    # OPENAI / GEMINI
    # --------------------------------------------------------

    if provider in {
        "openai",
        "gemini",
    }:

        request_args = {
            **common,
            "response_format": ANSWER_SCHEMA,
        }

        # Only expose tools when necessary.
        if selected_tools:
            request_args["tools"] = (
                selected_tools
            )

            request_args["tool_choice"] = (
                "auto"
            )


        first = (
            client.chat.completions.create(
                **request_args
            )
        )

        message = (
            first.choices[0].message
        )


        # ----------------------------------------------------
        # A tool was actually called.
        # ----------------------------------------------------

        if message.tool_calls:

            messages.append(
                {
                    "role": "assistant",
                    "content":
                        message.content or "",

                    "tool_calls": [
                        call.model_dump()
                        for call
                        in message.tool_calls
                    ],
                }
            )

            used_tools = []

            for tool_call in (
                message.tool_calls
            ):

                args = json.loads(
                    tool_call.function.arguments
                    or "{}"
                )

                result = _tool(
                    tool_call.function.name,
                    args,
                )

                used_tools.append(
                    tool_call.function.name
                )

                messages.append(
                    {
                        "role": "tool",

                        "tool_call_id":
                            tool_call.id,

                        "content":
                            json.dumps(result),
                    }
                )


            final = (
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    response_format=ANSWER_SCHEMA,
                )
            )

            payload = _json(
                final.choices[0]
                .message.content
            )

            payload["used_tool"] = (
                used_tools[0]
                if used_tools
                else None
            )

            return (
                AssistantAnswer
                .model_validate(payload)
            )


        # ----------------------------------------------------
        # Normal RAG question — no tool.
        # ----------------------------------------------------

        payload = _json(
            message.content
        )

        payload["used_tool"] = None

        return (
            AssistantAnswer
            .model_validate(payload)
        )


    # --------------------------------------------------------
    # LOCAL vLLM
    # --------------------------------------------------------

    try:

        response = (
            client.chat.completions.create(
                **common,
                response_format={
                    "type": "json_object"
                },
            )
        )

    except Exception:

        response = (
            client.chat.completions.create(
                **common
            )
        )


    payload = _json(
        response.choices[0]
        .message.content
    )

    payload["used_tool"] = None

    return (
        AssistantAnswer
        .model_validate(payload)
    )