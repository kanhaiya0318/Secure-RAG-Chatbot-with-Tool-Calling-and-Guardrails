import logging
from langsmith import traceable

from app.cache import answer_cache
from app.services.llm_service import LLMService
from app.services.memory_service import compress
from app.config import settings
from app.utils.timing import timer
from app.guardrails import (
    OutputFilter, LEAK_MESSAGE, check_chunk, collect_sources, sources_footer,
)
from app.services.tools import SearchDocs, WebSearch   # add WebSearch

logger = logging.getLogger(__name__)

async def run_tool_calls(parts: list, rag, client) -> list:
    response_parts = []
    calls_run = 0
    for part in parts:
        call = part.get("functionCall")
        if not call:
            continue
        name = call.get("name", "")
        args = call.get("args", {})
        calls_run += 1

        if calls_run > settings.MAX_TOOL_CALLS_PER_TURN:
            result = {"error": "Too many tool calls in one turn. Answer with what you already have."}
        elif name == "SearchDocs":
            with timer("SearchDocs : "):
                result = await SearchDocs(args.get("query", ""), rag)
        elif name == "WebSearch":
            with timer("WebSearch : "):
                result = await WebSearch(
                    args.get("query", ""),
                    client,
                    time_range=args.get("time_range"),
                )
        else:
            result = {"error": "Unknown function name"}

        fr = {"name": name, "response": result}
        if call.get("id"):
            fr["id"] = call["id"]
        response_parts.append({"functionResponse": fr})
    return response_parts


@traceable(
        name="chat_turn",
        run_type="chain",
        process_inputs=lambda i:{"prompt":i.get("prompt")},  # don't log the whole session or objects
        reduce_fn=lambda chunks: "".join(chunks),            # join streamed text into one output
)
async def reply(db: dict, prompt: str, llm_service: LLMService, rag):
    first_turn = not db["history"] and not db["summary"]

    # cache hit: no Gemini call, no tools
    if settings.ANSWER_CACHE_ENABLED and first_turn:
        cached = answer_cache.get(prompt)
        if cached is not None:
            logger.info(f"Answer cache hit (hits={answer_cache.hits}, misses={answer_cache.misses})")
            db["history"].append({"role": "user", "parts": [{"text": prompt}]})
            db["history"].append({"role": "model", "parts": [{"text": cached}]})
            yield cached
            return
        logger.info(f"Answer cache miss (hits={answer_cache.hits}, misses={answer_cache.misses})")
    elif settings.ANSWER_CACHE_ENABLED:
        logger.info("Answer cache skipped: not the first message of this session")

    db["history"].append({"role": "user", "parts": [{"text": prompt}]})

    with timer("Compress : "):
        await compress(db, llm_service)

    turn_start = len(db["history"]) - 1
    sources = []
    tools_used = set()
    cacheable = True

    for i in range(settings.RETRY_LLM_CALL):
        out_filter = OutputFilter()
        shown, other_parts = [], []
        try:
            async for chunk in llm_service.generate_content_stream(
                db["history"], db["summary"]
            ):
                blocked = check_chunk(chunk)
                if blocked:
                    reason, message = blocked
                    logger.warning(f"Gemini blocked the turn: {reason}")
                    del db["history"][turn_start:]
                    yield message
                    return

                if "candidates" not in chunk:
                    continue
                parts = chunk["candidates"][0].get("content", {}).get("parts", [])

                for part in parts:
                    if "text" in part:
                        safe = out_filter.feed(part["text"])
                        if safe:
                            shown.append(safe)
                            yield safe
                    else:
                        other_parts.append(part)
                if out_filter.leaked:
                    break
        except Exception as e:
            logger.error(f"Gemini stream failed: {e}")
            yield "\n[error: could not get a response from the model]"
            return

        tail = out_filter.flush()
        if out_filter.leaked:
            logger.warning("System prompt marker found in model output; reply stopped")
            del db["history"][turn_start:]
            yield LEAK_MESSAGE
            return
        if tail:
            shown.append(tail)
            yield tail

        text = "".join(shown)
        parts = ([{"text": text}] if text else []) + other_parts
        db["history"].append({"role": "model", "parts": parts})

        if not any("functionCall" in p for p in parts):
            footer = sources_footer(sources, text)
            if footer:
                yield footer

            store = (settings.ANSWER_CACHE_ENABLED and first_turn and cacheable
                     and tools_used == {"SearchDocs"} and bool(text))
            logger.info(
                f"Answer cache store={store} (first_turn={first_turn}, "
                f"cacheable={cacheable}, tools_used={sorted(tools_used)})"
            )
            if store:
                answer_cache.set(prompt, text + footer)
            return

        response_parts = await run_tool_calls(parts, rag, llm_service.client)
        for p in response_parts:
            fr = p["functionResponse"]
            tools_used.add(fr["name"])
            if "error" in (fr.get("response") or {}):
                cacheable = False
        sources += collect_sources(response_parts)
        db["history"].append({"role": "user", "parts": response_parts})

    yield "\n[error: the model used too many tool turns without a final answer]"