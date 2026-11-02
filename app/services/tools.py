import httpx
from typing import Literal
from langsmith import traceable

from app.config import settings
from app.utils.timing import logger
from app.guardrails import wrap_untrusted, UNTRUSTED_NOTICE


def _drop_object(inputs:dict)->dict:
    return {k:v for k,v in inputs.items() if k not in ("rag", "client")}

SEARCH_WEB_DECLARATION = {
    "name": "WebSearch",
    "description": (
        "Search the web for information requiring external verification, "
        "including current news, prices, recent events, and unfamiliar facts. "
        "Use time_range only when the user asks for recent information. "
        "Omit it for timeless or general knowledge questions."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": (
                    "A specific search query. Preserve important details "
                    "such as location, date, product, and subject."
                )
            },
            "time_range": {
                "type": "STRING",
                "enum": ["day", "week", "month", "year"],
                "description": (
                    "Relative time filter. Use day for today or the latest "
                    "daily updates, week for this week, month for this month, "
                    "and year for this year. Omit for timeless questions."
                )
            },
        },
        "required": ["query"],
    },
}

@traceable(name="WebSearch", run_type="tool", process_inputs=_drop_object)
async def WebSearch(
        query: str, 
        client: httpx.AsyncClient, 
        time_range:Literal["year", "month", "week", "day"] | None = None, 
    ):
    try:
        url = "https://api.tavily.com/search"
        query = str(query or "").strip()[: settings.MAX_QUERY_CHARS]
        if not query:
            return {"error": "Empty search query."}

        payload={"query": query,
                "max_results" : settings.SEARCH_MAX_RESULTS,
                "time_range": time_range,
                "include_images": False,
                "topic": "general",
                "country": "india",
            "search_depth": settings.SEARCH_DEPTH,
            }
        logger.info(f"response of Websearch: {payload}")
        header={"Authorization": f"Bearer {settings.TAVILY_API_KEY}"}

        response = await client.post(url, json = payload, headers = header)
        response.raise_for_status()

        response_json = response.json()
        results = []
        for i in response_json["results"][:2]:
            result = {"url": i.get("url", ""),
                    "title": i.get("title",""),
                    "content":wrap_untrusted(i.get("content", "")[:500]),
                    "published_date": i.get("published_date", "")
            }
            results.append(result)
        return {"notice": UNTRUSTED_NOTICE, "results": results}
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Tavily status: {e.response.status_code}"
        )
        logger.error(
            f"Tavily response: {e.response.text}"
        )
        return {"error": "Tavily search request failed."}

    except Exception:
        logger.exception("Web Search Failed")
        return {"error": "Could not search the web right now."}


SEARCH_DOCS_DECLARATION = {
    "name": "SearchDocs",
    "description": (
        "Search the user's uploaded documents (PDFs). Use this for any question "
        "that may be answered by the content of those documents."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "A short, specific search query describing what to find."
            }
        },
        "required": ["query"],
    },
}

@traceable(name="SearchDocs", run_type="tool", process_inputs=_drop_object)
async def SearchDocs(query:str, rag, k:int = 2):
    query = str(query or "").strip()[: settings.MAX_QUERY_CHARS]
    if not query:
        return {"error": "Empty search query."}
    try:
        docs = await rag.aretrieve(query, k=k)
    except Exception as e:
        logger.error(f"SearchDocs failed: {e}")
        return {"error": "Could not search the documents right now."}

    if not docs:
        return {"results": [], "note": "No relevant passages found in the uploaded documents."}

    return {
        "notice": UNTRUSTED_NOTICE,
        "results": [
            {
                "source": d.metadata.get("source", ""),
                "page": d.metadata.get("page_number"),
                "score": round(float(d.metadata.get("score", 0)), 4),
                "text": wrap_untrusted(d.page_content),
            }
            for d in docs
        ]
    }