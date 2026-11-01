import httpx
from typing import Literal
from langsmith import traceable

from app.config import settings
from app.utils.timing import logger
from app.guardrails import wrap_untrusted, UNTRUSTED_NOTICE


def _drop_object(inputs:dict)->dict:
    return {k:v for k,v in inputs.items() if k not in ("rag", "client")}

SEARCH_WEB_DECLARATION = {
    "name" : "WebSearch",
    "description": "Search the web for current or recent information: news, sports results,prices, or facts you are unsure about.",
    "parameters" : {
        "type" : "OBJECT",
        "properties" : {
            "query":{
                "type" : "STRING",
                "description" : "A sort specific search queary"
            },
            "time_range": {
                "type" : "STRING",
                "enum":["day", "month", "week","year"],
                "description": "Only return results published or updated within this window back from today. Use 'day' for today's news, 'week' for this week.Omit for timeless questions."
            },
            "filter_by_published_date":{
                "type":"BOOLEAN",
                "description":"If true, drop results without a detectable date or outside "
                                "time_range. Use only together with time_range, when freshness matters."
            }},
        "required": ["query"]
    },
}

@traceable(name="WebSearch", run_type="tool", process_inputs=_drop_object)
async def WebSearch(
        query: str, 
        client: httpx.AsyncClient, 
        time_range:Literal["year", "month", "week", "day"] | None = None, 
        filter_by_published_date:bool = False
    ):
    try:
        url = "https://api.tavily.com/search"
        payload={"query": query,
                "max_results" : settings.SEARCH_MAX_RESULTS,
                "time_range": time_range,
                "include_images": False,
                "filter_by_published_date": filter_by_published_date, 
            "search_depth": settings.SEARCH_DEPTH
            }
        logger.info(f"response of Websearch: {payload}")
        header={"Authorization": f"Bearer {settings.TAVILY_API_KEY}"}

        response = await client.post(url, json = payload, headers = header)
        response.raise_for_status()

        response_json = response.json()
        results = []
        for i in response_json["results"][:3]:
            result = {"url": i.get("url", ""),
                    "title": i.get("title",""),
                    "content":wrap_untrusted(i.get("content", "")[:300]),
                    "published_date": i.get("published_date", "")
            }
            results.append(result)
        return {"notice": UNTRUSTED_NOTICE, "results": results}
    except Exception as e:
        logger.error(f"Web Search Faild {e}")
        return {"error":"Could not search the web right now."}


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