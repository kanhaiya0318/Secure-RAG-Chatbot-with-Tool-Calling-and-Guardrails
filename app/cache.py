import re
import time
from collections import OrderedDict

from app.config import settings

_WS = re.compile(r"\s+")


def normalize_question(question: str) -> str:
    text = _WS.sub(" ", (question or "").strip().lower())
    return text.rstrip(" ?.!")    # right strip


class AnswerCache:
    """Exact-match LRU cache with expiry. In-memory, single process."""

    def __init__(self, max_entries: int, ttl_s: int):
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, question: str) -> str | None:
        norm = normalize_question(question)
        return f"{settings.GEMINI_MODEL}|{norm}" if norm else None

    def get(self, question: str) -> str | None:
        key = self._key(question)
        item = self._data.get(key) if key else None
        if item is None:
            self.misses += 1
            return None
        stored_at, answer = item
        if time.monotonic() - stored_at > self.ttl_s:
            del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return answer

    def set(self, question: str, answer: str) -> None:
        key = self._key(question)
        if not key or not answer:
            return
        self._data[key] = (time.monotonic(), answer)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


answer_cache = AnswerCache(settings.ANSWER_CACHE_MAX_ENTRIES, settings.ANSWER_CACHE_TTL_SECONDS)