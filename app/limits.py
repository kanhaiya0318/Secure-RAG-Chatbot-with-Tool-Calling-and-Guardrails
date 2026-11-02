import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.config import settings


class RateLimiter:
    """Sliding-window limiter. In-memory, so it works for a single process only."""

    def __init__(self, max_calls: int, window_s: int):
        self.max_calls = max_calls
        self.window_s = window_s
        self._hits: dict[str, deque] = defaultdict(deque)

    def hit(self, key: str) -> None:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        if len(q) >= self.max_calls:
            retry = int(self.window_s - (now - q[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests. Try again in {retry}s.",
                headers={"Retry-After": str(retry)},
            )
        q.append(now)
        if len(self._hits) > 10_000:
            self._purge(now)

    def _purge(self, now: float) -> None:
        stale = [k for k, q in self._hits.items() if not q or now - q[-1] > self.window_s]
        for k in stale:
            del self._hits[k]


chat_ip_limiter = RateLimiter(settings.CHAT_PER_MIN_IP, 60)
chat_session_limiter = RateLimiter(settings.CHAT_PER_MIN_SESSION, 60)
upload_limiter = RateLimiter(settings.UPLOADS_PER_10MIN, 600)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def get_or_create_session(db: dict, session_id: str) -> dict:
    now = time.time()
    if session_id in db:
        db[session_id]["last_active"] = now
        return db[session_id]

    if len(db) >= settings.MAX_SESSIONS:
        # make room by dropping sessions that have been idle too long
        expired = [
            sid for sid, s in db.items()
            if now - s.get("last_active", 0) > settings.SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del db[sid]
        if len(db) >= settings.MAX_SESSIONS:
            raise HTTPException(status_code=503, detail="Server is at capacity. Please try again later.")

    db[session_id] = {"summary": "", "history": [], "last_active": now}
    return db[session_id]