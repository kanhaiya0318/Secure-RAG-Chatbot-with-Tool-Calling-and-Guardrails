import uuid
import time
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
MAX_CHARS = 200 

st.set_page_config(page_title="FastAPI Chatbot", page_icon="💬")
st.title("💬 FastAPI Chatbot")


# ---------- helpers ----------
def create_chat() -> str:
    sid = str(uuid.uuid4())
    st.session_state.chats[sid] = {"title": "New chat", "messages": []}
    return sid


def new_chat():
    # don't pile up empty chats
    current = st.session_state.chats[st.session_state.current]
    if current["messages"]:
        st.session_state.current = create_chat()


def switch_chat(sid: str):
    st.session_state.current = sid


def delete_chat(sid: str):
    """Delete one chat on the backend and in the UI."""
    try:
        requests.post(f"{API_URL}/sessions/{sid}/end", timeout=10)
    except requests.RequestException:
        pass  # 404 / backend down: still remove it from the UI
    st.session_state.chats.pop(sid, None)
    if st.session_state.current == sid:
        remaining = list(st.session_state.chats)
        st.session_state.current = remaining[-1] if remaining else create_chat()

def chat_stream(session_id: str, prompt: str):
    try:
        with requests.post(
            f"{API_URL}/chat/query",
            json={"session_id":session_id, "message": prompt},
            stream=True,
            timeout = (20.0, 20.0)             # (connect, time bw 2 chunks)
        ) as resp:
            resp.raise_for_status()
            resp.encoding = "utf-8"
            for piece in resp.iter_content(chunk_size = None ,decode_unicode=True):
                if piece:
                    yield piece
    except requests.HTTPError as e:
        code = e.response.status_code
        if code == 429:
            wait = e.response.headers.get("Retry-After", "a few")
            yield f"⏳ Too many requests. Please wait {wait} seconds."
        elif code == 503:
            yield "⚠️ The server is busy. Please try again later."
        else:
            yield f"⚠️ Backend error ({code})"


def fetch_documents():
    try:
        resp = requests.get(f"{API_URL}/rag/documents", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def upload_pdf(file):
    try:
        resp = requests.post(
            f"{API_URL}/rag/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=120,
        )
        if resp.status_code >= 400:
            detail = resp.json().get("detail", resp.text)
            return False, f"⚠️ {detail}"
        data = resp.json()
        if data["skipped"]:
            return True, f"Already indexed: {data['source']}"
        return True, f"Indexed {data['source']} ({data['chunks']} chunks)"
    except requests.ConnectionError:
        return False, "⚠️ Can't reach the backend. Is uvicorn running on port 8000?"
    except requests.RequestException as e:
        return False, f"⚠️ Upload failed: {e}"


def delete_document(doc_id: str):
    try:
        requests.delete(f"{API_URL}/rag/documents/{doc_id}", timeout=10)
    except requests.RequestException:
        st.toast("⚠️ Could not delete the document")


def timed(gen):
    t0 = time.perf_counter(); first = None
    for piece in gen:
        if first is None:
            first = time.perf_counter() - t0
        yield piece
    print(f"[ui] first word {first:.1f}s | total {time.perf_counter()-t0:.1f}s")



# ---------- session state ----------
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.current = create_chat()

current_id = st.session_state.current
chat = st.session_state.chats[current_id]

# ---------- sidebar ----------
with st.sidebar:
    st.button("➕ New chat", on_click=new_chat, use_container_width=True)

    st.divider()
    st.subheader("📄 Documents")
    pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")
    if pdf and st.button("Index PDF", use_container_width=True):
        with st.spinner("Indexing..."):
            ok, msg = upload_pdf(pdf)
        (st.success if ok else st.error)(msg)

    docs = fetch_documents()
    if docs is None:
        st.caption("Documents unavailable")
    elif not docs:
        st.caption("No documents yet")
    for d in docs or []:
        c1, c2 = st.columns([5, 1])
        c1.caption(f"{d['source']} · {d['chunks']} chunks")
        c2.button("🗑️", key=f"deldoc_{d['doc_id']}", on_click=delete_document, args=(d["doc_id"],))
    st.divider()

    st.subheader("Chats")
    for sid, c in reversed(list(st.session_state.chats.items())):
        col1, col2 = st.columns([5, 1])
        col1.button(
            c["title"],
            key=f"open_{sid}",
            on_click=switch_chat,
            args=(sid,),
            type="primary" if sid == current_id else "secondary",
            use_container_width=True,
        )
        col2.button("🗑️", key=f"del_{sid}", on_click=delete_chat, args=(sid,))

# ---------- render history ----------
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- handle new input ----------
if prompt := st.chat_input("Type your message...", max_chars=MAX_CHARS):
    if not chat["messages"]:
        chat["title"] = prompt[:30] + ("…" if len(prompt) > 30 else "")
    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # with st.spinner("Thinking..."):
        #     answer = chat_n(session_id, prompt)
        # st.markdown(answer)
        answer = st.write_stream(timed(chat_stream(current_id, prompt)))
        

    chat["messages"].append({"role": "assistant", "content": answer})
    st.rerun()  # refresh sidebar so the new title shows