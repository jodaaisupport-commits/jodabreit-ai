import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
SYSTEM_PROMPT = "Du bist ein hilfreicher KI-Assistent. Antworte auf Deutsch."

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBED_BATCH = 50
TOP_K = 4
MAX_CONTEXT_CHARS = 8000
CACHE_DIR = Path(".rag_cache")

PROVIDER_MODELS = {
    "Groq": {
        "Llama 3.3 70B (stark)": "llama-3.3-70b-versatile",
        "Llama 3.1 8B (schnell)": "llama-3.1-8b-instant",
        "GPT-OSS 120B": "openai/gpt-oss-120b",
    },
    "Gemini": {
        "Gemini 3.6 Flash (Standard)": "gemini-3.6-flash",
        "Gemini 3.5 Flash": "gemini-3.5-flash",
        "Gemini 3.5 Flash Lite (schnell)": "gemini-3.5-flash-lite",
    },
}

_INDEX_CACHE = {}


def _raise_gradio_error(message):
    try:
        import gradio as gr
    except ImportError:
        raise ValueError(message) from None
    raise gr.Error(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()[:16]


def extract_text(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            _raise_gradio_error("Zum Lesen von PDFs muss pypdf installiert sein.")
        try:
            reader = PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            _raise_gradio_error(f"Das PDF konnte nicht gelesen werden: {exc}")
    else:
        _raise_gradio_error("Nicht unterstützter Dateityp. Bitte eine PDF- oder TXT-Datei wählen.")

    text = text.strip()
    if not text:
        _raise_gradio_error(
            "Das Dokument enthält keinen lesbaren Text. Bei gescannten PDFs wird OCR benötigt."
        )
    return text


def chunk_text(text):
    text = text.strip()
    chunks = []
    start = 0
    boundaries = ("\n\n", ". ", "! ", "? ")

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            candidates = [
                (
                    text.rfind(
                        separator,
                        start + CHUNK_SIZE // 2,
                        end - len(separator) + 1,
                    ),
                    separator,
                )
                for separator in boundaries
            ]
            position, separator = max(candidates, key=lambda candidate: candidate[0])
            if position > start + CHUNK_SIZE // 2:
                end = position + len(separator)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _genai_client():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _raise_gradio_error(
            "Für Gemini Embeddings wird GEMINI_API_KEY benötigt. "
            "Alternativ den TF-IDF-Modus wählen."
        )
    return genai.Client(api_key=api_key)


class RAGIndex:
    def __init__(self, payload):
        self.payload = payload
        self._client = None

    @property
    def gemini_client(self):
        if self._client is None:
            self._client = _genai_client()
        return self._client


def _embedding_values(response):
    return [embedding.values for embedding in response.embeddings]


def _embed_chunks(chunks, index, model=None):
    import numpy as np

    client = index.gemini_client
    model = model or "gemini-embedding-001"
    vectors = []
    for offset in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[offset : offset + EMBED_BATCH]
        try:
            response = client.models.embed_content(model=model, contents=batch)
        except Exception:
            # Fallback for accounts where gemini-embedding-001 is unavailable.
            if model != "gemini-embedding-001":
                raise
            model = "text-embedding-004"
            response = client.models.embed_content(model=model, contents=batch)
        vectors.extend(_embedding_values(response))
    return _normalize(vectors, np), model


def _normalize(vectors, np):
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _cache_path(digest, mode):
    suffix = "tfidf" if mode == "TF-IDF (lokal & schlank)" else "gemini"
    return CACHE_DIR / f"{digest}_{suffix}.pkl"


def _load_pickle(cache_path, kind):
    try:
        with cache_path.open("rb") as file:
            payload = pickle.load(file)
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != kind
            or not isinstance(payload.get("chunks"), list)
            or not all(isinstance(chunk, str) for chunk in payload["chunks"])
            or "matrix" not in payload
            or (kind == "tfidf" and "vectorizer" not in payload)
            or (kind == "gemini" and not payload.get("embedding_model"))
        ):
            return None
        return payload
    except Exception:
        return None


def _build_index(path, mode, digest):
    chunks = chunk_text(extract_text(path))
    if not chunks:
        _raise_gradio_error("Im Dokument wurden keine Textabschnitte gefunden.")

    payload = {
        "kind": "tfidf" if mode == "TF-IDF (lokal & schlank)" else "gemini",
        "doc_name": Path(path).name,
        "chunks": chunks,
    }
    if payload["kind"] == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        try:
            payload["matrix"] = vectorizer.fit_transform(chunks)
        except ValueError as exc:
            _raise_gradio_error(f"Das Dokument enthält keine indexierbaren Wörter: {exc}")
        payload["vectorizer"] = vectorizer
    else:
        payload["matrix"], payload["embedding_model"] = _embed_chunks(
            chunks, RAGIndex(payload)
        )

    cache_path = _cache_path(digest, mode)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=CACHE_DIR, prefix=f"{digest}_", suffix=".tmp", delete=False
        ) as file:
            temporary_path = Path(file.name)
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, cache_path)
    except OSError:
        try:
            temporary_path.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):
            pass
    return payload


def get_index(path, mode):
    digest = file_sha256(path)
    cache_key = (str(Path(path).resolve()), mode)
    cached = _INDEX_CACHE.get(cache_key)
    if cached and cached[0] == digest:
        return RAGIndex(cached[1]), False

    cache_path = _cache_path(digest, mode)
    kind = "tfidf" if mode == "TF-IDF (lokal & schlank)" else "gemini"
    payload = _load_pickle(cache_path, kind) if cache_path.exists() else None
    created = payload is None
    if created:
        payload = _build_index(path, mode, digest)
    else:
        payload["doc_name"] = Path(path).name

    _INDEX_CACHE[cache_key] = (digest, payload)
    return RAGIndex(payload), created


def retrieve(index, query):
    import numpy as np

    payload = index.payload
    if payload["kind"] == "tfidf":
        from sklearn.metrics.pairwise import cosine_similarity

        query_vector = payload["vectorizer"].transform([query])
        similarities = cosine_similarity(query_vector, payload["matrix"])[0]
        threshold = 0.01
    else:
        query_vector, _ = _embed_chunks(
            [query], index, model=payload.get("embedding_model")
        )
        similarities = payload["matrix"] @ query_vector[0]
        threshold = 0.3

    best = np.argsort(similarities)[::-1]
    hits = [
        (int(i), float(similarities[i]))
        for i in best
        if similarities[i] > threshold
    ][:TOP_K]

    sections = []
    context_length = 0
    for chunk_number, score in hits:
        chunk = payload["chunks"][chunk_number]
        section = f"[Abschnitt {chunk_number + 1}]\n{chunk}"
        separator_length = len("\n\n---\n\n") if sections else 0
        remaining = MAX_CONTEXT_CHARS - context_length - separator_length
        if remaining <= 0:
            break
        section = section[:remaining]
        sections.append(section)
        context_length += len(section) + separator_length

    citations = [(number + 1, score) for number, score in hits]
    return "\n\n---\n\n".join(sections), citations


def _rag_system_prompt(context):
    if not context:
        return (
            f"{SYSTEM_PROMPT} Es wurden keine passenden Abschnitte gefunden. "
            "Sage offen, wenn eine Information im Dokument fehlt, und erfinde nichts."
        )
    return (
        f"{SYSTEM_PROMPT} Beantworte Fragen zum Dokument ausschließlich anhand der "
        "folgenden Abschnitte. Wenn die Information darin fehlt, sage das offen.\n\n"
        f"{context}"
    )


def _groq_stream(messages, model):
    import requests

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        _raise_gradio_error("Bitte den API-Key in der Umgebungsvariable GROQ_API_KEY setzen.")
    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
        timeout=(15, 120),
    )
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        delta = event.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            yield delta


def _gemini_stream(messages, model, system_prompt):
    from google.genai import types

    client = _genai_client()
    contents = [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in messages
        if message["role"] in {"user", "assistant"}
    ]
    config = types.GenerateContentConfig(system_instruction=system_prompt)
    stream = client.models.generate_content_stream(
        model=model, contents=contents, config=config
    )
    for response in stream:
        if response.text:
            yield response.text


def chat(message, history, provider, model_label, retrieval_mode, document):
    history = list(history or [])
    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)
    ]

    system_prompt = SYSTEM_PROMPT
    citations = []
    if document:
        index, created = get_index(document, retrieval_mode)
        if created:
            import gradio as gr

            gr.Info(
                f"📚 Index erstellt & gespeichert: {len(index.payload['chunks'])} Abschnitte "
                f"aus „{Path(document).name}“"
            )
        context, citations = retrieve(index, message)
        system_prompt = _rag_system_prompt(context)

    model = PROVIDER_MODELS[provider][model_label]
    if provider == "Groq":
        request_messages = [{"role": "system", "content": system_prompt}, *messages]
        request_messages.append({"role": "user", "content": message})
        stream = _groq_stream(request_messages, model)
    else:
        messages.append({"role": "user", "content": message})
        stream = _gemini_stream(messages, model, system_prompt)

    conversation = [*history, {"role": "user", "content": message}]
    answer = ""
    try:
        for token in stream:
            answer += token
            yield [*conversation, {"role": "assistant", "content": answer}]
    except Exception as exc:
        _raise_gradio_error(f"Die Anfrage ist fehlgeschlagen: {exc}")

    if citations:
        footer = "\n\n---\n📎 **Gefundene Abschnitte:** " + ", ".join(
            f"#{number} ({score:.2f})" for number, score in citations
        )
        answer += footer
    yield [*conversation, {"role": "assistant", "content": answer}]


def build_app():
    import gradio as gr

    with gr.Blocks(title="Kostenloser KI-Chat mit Dokumenten-RAG") as app:
        gr.Markdown("# 🆓 KI-Chat mit Dokumenten-RAG")
        with gr.Row():
            provider = gr.Dropdown(
                choices=list(PROVIDER_MODELS), value="Groq", label="Anbieter"
            )
            model = gr.Dropdown(
                choices=list(PROVIDER_MODELS["Groq"]),
                value=next(iter(PROVIDER_MODELS["Groq"])),
                label="Modell",
            )
            retrieval_mode = gr.Dropdown(
                choices=["TF-IDF (lokal & schlank)", "Gemini Embeddings (semantisch)"],
                value="TF-IDF (lokal & schlank)",
                label="Dokumentensuche",
            )
        document = gr.File(
            label="Dokument (optional, PDF oder TXT)",
            file_types=[".pdf", ".txt"],
            type="filepath",
        )
        chatbot = gr.Chatbot(label="Chat")
        message = gr.Textbox(
            label="Nachricht", placeholder="Stelle eine Frage …", lines=2
        )

        provider.change(
            lambda selected: gr.update(
                choices=list(PROVIDER_MODELS[selected]),
                value=next(iter(PROVIDER_MODELS[selected])),
            ),
            inputs=provider,
            outputs=model,
        )
        message.submit(
            chat,
            inputs=[message, chatbot, provider, model, retrieval_mode, document],
            outputs=chatbot,
        ).then(lambda: "", outputs=message)

    return app


if __name__ == "__main__":
    build_app().queue().launch()
