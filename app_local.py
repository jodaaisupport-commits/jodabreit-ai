import json

import requests

import app as cloud_app


OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"
LOCAL_MODELS = ["llama3.2", "qwen2.5:7b", "mistral"]


def _ollama_stream(messages, model):
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        OLLAMA_API_URL,
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
        timeout=(5, 120),
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        content = event.get("message", {}).get("content")
        if content:
            yield content


def chat_local(message, history, model, document, document_type):
    history = list(history or [])
    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]
    system_prompt = cloud_app.SYSTEM_PROMPT
    citations = []

    if document:
        index, created = cloud_app.get_index_from_bytes(
            document, document_type, "TF-IDF (lokal & schlank)"
        )
        if created:
            import gradio as gr

            gr.Info(
                f"📚 Index erstellt & gespeichert: {len(index.payload['chunks'])} Abschnitte "
                f"aus „{index.payload['doc_name']}“"
            )
        context, citations = cloud_app.retrieve(index, message)
        system_prompt = cloud_app._rag_system_prompt(context)

    request_messages = [
        {"role": "system", "content": system_prompt},
        *messages,
        {"role": "user", "content": message},
    ]
    conversation = [*history, {"role": "user", "content": message}]
    answer = ""
    try:
        for token in _ollama_stream(request_messages, model):
            answer += token
            yield [*conversation, {"role": "assistant", "content": answer}]
    except Exception as exc:
        cloud_app._raise_gradio_error(
            "Die lokale Ollama-Anfrage ist fehlgeschlagen. Läuft Ollama unter "
            f"http://127.0.0.1:11434 und ist das Modell installiert? ({exc})"
        )

    if citations:
        footer = "\n\n---\n📎 **Gefundene Abschnitte:** " + ", ".join(
            f"#{number} ({score:.2f})" for number, score in citations
        )
        answer += footer
    yield [*conversation, {"role": "assistant", "content": answer}]


def build_app():
    import gradio as gr

    with gr.Blocks(title="Lokaler Ollama-Chat mit Dokumenten-RAG") as interface:
        gr.Markdown(
            "# 🖥️ Lokaler KI-Chat\n"
            "Chat und TF-IDF-Dokumentsuche laufen lokal; Anfragen gehen nur an Ollama."
        )
        model = gr.Dropdown(
            choices=LOCAL_MODELS,
            value=LOCAL_MODELS[0],
            allow_custom_value=True,
            label="Ollama-Modell (zuvor mit `ollama pull` installieren)",
        )
        document = gr.File(
            label="Dokument (optional, PDF oder TXT)",
            file_types=[".pdf", ".txt"],
            type="binary",
        )
        document_type = gr.Dropdown(
            choices=[".pdf", ".txt"], value=".pdf", label="Dateityp des Dokuments"
        )
        chatbot = gr.Chatbot(label="Chat")
        message = gr.Textbox(
            label="Nachricht", placeholder="Stelle eine Frage …", lines=2
        )
        message.submit(
            chat_local,
            inputs=[message, chatbot, model, document, document_type],
            outputs=chatbot,
        ).then(lambda: "", outputs=message)
    return interface


if __name__ == "__main__":
    build_app().queue().launch()
