# jodabreit-ai

Kostenloser KI-Chat mit Dokumenten-RAG. Die Gradio-App unterstützt Groq- und Gemini-Modelle, PDF-/TXT-Uploads sowie lokale TF-IDF- oder semantische Gemini-Embedding-Suche.

## Starten

```bash
python -m pip install -r requirements.txt
python app.py
```

Für Groq-Modelle `GROQ_API_KEY`, für Gemini-Modelle und Gemini Embeddings `GEMINI_API_KEY` als Umgebungsvariable setzen. Die TF-IDF-Dokumentsuche funktioniert lokal und benötigt keinen Gemini-Key. Dokumentindizes werden in `.rag_cache/` gespeichert.

## Lokal mit Ollama

Ollama installieren und in einem Terminal den lokalen Dienst starten:

```bash
ollama serve
```

In einem zweiten Terminal ein Modell laden (dafür ist beim ersten Mal Internet nötig) und die App starten:

```bash
ollama pull llama3.2
python app_local.py
```

Nach dem Modell-Download sendet die lokale Variante Chat-Anfragen ausschließlich an `127.0.0.1:11434`. Dokumente werden mit TF-IDF lokal indexiert; es werden keine Cloud-APIs benötigt.
