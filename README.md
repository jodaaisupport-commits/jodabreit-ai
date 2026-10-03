# jodabreit-ai

Kostenloser KI-Chat mit Dokumenten-RAG. Die Gradio-App unterstützt Groq- und Gemini-Modelle, PDF-/TXT-Uploads sowie lokale TF-IDF- oder semantische Gemini-Embedding-Suche.

## Starten

```bash
python -m pip install -r requirements.txt
python app.py
```

Für Groq-Modelle `GROQ_API_KEY`, für Gemini-Modelle und Gemini Embeddings `GEMINI_API_KEY` als Umgebungsvariable setzen. Die TF-IDF-Dokumentsuche funktioniert lokal und benötigt keinen Gemini-Key. Dokumentindizes werden in `.rag_cache/` gespeichert.
