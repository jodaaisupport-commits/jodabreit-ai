import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class AppUtilityTests(unittest.TestCase):
    def test_chunk_text_respects_overlap_and_limit(self):
        text = ("This is a complete sentence. " * 100).strip()

        chunks = app.chunk_text(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= app.CHUNK_SIZE for chunk in chunks))
        for previous, following in zip(chunks, chunks[1:]):
            self.assertIn(previous[-180:], following)

    def test_chunk_text_returns_empty_for_whitespace(self):
        self.assertEqual(app.chunk_text(" \n\t "), [])

    def test_text_extraction_replaces_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_bytes(b"Gr\xc3\xbc\xc3\x9fe\xff")

            self.assertEqual(app.extract_text(path), "Grüße�")

    def test_file_hash_tracks_contents_not_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.txt"
            second = Path(directory) / "renamed.txt"
            first.write_text("same contents", encoding="utf-8")
            second.write_text("same contents", encoding="utf-8")
            initial = app.file_sha256(first)

            self.assertEqual(initial, app.file_sha256(second))
            second.write_text("changed contents", encoding="utf-8")
            self.assertNotEqual(initial, app.file_sha256(second))

    def test_cache_path_rejects_non_hash_names(self):
        with self.assertRaises(ValueError):
            app._cache_path("../outside", "TF-IDF (lokal & schlank)")

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is required")
    def test_index_reuses_disk_cache_for_renamed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "facts.txt"
            document.write_text(
                "Paris is the capital of France. Berlin is the capital of Germany. " * 30,
                encoding="utf-8",
            )
            renamed = root / "renamed.txt"
            renamed.write_bytes(document.read_bytes())

            with patch.object(app, "CACHE_DIR", root / ".rag_cache"):
                app._INDEX_CACHE.clear()
                index, created = app.get_index(document, "TF-IDF (lokal & schlank)")
                context, citations = app.retrieve(index, "What is the capital of France?")
                app._INDEX_CACHE.clear()
                reloaded, reindexed = app.get_index(
                    renamed, "TF-IDF (lokal & schlank)"
                )
                cache_path = app._cache_path(
                    app.file_sha256(document), "TF-IDF (lokal & schlank)"
                )
                cache_path.write_bytes(b"damaged cache")
                app._INDEX_CACHE.clear()
                rebuilt, recreated = app.get_index(
                    document, "TF-IDF (lokal & schlank)"
                )

            self.assertTrue(created)
            self.assertTrue(context)
            self.assertTrue(citations)
            self.assertFalse(reindexed)
            self.assertEqual(reloaded.payload["doc_name"], renamed.name)
            self.assertTrue(recreated)
            self.assertTrue(rebuilt.payload["chunks"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is required")
    def test_embedding_model_fallback_and_normalization(self):
        class FakeModels:
            def __init__(self):
                self.calls = []

            def embed_content(self, model, contents):
                self.calls.append((model, contents))
                if model == "gemini-embedding-001":
                    raise RuntimeError("model unavailable")
                return types.SimpleNamespace(
                    embeddings=[types.SimpleNamespace(values=[3.0, 4.0])]
                )

        class FakeClient:
            def __init__(self):
                self.models = FakeModels()

        client = FakeClient()
        index = app.RAGIndex({})
        with patch.object(app, "_genai_client", return_value=client):
            vectors, model = app._embed_chunks(["a chunk"], index)

        self.assertEqual(model, "text-embedding-004")
        self.assertEqual([call[0] for call in client.models.calls], [
            "gemini-embedding-001",
            "text-embedding-004",
        ])
        self.assertAlmostEqual(float(vectors[0][0]), 0.6)
        self.assertAlmostEqual(float(vectors[0][1]), 0.8)

    def test_groq_stream_decodes_sse_bytes_and_uses_environment_key(self):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def iter_lines(self, decode_unicode):
                self.assert_decode_unicode = decode_unicode
                yield b'data: {"choices":[{"delta":{"content":"Hallo"}}]}'
                yield b"data: [DONE]"

        response = FakeResponse()
        test_key = "test-placeholder"
        with patch.dict(os.environ, {"GROQ_API_KEY": test_key}):
            with patch("requests.post", return_value=response) as post:
                result = list(app._groq_stream([], "test-model"))

        self.assertEqual(result, ["Hallo"])
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer " + test_key
        )

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio is required")
    def test_build_app(self):
        interface = app.build_app()
        self.assertIsNotNone(interface)
        interface.close()

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is required")
    def test_binary_upload_uses_content_hash_cache(self):
        content = b"Paris is the capital of France. " * 30
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app, "CACHE_DIR", Path(directory) / ".rag_cache"):
                app._INDEX_CACHE.clear()
                index, created = app.get_index_from_bytes(
                    content, ".txt", "TF-IDF (lokal & schlank)"
                )
                app._INDEX_CACHE.clear()
                reloaded, reindexed = app.get_index_from_bytes(
                    content, ".txt", "TF-IDF (lokal & schlank)"
                )

        self.assertTrue(created)
        self.assertFalse(reindexed)
        self.assertEqual(index.payload["chunks"], reloaded.payload["chunks"])


if __name__ == "__main__":
    unittest.main()
