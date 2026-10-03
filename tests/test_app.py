import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
