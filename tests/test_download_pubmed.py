"""
Unit tests for download_pubmed.py
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from download_pubmed import preprocess_text, get_data_file, search_pubmed, fetch_articles_batch


def _make_mock_session(response=None, side_effect=None):
    """Helper: returns a mock that replaces _make_session() in download_pubmed."""
    mock_session = MagicMock()
    if side_effect is not None:
        mock_session.get.side_effect = side_effect
    else:
        mock_session.get.return_value = response
    return mock_session


class TestPreprocessText(unittest.TestCase):

    def test_strips_whitespace(self):
        self.assertEqual(preprocess_text("  hello world  "), "hello world")

    def test_removes_multiple_spaces(self):
        self.assertEqual(preprocess_text("hello   world"), "hello world")

    def test_removes_special_characters(self):
        result = preprocess_text("hello@world#test!")
        self.assertNotIn("@", result)
        self.assertNotIn("#", result)
        self.assertNotIn("!", result)

    def test_keeps_dots_and_commas(self):
        result = preprocess_text("Dr. Smith, M.D.")
        self.assertIn(".", result)
        self.assertIn(",", result)

    def test_empty_string(self):
        self.assertEqual(preprocess_text(""), "")


class TestGetDataFile(unittest.TestCase):

    def test_returns_correct_path(self):
        path = get_data_file("prostatitis", data_dir="/tmp")
        self.assertTrue(path.startswith("/tmp"))
        self.assertIn("prostatitis", path)
        self.assertTrue(path.endswith(".txt"))

    def test_replaces_spaces_with_underscores(self):
        path = get_data_file("chronic prostatitis", data_dir="/tmp")
        self.assertIn("chronic_prostatitis", path)

    def test_lowercases_query(self):
        path = get_data_file("Prostatitis", data_dir="/tmp")
        self.assertIn("prostatitis", path)

    def test_truncates_long_query(self):
        long_query = "a" * 200
        path = get_data_file(long_query, data_dir="/tmp")
        filename = os.path.basename(path)
        # filename = <query>_data.txt, so query part should be <= 80 chars
        self.assertLessEqual(len(filename), 90)


class TestSearchPubmed(unittest.TestCase):

    @patch("download_pubmed._make_session")
    def test_returns_id_list_on_success(self, mock_make_session):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "esearchresult": {"idlist": ["12345", "67890"]}
        }
        mock_response.raise_for_status = MagicMock()
        mock_make_session.return_value = _make_mock_session(response=mock_response)

        result = search_pubmed("prostatitis", retmax=2)
        self.assertEqual(result, ["12345", "67890"])

    @patch("download_pubmed._make_session")
    def test_returns_empty_list_on_network_error(self, mock_make_session):
        import requests as req
        mock_make_session.return_value = _make_mock_session(
            side_effect=req.exceptions.RequestException("Network error")
        )
        result = search_pubmed("prostatitis")
        self.assertEqual(result, [])


class TestFetchArticlesBatch(unittest.TestCase):

    @patch("download_pubmed._make_session")
    def test_parses_xml_correctly(self, mock_make_session):
        xml_content = b"""<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>12345</PMID>
              <Article>
                <ArticleTitle>Test Article Title</ArticleTitle>
                <Abstract>
                  <AbstractText>This is a test abstract.</AbstractText>
                </Abstract>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>"""

        mock_response = MagicMock()
        mock_response.content = xml_content
        mock_response.raise_for_status = MagicMock()
        mock_make_session.return_value = _make_mock_session(response=mock_response)

        result = fetch_articles_batch(["12345"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pmid"], "12345")
        self.assertIn("Test Article Title", result[0]["title"])
        self.assertIn("test abstract", result[0]["abstract"])

    @patch("download_pubmed._make_session")
    def test_skips_articles_without_abstract(self, mock_make_session):
        xml_content = b"""<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>99999</PMID>
              <Article>
                <ArticleTitle>No Abstract Article</ArticleTitle>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>"""

        mock_response = MagicMock()
        mock_response.content = xml_content
        mock_response.raise_for_status = MagicMock()
        mock_make_session.return_value = _make_mock_session(response=mock_response)

        result = fetch_articles_batch(["99999"])
        self.assertEqual(result, [])

    @patch("download_pubmed._make_session")
    def test_returns_empty_on_network_error(self, mock_make_session):
        import requests as req
        mock_make_session.return_value = _make_mock_session(
            side_effect=req.exceptions.RequestException("fail")
        )
        result = fetch_articles_batch(["123"])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
