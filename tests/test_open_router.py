"""
Unit tests for open_router.py
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestGenerateMessagesFromDocuments(unittest.TestCase):

    def setUp(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key"

    def test_messages_contain_user_query(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from open_router import generate_messages_from_documents
            from langchain_core.documents import Document

            doc = Document(page_content="Abstract content", metadata={"source": "123", "title": "Test Title"})
            messages = generate_messages_from_documents([doc], "What are treatments for prostatitis?")

            user_message = messages[-1]["content"]
            self.assertIn("What are treatments for prostatitis?", user_message)

    def test_messages_contain_document_content(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from open_router import generate_messages_from_documents
            from langchain_core.documents import Document

            doc = Document(page_content="Special abstract text", metadata={"source": "456", "title": "Doc Title"})
            messages = generate_messages_from_documents([doc], "Test query")

            user_message = messages[-1]["content"]
            self.assertIn("Special abstract text", user_message)

    def test_system_message_is_first(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from open_router import generate_messages_from_documents
            from langchain_core.documents import Document

            doc = Document(page_content="Abstract", metadata={"source": "1", "title": "T"})
            messages = generate_messages_from_documents([doc], "query")
            self.assertEqual(messages[0]["role"], "system")

    def test_multiple_documents_all_included(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from open_router import generate_messages_from_documents
            from langchain_core.documents import Document

            docs = [
                Document(page_content=f"Abstract {i}", metadata={"source": str(i), "title": f"Title {i}"})
                for i in range(3)
            ]
            messages = generate_messages_from_documents(docs, "query")
            user_content = messages[-1]["content"]
            for i in range(3):
                self.assertIn(f"Abstract {i}", user_content)


class TestQueryOpenRouter(unittest.TestCase):

    def setUp(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key"

    @patch("open_router.requests.post")
    def test_returns_response_on_success(self, mock_post):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from open_router import query_openrouter
            from langchain_core.documents import Document

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"choices": [{"message": {"content": "Test answer"}}]}
            mock_post.return_value = mock_response

            doc = Document(page_content="Abstract", metadata={"source": "1", "title": "T"})
            result = query_openrouter([doc], "test query")
            self.assertIsNotNone(result)
            self.assertIn("choices", result)

    @patch("open_router.requests.post")
    def test_returns_none_on_api_error(self, mock_post):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from open_router import query_openrouter
            from langchain_core.documents import Document

            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_post.return_value = mock_response

            doc = Document(page_content="Abstract", metadata={"source": "1", "title": "T"})
            result = query_openrouter([doc], "test query")
            self.assertIsNone(result)

    @patch("open_router.requests.post")
    def test_returns_none_on_exception(self, mock_post):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from open_router import query_openrouter
            from langchain_core.documents import Document

            mock_post.side_effect = Exception("Connection error")
            doc = Document(page_content="Abstract", metadata={"source": "1", "title": "T"})
            result = query_openrouter([doc], "test query")
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
