"""
Unit tests for bio_clinical_embeddings.py
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestLoadDocumentsFromArticles(unittest.TestCase):

    def test_returns_correct_number_of_documents(self):
        with patch("bio_clinical_embeddings.HuggingFaceEmbeddings"), \
             patch("bio_clinical_embeddings.download_articles"):
            from bio_clinical_embeddings import load_documents_from_articles
            articles = [
                {"pmid": "111", "title": "Title A", "abstract": "Abstract A"},
                {"pmid": "222", "title": "Title B", "abstract": "Abstract B"},
            ]
            docs = load_documents_from_articles(articles)
            self.assertEqual(len(docs), 2)

    def test_document_content_is_abstract(self):
        with patch("bio_clinical_embeddings.HuggingFaceEmbeddings"), \
             patch("bio_clinical_embeddings.download_articles"):
            from bio_clinical_embeddings import load_documents_from_articles
            articles = [{"pmid": "111", "title": "My Title", "abstract": "My Abstract"}]
            docs = load_documents_from_articles(articles)
            self.assertEqual(docs[0].page_content, "My Abstract")

    def test_document_metadata_contains_pmid_and_title(self):
        with patch("bio_clinical_embeddings.HuggingFaceEmbeddings"), \
             patch("bio_clinical_embeddings.download_articles"):
            from bio_clinical_embeddings import load_documents_from_articles
            articles = [{"pmid": "999", "title": "Test Title", "abstract": "Test Abstract"}]
            docs = load_documents_from_articles(articles)
            self.assertEqual(docs[0].metadata["source"], "999")
            self.assertEqual(docs[0].metadata["title"], "Test Title")

    def test_empty_articles_returns_empty_list(self):
        with patch("bio_clinical_embeddings.HuggingFaceEmbeddings"), \
             patch("bio_clinical_embeddings.download_articles"):
            from bio_clinical_embeddings import load_documents_from_articles
            docs = load_documents_from_articles([])
            self.assertEqual(docs, [])


class TestSearchWithScore(unittest.TestCase):

    def test_returns_filtered_results_within_threshold(self):
        with patch("bio_clinical_embeddings.HuggingFaceEmbeddings"), \
             patch("bio_clinical_embeddings.download_articles"):
            from bio_clinical_embeddings import search_with_score
            from langchain_core.documents import Document

            mock_store = MagicMock()
            doc1 = Document(page_content="Relevant", metadata={"source": "1", "title": "T1"})
            doc2 = Document(page_content="Irrelevant", metadata={"source": "2", "title": "T2"})
            mock_store.similarity_search_with_score.return_value = [
                (doc1, 0.10),
                (doc2, 0.80),
            ]

            results = search_with_score(mock_store, "prostatitis", k=5, score_threshold=0.30, min_results=1)
            scores = [score for _, score in results]
            self.assertTrue(all(s <= 0.30 for s in scores))

    def test_guarantees_min_results(self):
        with patch("bio_clinical_embeddings.HuggingFaceEmbeddings"), \
             patch("bio_clinical_embeddings.download_articles"):
            from bio_clinical_embeddings import search_with_score
            from langchain_core.documents import Document

            mock_store = MagicMock()
            doc1 = Document(page_content="Doc1", metadata={"source": "1", "title": "T1"})
            doc2 = Document(page_content="Doc2", metadata={"source": "2", "title": "T2"})
            doc3 = Document(page_content="Doc3", metadata={"source": "3", "title": "T3"})
            # All scores above threshold → should still return min_results=3
            mock_store.similarity_search_with_score.return_value = [
                (doc1, 0.50),
                (doc2, 0.60),
                (doc3, 0.70),
            ]

            results = search_with_score(mock_store, "anything", k=5, score_threshold=0.30, min_results=3)
            self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
