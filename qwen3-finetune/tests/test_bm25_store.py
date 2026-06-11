"""测试 rag/bm25_store.py BM25 检索存储"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))

from bm25_store import BM25Store


class TestBM25StoreInit:
    """测试初始化"""

    def test_create_new_index(self, temp_dir):
        store = BM25Store(str(temp_dir), "test_coll")
        assert store.count() == 0
        assert store.index_dir.exists()

    def test_open_existing_index(self, temp_dir):
        store1 = BM25Store(str(temp_dir), "test_coll")
        store1.add_documents(["hello world"], [{"repo": "a/b"}])
        assert store1.count() == 1

        store2 = BM25Store(str(temp_dir), "test_coll")
        assert store2.count() == 1

    def test_multiple_collections(self, temp_dir):
        store1 = BM25Store(str(temp_dir), "coll_a")
        store1.add_documents(["doc in a"], [{}])
        store2 = BM25Store(str(temp_dir), "coll_b")
        store2.add_documents(["doc in b"], [{}])

        assert store1.count() == 1
        assert store2.count() == 1


class TestAddDocuments:
    """测试文档添加"""

    def test_add_and_count(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(["doc one", "doc two"], [{}, {}])
        assert store.count() == 2

    def test_add_with_metadata(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(
            ["some content"],
            [{"repo": "django/django", "file_type": "py"}],
        )
        assert store.count() == 1

        results = store.similarity_search("some content", k=1)
        assert results[0]["metadata"]["repo"] == "django/django"
        assert results[0]["metadata"]["file_type"] == "py"

    def test_add_empty_document(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(["   ", "  real doc  "], [{}, {}])
        assert store.count() == 1

    def test_batch_add(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        docs = [f"document number {i}" for i in range(50)]
        metas = [{"chunk_index": i} for i in range(50)]
        store.add_documents(docs, metas)
        assert store.count() == 50


class TestSimilaritySearch:
    """测试检索"""

    def test_basic_search(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(
            ["Python asyncio guide for beginners", "Django ORM tutorial"],
            [{}, {}],
        )
        results = store.similarity_search("asyncio", k=3)
        assert len(results) >= 1
        assert "asyncio" in results[0]["content"]

    def test_search_returns_k_results(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        docs = [f"document about topic {i}" for i in range(20)]
        store.add_documents(docs, [{} for _ in docs])

        results = store.similarity_search("document", k=5)
        assert len(results) == 5

    def test_empty_query(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(["some document"], [{}])
        results = store.similarity_search("  ", k=5)
        assert results == []

    def test_search_on_empty_index(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        results = store.similarity_search("anything", k=5)
        assert results == []

    def test_score_ordering(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(
            ["asyncio asyncio asyncio asyncio", "asyncio"],
            [{}, {}],
        )
        results = store.similarity_search("asyncio", k=2)
        assert len(results) == 2
        assert results[0]["distance"] >= results[1]["distance"]

    def test_relevance(self, temp_dir):
        """相关文档应排在不相关文档前面"""
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(
            [
                "Django database connection pooling configuration",
                "Python web scraping with BeautifulSoup",
                "Django ORM query optimization techniques",
            ],
            [{}, {}, {}],
        )
        results = store.similarity_search("Django database", k=3)
        assert "django" in results[0]["content"].lower()


class TestMetadataFiltering:
    """测试 metadata 过滤"""

    def test_filter_by_repo(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(
            ["Django ORM guide", "Flask routing guide"],
            [{"repo": "django/django"}, {"repo": "pallets/flask"}],
        )
        results = store.similarity_search("guide", k=5, filter_meta={"repo": "django/django"})
        assert len(results) >= 1
        for r in results:
            assert r["metadata"]["repo"] == "django/django"

    def test_filter_no_match(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(
            ["content here"], [{"repo": "test/repo"}],
        )
        results = store.similarity_search(
            "content", k=5, filter_meta={"repo": "nonexistent/repo"}
        )
        assert len(results) == 0


class TestDeleteCollection:
    """测试删除"""

    def test_delete_resets_count(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(["doc1", "doc2"], [{}, {}])
        assert store.count() == 2

        store.delete_collection()
        assert store.count() == 0


class TestInterfaceCompatibility:
    """验证 BM25Store 与 VectorStore 接口一致性"""

    def test_methods_exist(self):
        store = BM25Store.__new__(BM25Store)
        assert hasattr(store, "add_documents")
        assert hasattr(store, "similarity_search")
        assert hasattr(store, "delete_collection")
        assert hasattr(store, "count")

    def test_search_return_format(self, temp_dir):
        store = BM25Store(str(temp_dir), "test")
        store.add_documents(["test document"], [{"file_type": "py"}])
        results = store.similarity_search("test", k=1)
        assert len(results) == 1

        r = results[0]
        assert "id" in r
        assert "content" in r
        assert "metadata" in r
        assert "distance" in r
        assert r["metadata"]["file_type"] == "py"
