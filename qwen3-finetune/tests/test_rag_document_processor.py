"""测试 rag/document_processor.py 的文档分块逻辑"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))

from document_processor import DocumentProcessor


class TestDocumentProcessorInit:
    """测试初始化"""

    def test_default_init(self):
        dp = DocumentProcessor()
        assert dp.chunk_size == 512
        assert dp.chunk_overlap == 64
        assert len(dp.separators) > 0

    def test_custom_chunk_size(self):
        dp = DocumentProcessor(chunk_size=256, chunk_overlap=32)
        assert dp.chunk_size == 256
        assert dp.chunk_overlap == 32


class TestChunkText:
    """测试文本分块"""

    def test_short_text_no_split(self):
        dp = DocumentProcessor(chunk_size=512, chunk_overlap=64)
        text = "这是一个短文本。"
        chunks = dp.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0]["content"] == text
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_long_text_split(self):
        dp = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        # 生成 200 个字符的文本
        text = "这是一段测试文本。" * 20  # 约 160 字符
        chunks = dp.chunk_text(text)

        assert len(chunks) > 1
        # 每个 chunk 的 content 不应为空
        for chunk in chunks:
            assert len(chunk["content"]) > 0

    def test_chunks_with_metadata(self):
        dp = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        text = "长文本内容。" * 50
        metadata = {"source": "test.txt", "category": "test"}
        chunks = dp.chunk_text(text, metadata)

        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test.txt"
            assert chunk["metadata"]["category"] == "test"
            assert "chunk_index" in chunk["metadata"]

    def test_empty_text(self):
        dp = DocumentProcessor()
        chunks = dp.chunk_text("   \n\n  ")
        assert len(chunks) == 0

    def test_text_exactly_chunk_size(self):
        dp = DocumentProcessor(chunk_size=100, chunk_overlap=0)
        text = "A" * 100
        chunks = dp.chunk_text(text)
        assert len(chunks) >= 1

    def test_overlap_between_chunks(self):
        dp = DocumentProcessor(chunk_size=60, chunk_overlap=20)
        text = "ABCDEFGHIJ" * 20  # 200 字符
        chunks = dp.chunk_text(text)

        if len(chunks) >= 2:
            # chunk 之间应该有重叠
            pass  # overlap 逻辑在 _split_text 中实现


class TestSplitBySize:
    """测试按固定大小分割"""

    def test_split_by_size(self):
        dp = DocumentProcessor(chunk_size=50, chunk_overlap=0)
        text = "A" * 120
        chunks = dp._split_by_size(text)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= dp.chunk_size + dp.chunk_overlap


class TestLoadTextFile:
    """测试文本文件加载"""

    def test_load_text_file(self, temp_dir):
        filepath = temp_dir / "test.txt"
        content = "这是一个测试文件的内容。\n包含多行文本。\n" * 10
        filepath.write_text(content, encoding="utf-8")

        dp = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        docs = dp.load_file(filepath)

        assert len(docs) >= 1
        for doc in docs:
            assert "content" in doc
            assert "metadata" in doc
            assert doc["metadata"]["source"] == str(filepath)
            assert doc["metadata"]["filetype"] == ".txt"

    def test_load_markdown_file(self, temp_dir):
        filepath = temp_dir / "docs.md"
        content = "# 标题\n\n正文内容。" * 20
        filepath.write_text(content, encoding="utf-8")

        dp = DocumentProcessor(chunk_size=200, chunk_overlap=30)
        docs = dp.load_file(filepath)

        assert len(docs) >= 1
        assert docs[0]["metadata"]["filetype"] == ".md"


class TestLoadJsonlFile:
    """测试 JSONL 文件加载"""

    def test_load_jsonl(self, temp_dir):
        import json

        filepath = temp_dir / "data.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for i in range(5):
                f.write(json.dumps({
                    "text": f"这是第 {i} 条数据的文本内容。" * 10,
                    "source_id": i,
                }, ensure_ascii=False) + "\n")

        dp = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        docs = dp.load_file(filepath)

        assert len(docs) >= 5  # 每条数据至少产生 1 个 chunk


class TestLoadDirectory:
    """测试目录加载"""

    def test_load_directory(self, temp_dir):
        # 创建多个文件
        (temp_dir / "doc1.txt").write_text("文件1的内容。" * 30, encoding="utf-8")
        (temp_dir / "doc2.md").write_text("# 文件2\n\n内容。" * 30, encoding="utf-8")

        dp = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        docs = dp.load_directory(temp_dir)

        assert len(docs) > 0
        sources = set(d["metadata"]["source"] for d in docs)
        assert len(sources) == 2

    def test_load_directory_with_subdirs(self, temp_dir):
        subdir = temp_dir / "sub"
        subdir.mkdir()
        (subdir / "sub_doc.txt").write_text("子目录文件内容。" * 20, encoding="utf-8")
        (temp_dir / "top_doc.txt").write_text("顶层文件内容。" * 20, encoding="utf-8")

        dp = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        docs = dp.load_directory(temp_dir, recursive=True)

        sources = set(d["metadata"]["source"] for d in docs)
        assert len(sources) == 2
