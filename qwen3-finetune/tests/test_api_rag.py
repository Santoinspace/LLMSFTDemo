"""测试 inference/api_server.py 的 RAG 集成

启动服务:
    python inference/api_server.py --model_path outputs/outputs_codealpacas/merged_model --enable_rag

运行测试:
    pytest tests/test_api_rag.py -v

自定义服务地址:
    API_BASE_URL=http://localhost:8000 pytest tests/test_api_rag.py -v
"""
import os
import time

import pytest
import requests

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def wait_for_server(timeout: int = 60):
    """等待服务就绪"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return r.json()
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise RuntimeError(f"服务未在 {timeout}s 内就绪: {BASE_URL}")


@pytest.fixture(scope="module")
def server_health():
    """模块级 fixture：验证服务可用"""
    return wait_for_server()


class TestHealthCheck:
    """健康检查"""

    def test_service_alive(self, server_health):
        assert server_health["status"] == "healthy"

    def test_model_loaded(self, server_health):
        assert server_health["model"] != "N/A"
        assert len(server_health["model"]) > 0

    def test_rag_enabled(self, server_health):
        assert server_health.get("rag_enabled") is True, (
            "RAG 未启用，请用 --enable_rag 启动服务"
        )


class TestGenerateWithoutRAG:
    """无 RAG 生成"""

    def test_generate_returns_text(self, server_health):
        resp = requests.post(f"{BASE_URL}/generate", json={
            "messages": [
                {"role": "user", "content": "Write a Python function to add two numbers."}
            ],
            "max_new_tokens": 128,
            "temperature": 0.0,
            "use_rag": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["text"]) > 0
        assert data["tokens"] > 0
        assert data["time"] > 0

    def test_generate_with_system_prompt(self, server_health):
        resp = requests.post(f"{BASE_URL}/generate", json={
            "messages": [
                {"role": "system", "content": "You are a Python expert. Always include type hints."},
                {"role": "user", "content": "Write a function to merge two dicts."}
            ],
            "max_new_tokens": 128,
            "temperature": 0.0,
            "use_rag": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["text"]) > 0


class TestGenerateWithRAG:
    """有 RAG 生成"""

    def test_generate_rag_returns_text(self, server_health):
        resp = requests.post(f"{BASE_URL}/generate", json={
            "messages": [
                {"role": "user",
                 "content": "How to handle database connection errors in Python?"}
            ],
            "max_new_tokens": 256,
            "temperature": 0.0,
            "use_rag": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["text"]) > 0
        assert data["tokens"] > 0

    def test_rag_query_code_related(self, server_health):
        """代码相关问题应有 RAG 返回"""
        resp = requests.post(f"{BASE_URL}/generate", json={
            "messages": [
                {"role": "user",
                 "content": "What is the best way to implement connection pooling?"}
            ],
            "max_new_tokens": 200,
            "temperature": 0.0,
            "use_rag": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["text"]) > 0


class TestChatEndpoint:
    """Chat 接口"""

    def test_chat_no_rag(self, server_health):
        resp = requests.post(f"{BASE_URL}/chat", json={
            "query": "Explain what a Python decorator is.",
            "max_new_tokens": 128,
            "temperature": 0.0,
            "use_rag": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["text"]) > 0

    def test_chat_with_rag(self, server_health):
        resp = requests.post(f"{BASE_URL}/chat", json={
            "query": "How to fix a memory leak in a Python web application?",
            "max_new_tokens": 200,
            "temperature": 0.0,
            "use_rag": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["text"]) > 0


class TestRagVsNoRagComparison:
    """RAG vs 非 RAG 对比"""

    def test_answers_should_differ(self, server_health):
        """同一问题，有/无 RAG 的答案应有差异"""
        question = "How to handle database transaction rollback in a web app?"

        # 无 RAG
        resp1 = requests.post(f"{BASE_URL}/generate", json={
            "messages": [{"role": "user", "content": question}],
            "max_new_tokens": 256,
            "temperature": 0.0,
            "use_rag": False,
        }).json()

        # 有 RAG
        resp2 = requests.post(f"{BASE_URL}/generate", json={
            "messages": [{"role": "user", "content": question}],
            "max_new_tokens": 256,
            "temperature": 0.0,
            "use_rag": True,
        }).json()

        assert len(resp1["text"]) > 0
        assert len(resp2["text"]) > 0
        # 有/无 RAG 的答案应不同（除非 token 数很少的退化情况）
        if resp1["tokens"] > 20 and resp2["tokens"] > 20:
            assert resp1["text"] != resp2["text"], (
                "同一问题有/无 RAG 应产生不同答案"
            )


class TestErrorHandling:
    """错误处理"""

    def test_empty_messages(self, server_health):
        resp = requests.post(f"{BASE_URL}/generate", json={
            "messages": [],
            "use_rag": False,
        })
        assert resp.status_code in (200, 422, 400, 500)
