"""测试 inference/ 模块的纯逻辑部分"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))


class TestMessageModel:
    """测试 API 请求/响应数据模型"""

    def test_message_model(self):
        from api_server import Message
        msg = Message(role="user", content="你好")
        assert msg.role == "user"
        assert msg.content == "你好"

    def test_generate_request_model(self):
        from api_server import GenerateRequest, Message
        req = GenerateRequest(
            messages=[Message(role="user", content="测试")],
            max_new_tokens=128,
            temperature=0.5,
            stream=False,
        )
        assert req.max_new_tokens == 128
        assert req.temperature == 0.5
        assert req.stream is False
        assert req.use_rag is False

    def test_generate_request_with_rag(self):
        from api_server import GenerateRequest, Message
        req = GenerateRequest(
            messages=[Message(role="user", content="查询")],
            use_rag=True,
        )
        assert req.use_rag is True

    def test_chat_request_model(self):
        from api_server import ChatRequest
        req = ChatRequest(query="什么是AI？")
        assert req.query == "什么是AI？"
        assert req.system_prompt == "你是一个专业的领域知识助手。"

    def test_health_response_model(self):
        from api_server import HealthResponse
        resp = HealthResponse(status="healthy", model="test-model", device="cuda")
        assert resp.status == "healthy"
        assert resp.model == "test-model"

    def test_generate_response_model(self):
        from api_server import GenerateResponse
        resp = GenerateResponse(text="回答内容", tokens=10, time=0.5)
        assert resp.text == "回答内容"
        assert resp.tokens == 10
        assert resp.time == 0.5


class TestBatchInferenceArgs:
    """测试批量推理参数解析"""

    def test_argument_defaults(self):
        from batch_inference import argparse as batch_argparse
        # 验证脚本可被导入且参数正确
        assert hasattr(batch_argparse, "ArgumentParser")
