"""
FastAPI 推理服务

提供以下接口：
- GET  /health    : 健康检查
- POST /generate  : 生成接口（支持流式和非流式）
- POST /chat      : 简化的对话接口

为后续 RAG 集成预留 use_rag 参数。
"""
import argparse
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# 允许从 rag/ 目录导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 全局模型和 tokenizer
_model = None
_tokenizer = None
_device = None

# RAG 检索器（可选加载）
_rag_retriever = None

# RAG system prompt 模板
RAG_SYSTEM_TEMPLATE = """你是一个专业的编码助手。请根据以下参考代码信息回答用户的问题。
如果参考信息不足以回答问题，请根据你的知识给出最佳答案，不要编造代码。

## 参考信息
{context}

## 回答要求
- 优先基于上述参考信息中相关代码模式进行回答
- 给出清晰、可运行的代码方案
- 简短解释关键实现思路"""


# =============================================================================
# 请求/响应模型
# =============================================================================

class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: List[Message] = Field(..., description="ChatML 格式消息列表")
    max_new_tokens: int = Field(256, ge=1, le=2048, description="最大生成 token 数")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="采样温度")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="nucleus sampling")
    stream: bool = Field(False, description="是否流式输出")
    use_rag: bool = Field(False, description="是否启用 RAG 检索增强（预留）")


class GenerateResponse(BaseModel):
    text: str
    tokens: int
    time: float


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    system_prompt: str = Field("你是一个专业的领域知识助手。", description="系统提示语")
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = Field(False)
    use_rag: bool = Field(False)


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    gpu_memory_mb: Optional[float] = None
    rag_enabled: bool = False


# =============================================================================
# 应用生命周期
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期管理"""
    # 启动时不需要加载模型（由 main 加载）
    logger.info("FastAPI 推理服务已启动")
    yield
    # 关闭时释放资源
    logger.info("推理服务已停止")


app = FastAPI(
    title="Qwen3 QLoRA 推理服务",
    description="Qwen3-1.7B QLoRA 微调模型推理 API，支持 RAG 检索增强",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# 健康检查
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**2

    return HealthResponse(
        status="healthy" if _model is not None else "no_model",
        model=str(_model.config._name_or_path) if _model is not None else "N/A",
        device=str(_device) if _device is not None else "N/A",
        gpu_memory_mb=round(gpu_mem, 2) if gpu_mem else None,
        rag_enabled=_rag_retriever is not None,
    )


# =============================================================================
# /generate 接口
# =============================================================================

def _do_rag_search(question: str) -> str:
    """检索相关文档，返回格式化后的上下文字符串"""
    global _rag_retriever
    if _rag_retriever is None:
        return ""

    results = _rag_retriever.similarity_search(question, k=5)
    if not results:
        return ""

    contexts = [r["content"] for r in results]
    return "\n\n---\n\n".join(
        f"[参考 {i+1}] (repo: {r['metadata'].get('repo', 'N/A')})\n{r['content'][:800]}"
        for i, r in enumerate(results)
    )


def _do_generate(messages: List[Message], max_new_tokens: int, temperature: float, top_p: float, use_rag: bool = False):
    """同步生成，可选 RAG 增强"""
    global _model, _tokenizer, _device

    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

    # RAG 增强：检索 + 改写 system prompt
    if use_rag and _rag_retriever is not None:
        user_question = msg_dicts[-1]["content"] if msg_dicts else ""
        context = _do_rag_search(user_question)
        if context:
            rag_prompt = RAG_SYSTEM_TEMPLATE.format(context=context)
            # 替换或插入 system message
            if msg_dicts and msg_dicts[0]["role"] == "system":
                msg_dicts[0]["content"] = rag_prompt
            else:
                msg_dicts.insert(0, {"role": "system", "content": rag_prompt})
            logger.info(f"RAG 检索完成，注入 {len(context)} 字符上下文")

    text = _tokenizer.apply_chat_template(
        msg_dicts, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(text, return_tensors="pt").to(_device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=_tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    new_tokens = outputs[0][input_len:]
    generated_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generated_text, len(new_tokens), elapsed


async def _stream_generate(messages: List[Message], max_new_tokens: int, temperature: float, top_p: float, use_rag: bool = False):
    """流式生成（SSE）"""
    global _model, _tokenizer, _device

    generated_text, tokens, elapsed = _do_generate(
        messages, max_new_tokens, temperature, top_p, use_rag=use_rag
    )

    # 按句子分块发送（模拟流式效果）
    import re
    sentences = re.split(r"([。！？\n])", generated_text)

    current = ""
    for i, part in enumerate(sentences):
        current += part
        # 每遇到标点或每 20 字符发送一次
        if part in ("。", "！", "？", "\n") or len(current) >= 20:
            yield f"data: {current}\n\n"
            current = ""
            import asyncio
            await asyncio.sleep(0.05)

    if current:
        yield f"data: {current}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/generate", response_model=None)
async def generate(request: GenerateRequest):
    """生成接口 - 支持 RAG 检索增强"""
    if _model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    if request.use_rag and _rag_retriever is None:
        logger.warning("RAG 模式已请求，但服务未启用 RAG（启动时需 --enable_rag）")

    if request.stream:
        return StreamingResponse(
            _stream_generate(
                request.messages,
                request.max_new_tokens,
                request.temperature,
                request.top_p,
                use_rag=request.use_rag,
            ),
            media_type="text/event-stream",
        )
    else:
        text, tokens, elapsed = _do_generate(
            request.messages,
            request.max_new_tokens,
            request.temperature,
            request.top_p,
            use_rag=request.use_rag,
        )
        return GenerateResponse(text=text, tokens=tokens, time=round(elapsed, 3))


@app.post("/chat", response_model=GenerateResponse)
async def chat(request: ChatRequest):
    """简化的对话接口"""
    messages = [
        Message(role="system", content=request.system_prompt),
        Message(role="user", content=request.query),
    ]
    gen_request = GenerateRequest(
        messages=messages,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        stream=False,
        use_rag=request.use_rag,
    )
    return await generate(gen_request)


# =============================================================================
# 启动入口
# =============================================================================

def load_model(model_path: str, load_in_4bit: bool = True):
    """预加载模型到全局变量"""
    global _model, _tokenizer, _device

    logger.info(f"加载模型: {model_path}")

    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}

    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        _device = "cuda"
    else:
        _device = "cpu"

    _model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    _model.eval()

    logger.info(f"模型加载完成，设备: {_device}")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Qwen3 推理服务 (支持 RAG)")
    parser.add_argument("--model_path", type=str, default="outputs/outputs_codealpacas/merged_model",
                        help="模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="监听地址")
    parser.add_argument("--port", type=int, default=8000,
                        help="监听端口")
    parser.add_argument("--no_4bit", action="store_true",
                        help="不使用 4-bit 量化")

    # RAG 参数
    parser.add_argument("--enable_rag", action="store_true",
                        help="启用 RAG 检索增强")
    parser.add_argument("--rag_retriever", type=str, default="bm25",
                        choices=["bm25", "chromadb"],
                        help="检索器类型 (默认: bm25)")
    parser.add_argument("--rag_persist_dir", type=str, default="bm25_index_swebench",
                        help="检索器索引目录")
    parser.add_argument("--rag_collection", type=str, default="swebench_instances",
                        help="检索器集合名称")
    parser.add_argument("--rag_top_k", type=int, default=5,
                        help="检索返回数量")

    args = parser.parse_args()

    # 预加载模型
    load_model(args.model_path, load_in_4bit=not args.no_4bit)

    # 加载 RAG 检索器
    if args.enable_rag:
        if args.rag_retriever == "bm25":
            from bm25_store import BM25Store
            _rag_retriever = BM25Store(
                persist_directory=args.rag_persist_dir,
                collection_name=args.rag_collection,
            )
        elif args.rag_retriever == "chromadb":
            from vector_store import VectorStore
            _rag_retriever = VectorStore(
                persist_directory=args.rag_persist_dir,
                collection_name=args.rag_collection,
            )
        logger.info(f"RAG 已启用: {args.rag_retriever}, "
                     f"文档数={_rag_retriever.count()}, top_k={args.rag_top_k}")

    logger.info(f"启动推理服务: http://{args.host}:{args.port}")
    logger.info(f"API 文档: http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
