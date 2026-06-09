"""
完整 RAG 流程

实现标准 RAG 流程：文档检索 -> prompt 构建 -> 模型生成
支持两种模式：with_rag / without_rag（便于对比）
检索结果注入 system prompt。
"""
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# RAG 专用 system prompt 模板
RAG_SYSTEM_TEMPLATE = """你是一个专业的领域知识助手。请根据以下参考信息回答用户的问题。
如果参考信息不足以回答问题，请如实说明，不要编造内容。

## 参考信息
{context}

## 回答要求
- 基于上述参考信息进行回答
- 回答要准确、简洁
- 如果参考信息不足，明确告知用户"""


class RAGPipeline:
    """RAG 检索增强生成流程"""

    def __init__(
        self,
        model_path: str = "outputs/merged_model",
        persist_dir: str = "./chroma_db",
        collection_name: str = "documents",
        top_k: int = 5,
        load_in_4bit: bool = True,
    ):
        """
        初始化 RAG 流程

        参数:
            model_path: 微调后模型路径
            persist_dir: ChromaDB 持久化目录
            collection_name: 向量集合名称
            top_k: 检索返回的文档数量
            load_in_4bit: 是否 4-bit 量化加载
        """
        self.top_k = top_k

        # 加载向量存储
        logger.info(f"加载向量存储: {persist_dir}/{collection_name}")
        self.vector_store = VectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
        logger.info(f"向量存储文档数: {self.vector_store.count()}")

        # 加载模型
        logger.info(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        logger.info(f"模型加载完成，设备: {self.device}")

    def query(
        self,
        question: str,
        use_rag: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        执行 RAG 查询

        参数:
            question: 用户问题
            use_rag: 是否启用 RAG（False 时为普通推理）
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            system_prompt: 自定义 system prompt

        返回:
            {
                "question": str,
                "answer": str,
                "use_rag": bool,
                "contexts": [str, ...],  # 仅 use_rag=True
                "tokens": int,
                "time": float,
            }
        """
        contexts = []
        rag_system_prompt = system_prompt or "你是一个专业的领域知识助手。"

        if use_rag and self.vector_store.count() > 0:
            # Step 1: 检索相关文档
            logger.info(f"检索相关文档 (top_k={self.top_k})...")
            search_results = self.vector_store.similarity_search(question, k=self.top_k)
            contexts = [r["content"] for r in search_results]

            # Step 2: 构建 RAG prompt
            context_text = "\n\n---\n\n".join(
                f"[来源 {i + 1}] {ctx}" for i, ctx in enumerate(contexts)
            )
            rag_system_prompt = RAG_SYSTEM_TEMPLATE.format(context=context_text)

        # Step 3: 构建消息并生成
        messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": question},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start_time

        new_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        logger.info(f"生成完成: {len(new_tokens)} tokens, {elapsed:.2f}s")

        result = {
            "question": question,
            "answer": answer,
            "use_rag": use_rag,
            "tokens": len(new_tokens),
            "time": round(elapsed, 3),
        }
        if use_rag:
            result["contexts"] = contexts

        return result

    def compare(
        self,
        question: str,
        max_new_tokens: int = 256,
    ) -> Dict:
        """
        对比 with_rag vs without_rag 两种模式

        返回:
            {"with_rag": dict, "without_rag": dict}
        """
        logger.info(f"开始对比测试: {question}")

        # 无 RAG 模式
        no_rag = self.query(question, use_rag=False, max_new_tokens=max_new_tokens)

        # 有 RAG 模式
        with_rag = self.query(question, use_rag=True, max_new_tokens=max_new_tokens)

        return {"with_rag": with_rag, "without_rag": no_rag}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 流程")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/merged_model",
        help="模型路径",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./chroma_db",
        help="向量数据库持久化目录",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="检索返回的文档数量",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="什么是深度学习？",
        help="用户问题",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rag", "no_rag", "compare"],
        default="rag",
        help="运行模式: rag/no_rag/compare",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="最大生成 token 数",
    )

    args = parser.parse_args()

    pipeline = RAGPipeline(
        model_path=args.model_path,
        persist_dir=args.persist_dir,
        top_k=args.top_k,
    )

    if args.mode == "compare":
        result = pipeline.compare(args.query, max_new_tokens=args.max_new_tokens)

        print("\n" + "=" * 70)
        print(f" 问题: {args.query}")
        print("=" * 70)

        print(f"\n--- 不使用 RAG ---")
        print(result["without_rag"]["answer"])
        print(f"(tokens: {result['without_rag']['tokens']}, "
              f"time: {result['without_rag']['time']}s)")

        print(f"\n--- 使用 RAG ---")
        print(result["with_rag"]["answer"])
        print(f"(tokens: {result['with_rag']['tokens']}, "
              f"time: {result['with_rag']['time']}s)")

        if "contexts" in result["with_rag"]:
            print(f"\n检索到的上下文 ({len(result['with_rag']['contexts'])} 条):")
            for i, ctx in enumerate(result["with_rag"]["contexts"], 1):
                print(f"  [{i}] {ctx[:100]}...")
    else:
        use_rag = args.mode == "rag"
        result = pipeline.query(args.query, use_rag=use_rag, max_new_tokens=args.max_new_tokens)

        print(f"\n{'='*60}")
        print(f" 模式: {'RAG' if use_rag else '标准推理'}")
        print(f" 问题: {args.query}")
        print(f"{'='*60}")
        print(f"\n回答:\n{result['answer']}")
        print(f"\n{'='*60}")
        print(f" tokens: {result['tokens']}, time: {result['time']}s")

        if "contexts" in result:
            print(f" 检索上下文: {len(result['contexts'])} 条")
