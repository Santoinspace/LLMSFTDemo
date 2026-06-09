"""
Embedding 生成模块

使用 BAAI/bge-m3 模型生成文本嵌入向量。
BGE-M3 支持多语言、多粒度的文本表示，向量维度 1024。
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 默认 embedding 模型
DEFAULT_MODEL = "BAAI/bge-m3"


class EmbeddingGenerator:
    """文本嵌入生成器"""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        """
        初始化 embedding 模型

        参数:
            model_name: HuggingFace 模型名称
            device: 设备（cuda/cpu），默认自动选择
        """
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"加载 embedding 模型: {model_name} (设备: {device})")

        # 延迟导入，避免必装依赖
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        self.dimension = self.model.config.hidden_size
        logger.info(f"Embedding 维度: {self.dimension}")
        logger.info(f"模型加载完成")

    def embed(
        self,
        texts: List[str],
        batch_size: int = 8,
        normalize: bool = True,
        instruction: str = "",
    ) -> List[List[float]]:
        """
        批量生成文本嵌入

        参数:
            texts: 文本列表
            batch_size: 批次大小
            normalize: 是否 L2 归一化
            instruction: BGE 模型的指令前缀（为空则自动处理）

        返回:
            嵌入向量列表，shape: [len(texts), 1024]
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # BGE 模型支持为 query 添加指令前缀
            if instruction:
                batch_texts = [instruction + t for t in batch_texts]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # 取 [CLS] token 的表示（或 mean pooling）
                embeddings = self._mean_pooling(outputs, inputs["attention_mask"])

            # L2 归一化
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().tolist())

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Embedding 进度: {min(i + batch_size, len(texts))}/{len(texts)}")

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """生成查询嵌入（添加 BGE 查询指令）"""
        instruction = "为这个句子生成表示以用于检索相关文章："
        return self.embed([query], instruction=instruction, normalize=True)[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """生成文档嵌入"""
        return self.embed(documents, normalize=True)

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """Mean Pooling - 考虑 attention mask"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


def embed_file(
    input_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    text_field: str = "content",
) -> None:
    """
    对 jsonl 文件中的文本生成嵌入并保存

    参数:
        input_path: 输入文件路径（jsonl）
        output_dir: 输出目录
        model_name: embedding 模型名称
        text_field: 文本字段名
    """
    logger.info(f"读取文件: {input_path}")

    texts = []
    metadatas = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get(text_field, "")
                if text:
                    texts.append(text)
                    metadatas.append({k: v for k, v in item.items() if k != text_field})
            except json.JSONDecodeError:
                continue

    logger.info(f"读取到 {len(texts)} 条文本")

    # 生成嵌入
    generator = EmbeddingGenerator(model_name=model_name)
    embeddings = generator.embed_documents(texts)

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_embeddings.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for text, meta, emb in zip(texts, metadatas, embeddings):
            f.write(json.dumps({
                "text": text,
                "metadata": meta,
                "embedding": emb,
            }, ensure_ascii=False) + "\n")

    logger.info(f"嵌入已保存: {output_path} ({len(embeddings)} 条)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文本嵌入生成")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件路径（jsonl）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rag_db",
        help="输出目录",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="Embedding 模型名称",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="content",
        help="文本字段名",
    )

    args = parser.parse_args()
    embed_file(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        text_field=args.text_field,
    )
