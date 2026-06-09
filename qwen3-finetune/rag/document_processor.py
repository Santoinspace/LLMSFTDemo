"""
文档处理模块

支持多种格式文档的加载、分块和预处理。
- chunk_size=512, chunk_overlap=64
- 支持格式: txt, md, jsonl, pdf（可选）
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档加载与分块处理器"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
    ):
        """
        初始化文档处理器

        参数:
            chunk_size: 分块大小（字符数）
            chunk_overlap: 块间重叠大小
            separators: 分隔符优先级列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    def load_file(self, file_path: Path) -> List[Dict]:
        """
        加载单个文件

        返回:
            [{"content": str, "metadata": dict}, ...]
        """
        suffix = file_path.suffix.lower()

        loaders = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".jsonl": self._load_jsonl,
            ".json": self._load_json,
        }

        if suffix in loaders:
            return loaders[suffix](file_path)
        else:
            logger.warning(f"不支持的文件格式: {suffix}，尝试作为文本文件加载")
            return self._load_text(file_path)

    def load_directory(self, dir_path: Path, recursive: bool = True) -> List[Dict]:
        """加载目录中的所有支持文件"""
        all_docs = []
        pattern = "**/*" if recursive else "*"

        supported_extensions = {".txt", ".md", ".jsonl", ".json"}
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                docs = self.load_file(file_path)
                all_docs.extend(docs)
                logger.info(f"已加载: {file_path} ({len(docs)} 块)")

        logger.info(f"总计加载: {len(all_docs)} 个文档块")
        return all_docs

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        将文本按指定大小分块

        参数:
            text: 原始文本
            metadata: 文档元数据

        返回:
            [{"content": str, "metadata": dict}, ...]
        """
        if metadata is None:
            metadata = {}

        chunks = self._split_text(text)
        return [
            {
                "content": chunk.strip(),
                "metadata": {**metadata, "chunk_index": i, "chunk_size": len(chunk)},
            }
            for i, chunk in enumerate(chunks)
            if chunk.strip()
        ]

    def _split_text(self, text: str) -> List[str]:
        """递归文本分割"""
        # 如果文本已经在 chunk_size 内，直接返回
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 尝试用分隔符分割
        for separator in self.separators:
            if separator == "":
                # 按固定大小强切
                return self._split_by_size(text)

            splits = text.split(separator)
            if len(splits) <= 1:
                continue

            # 尝试用当前分隔符合并小块
            chunks = []
            current = ""
            for i, part in enumerate(splits):
                candidate = current + (separator if current else "") + part

                if len(candidate) > self.chunk_size and current:
                    chunks.append(current)
                    # 重叠：保留 current 的最后 chunk_overlap 字符作为新块的开头
                    if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                        current = current[-self.chunk_overlap:] + separator + part
                    else:
                        current = part
                else:
                    current = candidate

            if current.strip():
                chunks.append(current)

            if len(chunks) > 1:
                return chunks

        # 兜底：按固定大小强切
        return self._split_by_size(text)

    def _split_by_size(self, text: str) -> List[str]:
        """按固定 chunk_size 强制切割"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            # 已到达文本末尾，退出循环
            if end >= len(text):
                break
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
            # 防止无限循环：start 必须前进
            if start <= 0:
                start = end
        return chunks

    # ====== 文件加载器 ======

    def _load_text(self, file_path: Path) -> List[Dict]:
        """加载纯文本文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "filetype": file_path.suffix,
        }
        return self.chunk_text(text, metadata)

    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """加载 JSONL 文件"""
        chunks = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # 尝试多个常见字段名
                    text = item.get("text") or item.get("content") or item.get("body") or json.dumps(item, ensure_ascii=False)
                    metadata = {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "filetype": file_path.suffix,
                        "line": line_num,
                        **{k: v for k, v in item.items() if k not in ("text", "content", "body") and not isinstance(v, (dict, list))},
                    }
                    chunks.extend(self.chunk_text(text, metadata))
                except json.JSONDecodeError:
                    logger.warning(f"第 {line_num} 行 JSON 解析失败")
        return chunks

    def _load_json(self, file_path: Path) -> List[Dict]:
        """加载 JSON 文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 尝试提取文本内容
        if isinstance(data, dict):
            text = data.get("text") or data.get("content") or json.dumps(data, ensure_ascii=False)
        elif isinstance(data, list):
            text = "\n".join(
                item.get("text") or item.get("content") or json.dumps(item, ensure_ascii=False)
                for item in data
            )
        else:
            text = str(data)

        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "filetype": file_path.suffix,
        }
        return self.chunk_text(text, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文档处理器")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件或目录路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed_docs.jsonl",
        help="输出文件路径",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="分块大小（字符数）",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=64,
        help="块间重叠大小",
    )

    args = parser.parse_args()

    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    input_path = Path(args.input)

    # 加载文档
    if input_path.is_file():
        chunks = processor.load_file(input_path)
    elif input_path.is_dir():
        chunks = processor.load_directory(input_path)
    else:
        logger.error(f"路径不存在: {input_path}")
        exit(1)

    logger.info(f"共生成 {len(chunks)} 个文档块")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # 打印统计
    sizes = [c["chunk_size"] for c in chunks]
    if sizes:
        print(f"\n{'='*50}")
        print(f" 文档处理统计")
        print(f"{'='*50}")
        print(f"  总块数: {len(chunks)}")
        print(f"  平均块大小: {sum(sizes) / len(sizes):.0f} 字符")
        print(f"  最小块: {min(sizes)} 字符")
        print(f"  最大块: {max(sizes)} 字符")
        print(f"  输出文件: {output_path}")
