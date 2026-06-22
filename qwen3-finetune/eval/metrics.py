"""
评估指标计算模块

支持指标：
1. Perplexity (PPL)
2. ROUGE-1/2/L
3. BLEU-4
4. 领域准确率（关键词匹配 + 语义相似度）
5. 平均生成长度
6. 推理速度（tokens/sec）
"""
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


# =============================================================================
# Perplexity
# =============================================================================

def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 1,
) -> float:
    """
    计算模型在给定文本上的 Perplexity

    参数:
        model: 语言模型
        tokenizer: 分词器
        texts: 文本列表
        max_length: 最大序列长度
        batch_size: 批次大小（建议 1，避免 OOM）

    返回:
        Perplexity 值
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, text in enumerate(texts):
            # 截断文本
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            input_ids = encodings["input_ids"].to(device)
            labels = input_ids.clone()

            # 计算 loss
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            num_tokens = input_ids.shape[1]
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if (i + 1) % 50 == 0:
                logger.info(f"PPL 计算进度: {i + 1}/{len(texts)}")

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


# =============================================================================
# ROUGE
# =============================================================================

def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    计算 ROUGE-1/2/L 的 Precision, Recall, F1

    返回: 含 P/R/F1 三元组，同时保留 rouge1/rouge2/rougeL 作为 F1 别名（向下兼容）
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) 和 references ({len(references)}) 长度不一致"
        )

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
    )

    accum = {}
    for key in ["rouge1", "rouge2", "rougeL"]:
        for suffix in ["_p", "_r", "_f"]:
            accum[key + suffix] = 0.0

    count = len(predictions)
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in ["rouge1", "rouge2", "rougeL"]:
            accum[key + "_p"] += result[key].precision
            accum[key + "_r"] += result[key].recall
            accum[key + "_f"] += result[key].fmeasure

    scores = {}
    for key in ["rouge1", "rouge2", "rougeL"]:
        scores[key + "_p"] = round(accum[key + "_p"] / max(count, 1), 4)
        scores[key + "_r"] = round(accum[key + "_r"] / max(count, 1), 4)
        scores[key + "_f"] = round(accum[key + "_f"] / max(count, 1), 4)
        # 向下兼容：rouge1/rouge2/rougeL 等价于 F1
        scores[key] = scores[key + "_f"]

    return scores


# =============================================================================
# BLEU-4
# =============================================================================

def _tokenize_chinese(text: str) -> List[str]:
    """简单的中文分词（按字符分割，英文按空格分割）"""
    # 将中文字符逐个分割，英文单词保持完整
    tokens = []
    current_word = ""
    for char in text:
        if "一" <= char <= "鿿":
            if current_word:
                tokens.append(current_word)
                current_word = ""
            tokens.append(char)
        elif char.isspace():
            if current_word:
                tokens.append(current_word)
                current_word = ""
        else:
            current_word += char
    if current_word:
        tokens.append(current_word)
    return tokens


def compute_bleu(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    计算 BLEU-4 分数

    参数:
        predictions: 生成的文本列表
        references: 参考文本列表

    返回:
        BLEU-4 分数（0-1）
    """
    if len(predictions) != len(references):
        raise ValueError("predictions 和 references 长度不一致")

    smoother = SmoothingFunction().method1
    total_score = 0.0
    count = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize_chinese(pred)
        ref_tokens = _tokenize_chinese(ref)

        if len(pred_tokens) < 4 or len(ref_tokens) < 4:
            continue

        score = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoother,
        )
        total_score += score
        count += 1

    return round(total_score / max(count, 1), 4)


# =============================================================================
# BERTScore
# =============================================================================

_bertscore_model = None
_bertscore_tokenizer = None


def _get_bertscore_model():
    global _bertscore_model, _bertscore_tokenizer
    if _bertscore_model is None:
        from transformers import AutoModel, AutoTokenizer

        _bertscore_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        _bertscore_model = AutoModel.from_pretrained("bert-base-chinese")
        _bertscore_model.eval()
        # BERTScore 在 CPU 上计算，不与 LLM 争抢 GPU
    return _bertscore_model, _bertscore_tokenizer


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    计算 BERTScore F1（基于 bert-base-chinese）

    返回: {"bertscore_f1": float}
    """
    import torch

    model, tokenizer = _get_bertscore_model()
    # 始终 CPU，避免 GPU OOM（8GB 显存已被 LLM 占用）
    device = torch.device("cpu")
    model = model.to(device)

    all_precision = 0.0
    all_recall = 0.0
    count = 0

    for start in range(0, len(predictions), batch_size):
        batch_preds = predictions[start : start + batch_size]
        batch_refs = references[start : start + batch_size]

        for pred, ref in zip(batch_preds, batch_refs):
            if not pred.strip() or not ref.strip():
                all_precision += 0.0
                all_recall += 0.0
                count += 1
                continue

            pred_ids = tokenizer(pred, return_tensors="pt", truncation=True, max_length=128).to(device)
            ref_ids = tokenizer(ref, return_tensors="pt", truncation=True, max_length=128).to(device)

            with torch.no_grad():
                pred_emb = model(**pred_ids).last_hidden_state[0]  # (L1, D)
                ref_emb = model(**ref_ids).last_hidden_state[0]    # (L2, D)

            # Normalize
            pred_emb = pred_emb / (pred_emb.norm(dim=-1, keepdim=True) + 1e-12)
            ref_emb = ref_emb / (ref_emb.norm(dim=-1, keepdim=True) + 1e-12)

            # Cosine similarity matrix
            sim = torch.matmul(pred_emb, ref_emb.T)  # (L1, L2)

            # Precision: for each pred token, max sim to any ref token
            precision = sim.max(dim=-1).values.mean().item()
            # Recall: for each ref token, max sim to any pred token
            recall = sim.max(dim=0).values.mean().item()

            all_precision += max(precision, 0.0)
            all_recall += max(recall, 0.0)
            count += 1

    if count == 0:
        return {"bertscore_f1": 0.0}

    avg_precision = all_precision / count
    avg_recall = all_recall / count
    f1 = 2 * avg_precision * avg_recall / max(avg_precision + avg_recall, 1e-12)

    return {"bertscore_f1": round(f1, 4)}


# =============================================================================
# 领域准确率
# =============================================================================

def compute_domain_accuracy(
    predictions: List[str],
    references: List[str],
    keywords: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    计算领域准确率（基于关键词匹配和语义相似度）

    参数:
        predictions: 生成的文本列表
        references: 参考文本列表
        keywords: 领域关键词列表（可选）

    返回:
        {"keyword_accuracy": float, "overlap_accuracy": float}
    """
    keyword_hits = 0
    overlap_total = 0.0
    count = len(predictions)

    for pred, ref in zip(predictions, references):
        pred_lower = pred.lower()
        ref_lower = ref.lower()

        # 1. 关键词匹配准确率
        if keywords:
            ref_keywords = [kw for kw in keywords if kw.lower() in ref_lower]
            if ref_keywords:
                matched = sum(1 for kw in ref_keywords if kw.lower() in pred_lower)
                keyword_hits += 1 if matched > 0 else 0

        # 2. 参考文本关键内容重叠率
        ref_chars = set(ref_lower.replace(" ", ""))
        pred_chars = set(pred_lower.replace(" ", ""))
        if ref_chars:
            overlap = len(ref_chars & pred_chars) / len(ref_chars)
            overlap_total += overlap

    result = {
        "overlap_accuracy": round(overlap_total / max(count, 1), 4),
    }

    if keywords:
        result["keyword_accuracy"] = round(keyword_hits / max(count, 1), 4)
    else:
        result["keyword_accuracy"] = result["overlap_accuracy"]

    return result


# =============================================================================
# 生成长度与速度
# =============================================================================

def compute_generation_stats(
    predictions: List[str],
    generation_times: List[float],
    token_counts: List[int],
) -> Dict[str, float]:
    """
    计算生成统计信息

    参数:
        predictions: 生成的文本列表
        generation_times: 每次生成的耗时（秒）
        token_counts: 每次生成的 token 数量

    返回:
        {"avg_length": float, "tokens_per_sec": float}
    """
    # 平均生成长度（字符数）
    char_lengths = [len(p) for p in predictions]
    avg_length = sum(char_lengths) / max(len(char_lengths), 1)

    # 推理速度（tokens/sec）
    total_tokens = sum(token_counts)
    total_time = sum(generation_times)
    tokens_per_sec = total_tokens / max(total_time, 0.001)

    return {
        "avg_length": round(avg_length, 1),
        "avg_tokens": round(sum(token_counts) / max(len(token_counts), 1), 1),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }


# =============================================================================
# 汇总所有指标
# =============================================================================

def compute_all_metrics(
    model,
    tokenizer,
    test_texts: List[str],
    predictions: List[str],
    references: List[str],
    generation_times: List[float],
    token_counts: List[int],
    keywords: Optional[List[str]] = None,
    max_length: int = 512,
) -> Dict:
    """
    计算所有评估指标

    返回:
        完整的指标字典
    """
    logger.info("计算 Perplexity...")
    ppl = compute_perplexity(model, tokenizer, test_texts, max_length)

    logger.info("计算 ROUGE...")
    rouge = compute_rouge(predictions, references)

    logger.info("计算 BLEU-4...")
    bleu = compute_bleu(predictions, references)

    logger.info("计算领域准确率...")
    domain_acc = compute_domain_accuracy(predictions, references, keywords)

    logger.info("计算生成统计...")
    gen_stats = compute_generation_stats(predictions, generation_times, token_counts)

    return {
        "perplexity": round(ppl, 2),
        **rouge,
        "bleu4": bleu,
        **domain_acc,
        **gen_stats,
    }
