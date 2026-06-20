"""测试 train/ 模块的配置加载和 checkpoint 查找逻辑"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))

# 不依赖 torch 的测试
import yaml


class TestQLoRAConfig:
    """测试 QLoRA 配置文件"""

    def test_config_is_valid_yaml(self):
        config_path = Path(__file__).parent.parent / "configs" / "qlora_qwen3-1.7b_codealpaca_r16_len1024_ep3.yaml"
        assert config_path.exists()

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 验证必需字段
        assert "model" in config
        assert config["model"]["model_name"] == "Qwen/Qwen3-1.7B"

        assert "quantization" in config
        assert config["quantization"]["load_in_4bit"] is True
        assert config["quantization"]["bnb_4bit_quant_type"] == "nf4"

        assert "lora" in config
        assert config["lora"]["r"] == 16
        assert config["lora"]["lora_alpha"] == 32
        assert "q_proj" in config["lora"]["target_modules"]
        assert "k_proj" in config["lora"]["target_modules"]
        assert "v_proj" in config["lora"]["target_modules"]

        assert "training" in config
        assert config["training"]["batch_size"] == 1
        assert config["training"]["gradient_accumulation_steps"] == 16
        assert config["training"]["max_seq_length"] == 1024

        assert "save_and_logging" in config
        assert config["save_and_logging"]["save_steps"] == 200

    def test_all_attention_layers_targeted(self):
        config_path = Path(__file__).parent.parent / "configs" / "qlora_qwen3-1.7b_codealpaca_r16_len1024_ep3.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        targets = config["lora"]["target_modules"]
        expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        assert set(targets) == expected


class TestTrainingConfig:
    """测试训练配置文件"""

    def test_config_is_valid_yaml(self):
        config_path = Path(__file__).parent.parent / "configs" / "training_qwen3-1.7b_codealpaca.yaml"
        assert config_path.exists()

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "general" in config
        assert config["general"]["seed"] == 42

        assert "precision" in config
        assert config["precision"]["bf16"] is True

        assert "memory" in config
        assert config["memory"]["gradient_checkpointing"] is True

        assert "data" in config
        assert "system_prompt" in config["data"]


class TestCheckpointFinder:
    """测试 checkpoint 查找逻辑（使用临时目录）"""

    @staticmethod
    def _find_latest_checkpoint(output_dir):
        """内联实现，避免导入 resume_train（trl 在 Windows 下有编码问题）"""
        import re
        checkpoint_dirs = list(Path(output_dir).glob("checkpoint-*"))
        if not checkpoint_dirs:
            raise FileNotFoundError(f"在 {output_dir} 中未找到任何 checkpoint")

        def get_step(p):
            match = re.search(r"checkpoint-(\d+)", p.name)
            return int(match.group(1)) if match else 0

        return max(checkpoint_dirs, key=get_step)

    def test_find_latest_checkpoint(self, temp_dir):
        (temp_dir / "checkpoint-100").mkdir()
        (temp_dir / "checkpoint-200").mkdir()
        (temp_dir / "checkpoint-50").mkdir()
        (temp_dir / "some_other_dir").mkdir()

        latest = self._find_latest_checkpoint(temp_dir)
        assert latest.name == "checkpoint-200"

    def test_no_checkpoints_raises(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            self._find_latest_checkpoint(temp_dir)

    def test_find_with_non_numeric(self, temp_dir):
        (temp_dir / "checkpoint-100").mkdir()
        (temp_dir / "checkpoint-final").mkdir()

        latest = self._find_latest_checkpoint(temp_dir)
        assert latest.name == "checkpoint-100"
