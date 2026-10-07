# -*- coding: utf-8 -*-
"""GujinBridge 一键体验脚本。

默认加载仓库中当前训练好的 Base + LoRA，先运行三个能力示例，再进入交互模式。

Examples:
    python demo/try_gujinbridge.py
    python demo/try_gujinbridge.py --no-interactive
    python demo/try_gujinbridge.py --local-files-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = PROJECT_ROOT / "outputs-gujinbridge-v5-merged"
# DPO V1 is the selected release adapter after the frozen-set SFT/DPO/GRPO comparison.
DEFAULT_LORA_MODEL = PROJECT_ROOT / "outputs-gujinbridge-dpo-v1-qwen3-1.7b"
DEFAULT_SYSTEM_PROMPT = PROJECT_ROOT / "configs" / "gujinbridge_system_prompt.txt"

SHOWCASE_CASES = (
    (
        "古文译现代汉语",
        "将下列古文翻译成现代汉语：\n\n"
        "臣本布衣，躬耕于南阳，苟全性命于乱世，不求闻达于诸侯。",
    ),
    (
        "现代汉语译古文",
        "将下列现代汉语改写为简洁文言文：\n\n"
        "学习贵在长期坚持，即使遇到困难，也不应该半途而废。",
    ),
    (
        "古文断句与标点",
        "为下列古文断句并添加现代标点：\n\n"
        "故天将降大任于是人也必先苦其心志劳其筋骨饿其体肤",
    ),
)

COMMAND_PREFIXES = {
    "/c2m": "将下列古文翻译成现代汉语：\n\n",
    "/m2c": "将下列现代汉语改写为简洁文言文：\n\n",
    "/punct": "为下列古文断句并添加现代标点：\n\n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键查看当前 GujinBridge 模型的古今互译和断句能力。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        default=str(DEFAULT_BASE_MODEL),
        help="基础模型路径或 Hugging Face 名称；默认使用已合并的 V5 SFT 模型。",
    )
    parser.add_argument(
        "--lora-model",
        default=str(DEFAULT_LORA_MODEL),
        help="LoRA 适配器目录；传空字符串可只运行基础模型。",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=str(DEFAULT_SYSTEM_PROMPT),
        help="系统提示词 UTF-8 文件。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="单次最多生成 token 数。")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 表示稳定的贪心生成；大于 0 时启用采样。",
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="采样时的 top-p。")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="重复惩罚。")
    parser.add_argument("--load-in-4bit", action="store_true", help="以 4-bit 加载基础模型（需要 bitsandbytes）。")
    parser.add_argument("--local-files-only", action="store_true", help="只使用本地缓存，不访问网络。")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU（速度会明显变慢）。")
    parser.add_argument("--no-showcase", action="store_true", help="跳过启动时的三个内置演示。")
    parser.add_argument("--no-interactive", action="store_true", help="演示完成后直接退出。")
    return parser.parse_args()


def read_base_model_from_adapter(lora_model: Path) -> str | None:
    """从 PEFT adapter_config.json 读取训练时使用的基础模型。"""
    config_path = lora_model / "adapter_config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8-sig") as file:
        config = json.load(file)
    value = config.get("base_model_name_or_path")
    return str(value).strip() if value else None


def load_system_prompt(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"系统提示词文件不存在：{path}")
    return path.read_text(encoding="utf-8-sig").strip()


def load_model(args: argparse.Namespace):
    lora_path = Path(args.lora_model).expanduser().resolve() if args.lora_model else None
    if lora_path and not lora_path.is_dir():
        raise FileNotFoundError(f"LoRA 目录不存在：{lora_path}")

    base_model_name = args.base_model
    if not base_model_name and lora_path:
        base_model_name = read_base_model_from_adapter(lora_path)
    if not base_model_name:
        raise ValueError("无法确定基础模型，请通过 --base-model 指定。")

    device_map = "cpu" if args.cpu else "auto"
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32 if args.cpu else "auto",
        "low_cpu_mem_usage": True,
        "device_map": device_map,
        "local_files_only": args.local_files_only,
    }
    if args.load_in_4bit:
        if args.cpu:
            raise ValueError("--load-in-4bit 不能与 --cpu 同时使用。")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    print("\n正在加载 GujinBridge，请稍候……")
    print(f"  Base：{base_model_name}")
    print(f"  LoRA：{lora_path if lora_path else '未使用'}")
    # SFT 输出目录会保存训练时实际使用的 tokenizer 和 chat_template.jinja。
    # 优先从 LoRA 目录加载，避免基础模型缓存缺少 chat template。
    tokenizer_source = str(lora_path) if lora_path and (lora_path / "tokenizer_config.json").is_file() else base_model_name
    print(f"  Tokenizer：{tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    if lora_path:
        model = PeftModel.from_pretrained(base_model, str(lora_path), device_map=device_map)
    else:
        model = base_model
    model.eval()
    if args.temperature == 0:
        # 部分 Qwen generation_config 自带采样参数；贪心生成时清空它们，
        # 避免 Transformers 输出“采样参数会被忽略”的无关警告。
        for name in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, name):
                setattr(model.generation_config, name, None)

    device = model.get_input_embeddings().weight.device
    print(f"  设备：{device}")
    print("模型加载完成。\n")
    return model, tokenizer, device, base_model_name, lora_path


def render_chat_prompt(tokenizer, system_prompt: str, user_prompt: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # 兼容没有 enable_thinking 参数的 tokenizer。
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.inference_mode()
def generate_answer(model, tokenizer, device, system_prompt: str, user_prompt: str, args) -> tuple[str, float]:
    prompt = render_chat_prompt(tokenizer, system_prompt, user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    started_at = time.perf_counter()
    output = model.generate(**inputs, **generation_kwargs)
    elapsed = time.perf_counter() - started_at
    generated_ids = output[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, elapsed


def ask_once(model, tokenizer, device, system_prompt: str, prompt: str, args) -> None:
    print("GujinBridge：", end="", flush=True)
    answer, elapsed = generate_answer(model, tokenizer, device, system_prompt, prompt, args)
    print(answer or "（模型没有生成文本）")
    print(f"[耗时 {elapsed:.2f} 秒]\n")


def run_showcase(model, tokenizer, device, system_prompt: str, args) -> None:
    print("=" * 64)
    print("GujinBridge 能力演示")
    print("=" * 64)
    for index, (title, prompt) in enumerate(SHOWCASE_CASES, start=1):
        print(f"\n[{index}/3] {title}")
        print(f"输入：{prompt.splitlines()[-1]}")
        ask_once(model, tokenizer, device, system_prompt, prompt, args)


def expand_command(query: str) -> str:
    command, separator, content = query.partition(" ")
    if separator and command.lower() in COMMAND_PREFIXES:
        return COMMAND_PREFIXES[command.lower()] + content.strip()
    return query


def run_interactive(model, tokenizer, device, system_prompt: str, args) -> None:
    print("=" * 64)
    print("现在可以测试你自己的内容：")
    print("  /c2m 古文       古文译现代汉语")
    print("  /m2c 现代文     现代汉语译古文")
    print("  /punct 古文     断句并添加标点")
    print("  /demo            再跑一次内置演示")
    print("  /quit            退出")
    print("也可以直接输入完整问题。每次请求彼此独立，避免上下文干扰。")
    print("=" * 64)

    while True:
        try:
            query = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if not query:
            continue
        if query.lower() in {"/quit", "quit", "exit"}:
            print("已退出。")
            return
        if query.lower() == "/demo":
            run_showcase(model, tokenizer, device, system_prompt, args)
            continue
        if query.split(" ", 1)[0].lower() in COMMAND_PREFIXES and " " not in query:
            print("请在命令后面加上要处理的文本，例如：/c2m 学而时习之")
            continue
        ask_once(model, tokenizer, device, system_prompt, expand_command(query), args)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0。")
    if args.temperature < 0:
        raise ValueError("--temperature 不能小于 0。")

    system_prompt = load_system_prompt(args.system_prompt_file)
    model, tokenizer, device, _, _ = load_model(args)
    if not args.no_showcase:
        run_showcase(model, tokenizer, device, system_prompt, args)
    if not args.no_interactive:
        run_interactive(model, tokenizer, device, system_prompt, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(2)
