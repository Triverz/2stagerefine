import os, re, json, logging, torch, random, argparse
import numpy as np
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams
from tqdm import tqdm


logger = logging.getLogger(__name__)


def direct_translate(llm, tokenizer, sampling_params,
                     file_path: str, output_path: str, src_lang: str, tgt_lang: str, batchsize: int = 128):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line)['text'] for line in f if line.strip()]
    logger.info(f"成功从 {file_path} 加载 {len(data)} 条记录。")

    with open(output_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=len(data), desc=f"Translating {src_lang} -> {tgt_lang}", unit="sent")
        for i in range(0, len(data), batchsize):
            batch = data[i:i + batchsize]
            prompts = []
            for sent in batch:
                prompt = f"Translate the following sentence from {src_lang} to {tgt_lang}:\n"
                prompt += f"{sent}\n"
                messages = [{"role": "user", "content": prompt}]
                prompts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                )

            outputs = llm.generate(prompts, sampling_params)

            for src, output in zip(batch, outputs):
                item = {
                    "sentence": src,
                    "translation": output.outputs[0].text.strip()
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            f.flush()

            pbar.update(len(batch))

        pbar.close()
        logger.info(f"翻译完成，保存至 {output_path}")



