import os, logging, json, random, torch
import numpy as np
from transformers import AutoTokenizer, set_seed
from tqdm import tqdm
from vllm import LLM, SamplingParams
from .template import *



logger = logging.getLogger(__name__)


def extract_topic(input_file: str, out_file: str, llm, sampling_params, tokenizer, batchsize: int = 128):
    src_lang = os.path.basename(input_file).split('.')[0].split('_')[-1]
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line)['text'] for line in f if line.strip()]
    logger.info(f"成功从 {input_file} 加载 {len(data)} 条记录。")

    with open(out_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=len(data), desc=f"Extracting topics", unit="sent")
        for i in range(0, len(data), batchsize):
            batch = data[i:i + batchsize]
            prompts = []
            for sent in batch:
                prompt = "Use a few words to describe the topics of the following input sentence.\n"
                for demo_sent, demo_topic in zip(TRIGGER_SENTS[src_lang], TOPICS):
                    prompt += f"Input: {demo_sent}\n"
                    prompt += f"Topics: {demo_topic}\n"
                prompt += f"Input: {sent}\n"
                prompt += f"Topics: \n"

                messages = [{"role": "user", "content": prompt}]
                prompts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
                )

            outputs = llm.generate(prompts, sampling_params)
            for sent, output in zip(batch, outputs):
                item = {
                    "sentence": sent,
                    "topics": output.outputs[0].text.strip().split('Topics:')[-1].strip()
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            f.flush()

            pbar.update(len(batch))

        pbar.close()
        logger.info(f"提取Topics完成，保存至 {out_file}")




