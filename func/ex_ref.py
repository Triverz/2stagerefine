import os, json, logging, torch, random
from tqdm import tqdm
from .template import *

logger = logging.getLogger(__name__)


def ex_refine(input_file: str, est_file: str, out_file: str,
           llm, sampling_params, tokenizer, src_lang: str, tgt_lang: str, batchsize: int = 128):
    with open(input_file, 'r', encoding='utf-8') as f:
        src_texts = []
        trans_texts = []
        for line in f:
            if line.strip():
                data = json.loads(line)
                src_texts.append(data["sentence"])
                trans_texts.append(data["translation"])
    logger.info(f"成功从 {input_file} 加载 src_texts: {len(src_texts)} 条记录, trans_texts: {len(trans_texts)} 条记录")

    if not (len(src_texts) == len(trans_texts)):
        logger.error(f"错误: 文件行数不匹配!")
        return

    with (open(est_file, 'w', encoding='utf-8') as f_est, \
         open(out_file, 'w', encoding='utf-8') as f_out):
        pbar = tqdm(total=len(src_texts), desc=f"Translating {src_lang} -> {tgt_lang}", unit="sent")
        for i in range(0, len(src_texts), batchsize):
            src_batch = src_texts[i:i + batchsize]
            trans_batch = trans_texts[i:i + batchsize]

            logger.info(f"Estimating {src_lang} -> {tgt_lang}")
            header = "Given any instruction, make sure to only return the expected answer, nothing before and nothing after.\n\n"
            est_prompts = []
            for src_text, trans_text in zip(src_batch, trans_batch):
                est_prompt = ESTIMATE.format(src_lan=src_lang, tgt_lan=tgt_lang, origin=src_text, init_trans=trans_text)

                messages = [{"role": "user", "content": header + est_prompt}]
                est_prompts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
                )

            est_outputs = llm.generate(est_prompts, sampling_params)
            refine_prompts = []
            logger.info(f"Refining {src_lang} -> {tgt_lang}")

            for src, output, trans_text in zip(src_batch, est_outputs, trans_batch):
                item = {
                    "sentence": src,
                    "estimate_fdb": output.outputs[0].text.strip()
                }
                f_est.write(json.dumps(item, ensure_ascii=False) + '\n')

                refine_prompt = REFINE.format(src_lan=src_lang, tgt_lan=tgt_lang, raw_src=src,
                                              raw_mt=trans_text, estimate_fdb=output.outputs[0].text.strip())

                messages = [{"role": "user", "content": header + refine_prompt}]
                refine_prompts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
                )
            f_est.flush()

            ref_outputs = llm.generate(refine_prompts, sampling_params)
            for src, output in zip(src_batch, ref_outputs):
                item = {
                    "sentence": src,
                    "translation": output.outputs[0].text.strip()
                }
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            f_out.flush()

            pbar.update(len(src_batch))

        pbar.close()
        logger.info(f"完成，Estimate保存至 {est_file}\n Refine 保存至 {out_file}")


