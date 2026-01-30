import os, json, logging, torch, random, argparse
from tqdm import tqdm


logger = logging.getLogger(__name__)


def im_refine(input_file: str, out_file: str, draft_file: str, topic_file: str, kw_file: str,
           llm, sampling_params, tokenizer, src_lang: str, tgt_lang: str, batchsize: int = 128):

    with open(input_file, "r", encoding="utf-8") as f:
        src_texts = []
        ret_dense_texts = []
        ret_sparse_texts = []
        for line in f:
            if line.strip():
                data = json.loads(line)
                src_texts.append(data["src_text"])
                ret_dense_texts.append(data["retrieved_dense"])
                ret_sparse_texts.append(data["retrieved_sparse"])
    logger.info(f"成功从 {input_file} 加载 {len(src_texts)} src_texts 条记录,\n "
                f"ret_dense: {len(ret_dense_texts)}, ret_sparse: {len(ret_sparse_texts)} ")

    if not (len(src_texts) == len(ret_dense_texts) == len(ret_sparse_texts)):
        logger.error(f"错误: 文件行数不匹配!")
        return

    with open(draft_file, "r", encoding="utf-8") as f:
        draft_trans = [json.loads(line)['translation'] for line in f if line.strip()]
        logger.info(f"成功加载draft translation: {len(draft_trans)}")

    if not draft_trans:
        logger.error(f"draft trans is empty")
        return

    with open(topic_file, "r", encoding="utf-8") as f:
        topics = [json.loads(line)['topics'] for line in f if line.strip()]
        logger.info(f"成功加载topics: {len(topics)}")
    if not topics:
        logger.error(f"topics is empty")
        return

    with open(kw_file, "r", encoding="utf-8") as f:
        keywords = [json.loads(line)['keywords'] for line in f if line.strip()]
        logger.info(f"成功加载keywords: {len(keywords)}")
    if not keywords:
        logger.error(f"keywords is empty")
        return

    with (open(out_file, "w", encoding="utf-8") as f):
        pbar = tqdm(total=len(src_texts), desc=f"Translating {src_lang} -> {tgt_lang}", unit="sent")
        for i in range(0, len(src_texts), batchsize):
            src_batch = src_texts[i:i + batchsize]
            ret_dense_batch = ret_dense_texts[i:i + batchsize]
            ret_sparse_batch = ret_sparse_texts[i:i + batchsize]
            draft_batch = draft_trans[i:i + batchsize]
            topic_batch = topics[i:i + batchsize]
            kw_batch = keywords[i:i + batchsize]

            prompts = []
            for src_text, ret_dense, ret_sparse, d_tran, topic, kw in \
                zip(src_batch, ret_dense_batch, ret_sparse_batch, draft_batch, topic_batch, kw_batch):
                cn = 1
                prompt = (f"Pay attention to the topics, keywords and analyze the imperfect translation, "
                          f"then ONLY produce the perfect translation of source sentence from {src_lang} to {tgt_lang}.\n")
                for ret_d in ret_dense:
                    prompt += f"Source: {ret_d['src_text']}\n"
                    prompt += f"Topics: {ret_d['topics']}\n"
                    prompt += f"Keywords: {ret_d['keywords']}\n"
                    prompt += f"Imperfect Translation: {ret_d['model_translation']}\n"
                    prompt += f"Perfect Translation: {ret_d['ref_text']}\n"
                    cn += 1
                for ret_s in ret_sparse:
                    prompt += f"Source: {ret_s['src_text']}\n"
                    prompt += f"Topics: {ret_d['topics']}\n"
                    prompt += f"Keywords: {ret_d['keywords']}\n"
                    prompt += f"Imperfect Translation: {ret_s['model_translation']}\n"
                    prompt += f"Perfect Translation: {ret_s['ref_text']}\n"
                    cn += 1


                prompt += f"Source: {src_text}\n"
                prompt += f"Topics: {topic}\n"
                prompt += f"Keywords: {kw}\n"
                prompt += f"Imperfect Translation: {d_tran}\n"
                prompt += f"Perfect Translation: \n"

                messages = [{"role": "user", "content": prompt}]
                prompts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
                )

            outputs = llm.generate(prompts, sampling_params)
            for src, output in zip(src_batch, outputs):
                item = {
                    "sentence": src,
                    "translation": output.outputs[0].text.strip()
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            f.flush()

            pbar.update(len(src_batch))

        pbar.close()
        logger.info(f"完成，保存至 {out_file}")


