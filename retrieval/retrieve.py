import os, json, logging, argparse, torch, random
import numpy as np
from transformers import set_seed
from tqdm import tqdm
from .dense import DenseKB
from .sparse import SparseKB
from pathlib import Path

logger = logging.getLogger(__name__)



def retrieve(input_file: str, output_file: str, src_lang: str, tgt_lang: str,
              dense_kb: DenseKB, sparse_kb: SparseKB, dense_k: int, sparse_k: int):
    data = []
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
    logger.info(f"Loading input file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            entry = json.loads(line)
            # 检查语言是否存在
            if 'text' not in entry:
                raise KeyError(f" 文件 '{input_file}' 中 键 'text' not found in JSON line: {line}")
            data.append(entry.get('text'))
    logger.info(f"total {len(data)} sentences")

    start_idx = 0
    if os.path.exists(output_file):
        logger.info(f"检测到输出文件 {output_file}，正在计算断点...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    start_idx += 1
        logger.info(f"已处理 {start_idx} 条数据，将从第 {start_idx + 1} 条开始继续。")
    else:
        logger.info("未找到输出文件，将从头开始处理。")


    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f_out:
        for idx, text in tqdm(enumerate(data, 1), total=len(data), desc="Retrieving ..."):
            if idx <= start_idx:
                continue
            try:
                dense_res = dense_kb.search(text, src_lang, tgt_lang, dense_k)
            except Exception as e:
                logger.error(f"Dense search failed at ID {idx}: {e}")
            try:
                sparse_res = sparse_kb.search(text, tgt_lang, sparse_k)
            except Exception as e:
                logger.error(f"Sparse search failed at ID {idx}: {e}")

            output = {
                "id": idx,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src_text": text,
                "retrieved_dense": dense_res,
                "retrieved_sparse": sparse_res,
            }
            f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
            f_out.flush()
    logger.info(f"Retrieval done! Saved to {output_file}")

