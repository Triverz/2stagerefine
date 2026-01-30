import os
import json
import glob
import torch
import sacrebleu
from comet import load_from_checkpoint
from tqdm import tqdm



DATASET = "flores"

INPUT_DIR = f"./out/direct/Qwen3-30B-A3B-Instruct-2507/"



REF_DIR = f"./data/{DATASET}/devtest"


# 本地 COMET / XCOMET 模型路径
COMET_MODEL_PATH = "./comet_model/wmt22-comet-da/checkpoints/model.ckpt"
XCOMET_MODEL_PATH = "./comet_model/XCOMET-XL/checkpoints/model.ckpt"

# 批处理大小 (根据显存调整)
BATCH_SIZE = 16


# ===========================================

def load_jsonl(file_path: str, key_name: str):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        json_obj = json.loads(line)
                        if key_name in json_obj:
                            data.append(json_obj[key_name])
                        else:
                            print(f"文件 {file_path} 第 {i + 1} 行缺少键: '{key_name}'")
                    except json.JSONDecodeError:
                        print(f"文件 {file_path} 第 {i + 1} 行 JSON 解码失败: {line.strip()}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return None

    print(f"成功从 {file_path} (键: '{key_name}') 加载 {len(data)} 条记录。")
    return data


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载评测模型到 {device}...")

    # 1. 加载模型
    # 检查路径是否存在，不存在则视为模型名称自动下载
    print("Loading COMET...")
    comet_model = load_from_checkpoint(COMET_MODEL_PATH, reload_hparams=True, local_files_only=True)
    comet_model.eval()
    print("COMET model loaded.")

    print("Loading XCOMET...")
    xcomet_model = load_from_checkpoint(XCOMET_MODEL_PATH, reload_hparams=True, local_files_only=True)
    xcomet_model.eval()
    print("XCOMET model loaded.")

    # 2. 扫描翻译结果文件
    result_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl")))
    if not result_files:
        print(f"在 {INPUT_DIR} 没有找到 .jsonl 文件。")
        return

    print(f"找到 {len(result_files)} 个文件待评估。\n")
    print("-" * 100)

    # 3. 逐个文件评估
    for idx, hyp_file in enumerate(result_files, 1):
        filename = os.path.basename(hyp_file)
        tgt_lang = filename.split(".")[0].split("_")[1]

        ref_path = os.path.join(REF_DIR, f"{DATASET}_{tgt_lang}.jsonl")

        print(f"开始评估： {idx}. {filename}")

        # 读取模型生成的翻译
        src_data = load_jsonl(hyp_file, 'sentence')
        hyp_data = load_jsonl(hyp_file, 'translation')
        ref_data = load_jsonl(ref_path, 'text')

        if src_data is None or hyp_data is None or ref_data is None:
            print("文件加载失败，程序终止")

        if not (len(src_data) == len(hyp_data) == len(ref_data)):
            print(f"错误: 文件行数不匹配!")
            print(f"  源文件 ({hyp_file}): {len(src_data)} 行")
            print(f"  假设文件 ({hyp_file}): {len(hyp_data)} 行")
            print(f"  参考文件 ({ref_path}): {len(ref_data)} 行")
            return
        if not hyp_data:
            print("错误，未加载到任何数据，程序终止")

        print(f"数据加载成功，共 {len(hyp_data)} 行待评估。")

        bleu_score = sacrebleu.corpus_bleu(hyp_data, [ref_data], tokenize="flores200").score
        chrf_score = sacrebleu.corpus_chrf(hyp_data, [ref_data], word_order=2).score

        comet_data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(src_data, hyp_data, ref_data)]
        print("开始计算COMET分数")
        comet_score = comet_model.predict(comet_data, batch_size=BATCH_SIZE, gpus=1).system_score

        print("开始计算XCOMET分数")
        xcomet_score = xcomet_model.predict(comet_data, batch_size=BATCH_SIZE, gpus=1).system_score

        print(
            f"\n{idx}: {filename}\nBLEU = {bleu_score}\nchrF++ = {chrf_score}\nCOMET = {comet_score}\nXCOMET = {xcomet_score}\n")


if __name__ == "__main__":
    main()