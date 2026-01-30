import os, json, glob
import numpy as np
from bleurt import score



INPUT_DIR = "./out/direct/Qwen3-30B-A3B-Instruct-2507"


REF_DIR = "./data/flores/devtest"
DATASET = "flores"


BLEURT_CHECKPOINT = "./bleurt_model/BLEURT-20"

# 批处理大小
BATCH_SIZE = 32


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

    # 1. 加载 BLEURT 模型
    print("Loading BLEURT Scorer...")
    if not os.path.exists(BLEURT_CHECKPOINT):
        print(f"错误: 找不到 BLEURT 模型路径: {BLEURT_CHECKPOINT}")
        return

    try:
        scorer = score.BleurtScorer(BLEURT_CHECKPOINT)
        print("BLEURT Scorer loaded successfully.")
    except Exception as e:
        print(f"加载 BLEURT 失败: {e}")
        return

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


        try:
            tgt_lang = filename.split(".")[0].split("_")[1]
        except IndexError:
            print(f"警告: 无法从文件名 {filename} 解析语言代码，跳过。")
            continue

        ref_path = os.path.join(REF_DIR, f"{DATASET}_{tgt_lang}.jsonl")

        print(f"开始评估： {idx}. {filename}")


        # src_data = load_jsonl(hyp_file, 'sentence') # BLEURT 不需要源句
        hyp_data = load_jsonl(hyp_file, 'translation')
        ref_data = load_jsonl(ref_path, 'text')

        if hyp_data is None or ref_data is None:
            print("文件加载失败，程序终止")
            continue

        if len(hyp_data) != len(ref_data):
            print(f"错误: 文件行数不匹配!")
            print(f"  假设文件 ({hyp_file}): {len(hyp_data)} 行")
            print(f"  参考文件 ({ref_path}): {len(ref_data)} 行")
            return  # 或者 continue，取决于你想不想要强行终止

        if not hyp_data:
            print("错误，未加载到任何数据，程序终止")
            continue

        print(f"数据加载成功，共 {len(hyp_data)} 行待评估。")
        print("开始计算 BLEURT 分数...")

        # 计算分数
        try:
            scores = scorer.score(references=ref_data, candidates=hyp_data, batch_size=BATCH_SIZE)

            # 计算平均分
            avg_bleurt = np.mean(scores)

            print(f"\n{idx}: {filename}\nBLEURT = {avg_bleurt:.4f}\n")

        except Exception as e:
            print(f"计算出错: {e}")


if __name__ == "__main__":
    main()