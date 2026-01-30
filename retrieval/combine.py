import json, logging, glob, os

logger = logging.getLogger(__name__)

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
                            logger.info(f"文件 {file_path} 第 {i + 1} 行缺少键: '{key_name}'")
                    except json.JSONDecodeError:
                        logger.info(f"文件 {file_path} 第 {i + 1} 行 JSON 解码失败: {line.strip()}")
    except FileNotFoundError:
        logger.info(f"错误: 文件未找到 {file_path}")
        return None

    logger.info(f"成功从 {file_path} (键: '{key_name}') 加载 {len(data)} 条记录。")
    return data


def combine(trans_dir: str, ref_dir: str, out_dir: str, dataset: str):

    trans_files = sorted(glob.glob(os.path.join(trans_dir, "*.jsonl")))
    if not trans_files:
        logger.info(f"在 {trans_dir} 没有找到 .jsonl 文件。")
        return
    logger.info(f"找到 {len(trans_files)} 个文件待评估。\n")
    logger.info("-" * 100)

    for idx, trans_file in enumerate(trans_files, 1):
        logger.info(f"Processing {idx}. {trans_file}\n")
        filename = os.path.basename(trans_file)
        src_lang = filename.split(".")[0].split("_")[0]
        tgt_lang = filename.split(".")[0].split("_")[1]
        seed = filename.split(".")[0].split("_")[2]
        ref_file = os.path.join(ref_dir, f"{dataset}_{tgt_lang}.jsonl")

        src_data = load_jsonl(trans_file, 'sentence')
        translation = load_jsonl(trans_file, 'translation')
        ref_data = load_jsonl(ref_file, 'text')

        if not (len(src_data) == len(translation) == len(ref_data)):
            logger.warning(f"错误: 文件行数不匹配!")
            return

        logger.info(f"文件共有{len(src_data)}行")

        out_file = os.path.join(out_dir, f"{dataset}_{src_lang}_{tgt_lang}_{seed}_translation.jsonl")
        with open(out_file, 'w', encoding='utf-8') as f:
            for idx, (src, trans, ref) in enumerate(zip(src_data, translation, ref_data), 1):
                item = {
                    "id": idx,
                    "sentence": src,
                    "translation": trans,
                    "reference": ref,
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"合并完成，保存到{out_file}")

