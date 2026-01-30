import os, json, logging, jieba, bm25s, Stemmer
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)



class SparseKB:
    def __init__(self, data_dir: str, src_lang: str, topic_file: str, kw_file: str):

        self.data_dir = data_dir
        self.src_lang = src_lang
        self.retriever = None
        self.corpus = []

        # 1. 准备分词器 (EN用Stemmer, ZH用Jieba)
        if src_lang == "Chinese":

            self.tokenizer = lambda texts: [jieba.lcut(t) for t in texts]
        else:
            self.stemmer = Stemmer.Stemmer("english")
            self.tokenizer = lambda texts: bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer)

        self.build(data_dir, topic_file, kw_file)

    def build(self, jsonl_dir: str, topic_file: str, kw_file: str):
        """扫描目录构建索引"""
        logger.info(f"Building SparseKB from {jsonl_dir} ...")

        with open(topic_file, 'r', encoding="utf-8") as f:
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

        records = {}

        files = list(Path(jsonl_dir).glob("*.jsonl"))
        for f_path in tqdm(files, desc="Loading"):
            try:
                parts = f_path.stem.split('_')
                if len(parts) < 3: continue
                ds, s_lang, t_lang = parts[:3]  # 解析数据集名

                if s_lang.lower() != self.src_lang.lower():
                    raise ValueError(f"{s_lang} != {self.src_lang}")

                with open(f_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):

                        data = json.loads(line)
                        src, trans, ref = data.get('sentence'), data.get('translation'), data.get('reference')
                        if not src: continue

                        key = f"{ds}_{i}"

                        if key not in records:
                            records[key] = {
                                "src_text": src,
                                "topics": topics[i-1],
                                "keywords": keywords[i-1],
                                "model_translation": {},
                                "reference": {},  # 用字典存所有目标语言的翻译
                                "dataset": ds,
                                "line": i
                            }

                        records[key]["model_translation"][t_lang] = trans
                        records[key]["reference"][t_lang] = ref

            except Exception as e:
                logger.warning(f"Error reading {f_path.name}: {e}")

        self.corpus = list(records.values())
        corpus_texts = [item["src_text"] for item in self.corpus]
        logger.info(f"Indexing {len(corpus_texts)} unique sentences...")
        # 构建与保存
        self.retriever = bm25s.BM25(method="robertson", k1=1.5, b=0.75)
        self.retriever.index(self.tokenizer(corpus_texts))


    def search(self, query: str, tgt_lang: str, k: int = 1):
        """检索接口"""
        if not self.retriever: return []

        # 检索
        query_token = self.tokenizer([query])
        doc_ids, scores = self.retriever.retrieve(query_token, k=k)


        res = []
        for idx, score in zip(doc_ids[0], scores[0]):
            item = self.corpus[int(idx)]
            model_trans = item["model_translation"].get(tgt_lang, None)
            reference = item["reference"].get(tgt_lang, None)
            res.append({
                "src_text": item["src_text"],
                "model_translation": model_trans,
                "ref_text": reference,
                "topics": item["topics"],
                "keywords": item["keywords"],
                "dataset": item["dataset"],
                "line": item["line"],
                "score": float(score),
                "retrieval_type": "sparse"
            })
        return res
