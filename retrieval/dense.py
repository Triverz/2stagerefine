import os, json, hashlib, logging, re, chromadb, torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# 配置

logger = logging.getLogger(__name__)

class DenseKB:
    def __init__(self, model_path: str, db_dir: str, collection_name: str,  device: str = "cuda", batch_size: int = 8192):
        os.makedirs(os.path.abspath(db_dir), exist_ok=True)
        self.batch_size = batch_size
        self.ckpt_path = os.path.join(db_dir, "checkpoint.json")

        logger.info("Loading Model & DB...")
        self.model = SentenceTransformer(
            model_path,
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )
        self.client = chromadb.PersistentClient(path=db_dir)
        self.coll = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def build(self, jsonl_dir: str, topic_file: str, kw_file: str):
        """构建知识库（支持断点续传）"""
        ckpt = json.load(open(self.ckpt_path)) if os.path.exists(self.ckpt_path) else {"processed": {}, "total": 0}

        for f_path in Path(jsonl_dir).glob("*.jsonl"):
            fname = f_path.name
            try:
                ds, s_lang, t_lang, seed= f_path.stem.split('_')[:4]
                seed = int(re.findall(r'\d+', seed)[0])
            except:
                continue

            start_line = ckpt["processed"].get(fname, 0)
            if start_line:
                logger.info(f"Resuming {fname} from line {start_line}")

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

            batch = {"ids": [], "docs": [], "metas": []}
            total_lines = sum(1 for _ in open(f_path, encoding='utf-8') if _.strip())

            with open(f_path, 'r', encoding='utf-8') as f:
                pbar = tqdm(enumerate(f, 1), total=total_lines, initial=start_line, desc=fname)
                for i, line in pbar:
                    if i <= start_line:
                        continue
                    try:
                        data = json.loads(line)
                        src, trans, ref = data.get('sentence'), data.get('translation'), data.get('reference')
                        if not src or not trans or not ref: continue

                        # 生成双向数据 (源->目, 0-shot翻译->源, 可以源语句检索，也可以通过翻译检索错误语句)
                        batch["docs"].append(src)
                        batch["metas"].append(
                            {"dataset": ds, "filename": fname, "line": i, "src_lang": s_lang, "tgt_lang": t_lang,
                             "topics": topics[i-1], "keywords": keywords[i-1],
                             "model_translation": trans, "reference": ref, "seed": seed})
                        batch["ids"].append(hashlib.md5(f"{src}|{trans}|{ref}|{s_lang}|{t_lang}|{seed}|{i}".encode('utf-8')).hexdigest())

                        # batch["docs"].append(trans)
                        # batch["metas"].append(
                        #     {"dataset": ds, "filename": fname, "line": i, "src_lang": s_lang, "tgt_lang": t_lang,
                        #      "src": src, "reference": ref})
                        # batch["ids"].append(hashlib.md5(f"{trans}|{src}|{ref}|{s_lang}|{t_lang}".encode('utf-8')).hexdigest())

                    except:
                        continue

                    # 批量写入
                    if len(batch["ids"]) >= self.batch_size:
                        self._flush(batch)
                        ckpt["processed"][fname] = i
                        ckpt["total"] += len(batch["ids"]) #// 2
                        json.dump(ckpt, open(self.ckpt_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
                        batch = {"ids": [], "docs": [], "metas": []}

                if batch["ids"]:  # 处理尾部
                    self._flush(batch)
                    ckpt["processed"][fname] = i
                    ckpt["total"] += len(batch["ids"]) #// 2
                    json.dump(ckpt, open(self.ckpt_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    def _flush(self, batch):
        if not batch["ids"]: return
        vecs = self.model.encode(batch["docs"], batch_size=1024, normalize_embeddings=True).tolist()
        self.coll.add(ids=batch["ids"], embeddings=vecs, metadatas=batch["metas"], documents=batch["docs"])

    def search(self, query: str, src_lang: str, tgt_lang: str, k: int = 2):
        """语义检索"""
        vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()
        res = self.coll.query(query_embeddings=vec, n_results=k,
                              where={"$and": [{"src_lang": src_lang}, {"tgt_lang": tgt_lang}]})
        format_res = []
        for doc, meta, score in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
            format_res.append({
                "src_text": doc,
                "model_translation": meta["model_translation"],
                "ref_text": meta["reference"],
                "topics": meta["topics"],
                "keywords": meta["keywords"],
                "score": 1-score,
                "src_lang": meta["src_lang"],
                "tgt_lang": meta["tgt_lang"],
                "dataset": meta["dataset"],
                "filename": meta["filename"],
                "line": meta["line"],
                "translation_seed": meta["seed"],
                "retrieval_type": "dense"
            })
        return format_res


