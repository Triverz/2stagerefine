import os, argparse, logging, torch, random, gc
import numpy as np
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams


from func.translate import direct_translate
from func.topic import extract_topic
from func.kw import extract_keywords
from retrieval.combine import combine
from retrieval.dense import DenseKB
from retrieval.sparse import SparseKB
from retrieval.retrieve import retrieve
from func.im_ref import im_refine
from func.ex_ref import ex_refine


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(),
              logging.FileHandler("./main.log")]
)
logger = logging.getLogger(__name__)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    """
    定义并解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="Two-Stage Refinement")

    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称 (例如: tico, flores, ntrex)")
    parser.add_argument("--src_lang", type=str, help="源语言名称")
    parser.add_argument("--tgt_lang", type=str, help="目标语言")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")


    parser.add_argument("--output_dir", type=str,
                        help="所有输出文件的根目录")

    parser.add_argument("--llm_path", type=str, required=True,
                        help="LLM 模型路径 (Qwen)")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Embedding 模型路径")

    parser.add_argument("--batch_size", type=int, default=128,
                        help="推理和提取时的 Batch Size")

    parser.add_argument("--dense_k", type=int, default=2,
                        help="Dense retrieval top-k")
    parser.add_argument("--sparse_k", type=int, default=1,
                        help="Sparse retrieval top-k")
    parser.add_argument("--max_iterations", type=int, default=3, help="显式修正的迭代次数 (T)")


    args = parser.parse_args()
    return args

def clean_gpu():
    """强制清理显存，用于在LLM和Embedding模型切换时"""
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("已清理 GPU 显存")


def main(args):

    model_name = os.path.basename(args.llm_path.rstrip('/'))  # 防止路径末尾有/导致取不到名字
    test_file = (f"./data/{args.dataset}/{'devtest' if 'flores' in args.dataset else 'test'}/"
                 f"{args.dataset}_{args.src_lang}.jsonl")

    logger.info(" Preparing Offline Knowledge Base ")

    # 加载 LLM
    logger.info(f"Loading LLM: {args.llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    llm = LLM(
        model=args.llm_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        dtype='bfloat16',
        max_model_len=16384,
        enforce_eager=False
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id],
        skip_special_tokens=True,
        seed=args.seed
    )
    ret_src_file = (f"./data/{args.dataset}/{'ret' if args.dataset in 'ntrex' else 'dev'}/"
               f"{args.dataset}{'ret' if args.dataset in 'ntrex' else 'dev'}_{args.src_lang}.jsonl")
    ret_ref_file = (f"./data/{args.dataset}/{'ret' if args.dataset in 'ntrex' else 'dev'}/"
               f"{args.dataset}{'ret' if args.dataset in 'ntrex' else 'dev'}_{args.tgt_lang}.jsonl")
    ret_trans_file = (f"./retrieval/translations/{args.dataset}{'ret' if args.dataset in 'ntrex' else 'dev'}/"
                      f"{model_name}/{args.src_lang}_{args.tgt_lang}_seed{args.seed}_translation.jsonl")
    os.makedirs(os.path.dirname(ret_trans_file), exist_ok=True)

    comb_out_dir = (f"./retrieval/combine_data/{args.dataset}{'ret' if args.dataset in 'ntrex' else 'dev'}/{model_name}")
    os.makedirs(comb_out_dir, exist_ok=True)

    ret_res_file = (f"./retrieval/ret_res/{args.dataset}/{model_name}/"
                    f"{args.dataset}_{args.src_lang}_{args.tgt_lang}_seed{args.seed}_d{args.dense_k}_s{args.sparse_k}.jsonl")
    os.makedirs(os.path.dirname(ret_res_file), exist_ok=True)

    ret_topic_file = (f"./topics/{args.dataset}{'ret' if args.dataset in 'ntrex' else 'dev'}/{model_name}/"
                     f"{args.dataset}_{args.src_lang}_{args.tgt_lang}_seed{args.seed}_topics.jsonl")
    os.makedirs(os.path.dirname(ret_topic_file), exist_ok=True)

    ret_kw_file = (f"./keywords/{args.dataset}{'ret' if args.dataset in 'ntrex' else 'dev'}/{model_name}/"
                   f"{args.dataset}_{args.src_lang}_{args.tgt_lang}_seed{args.seed}_keywords.jsonl")
    os.makedirs(os.path.dirname(ret_kw_file), exist_ok=True)

    logger.info(" Translating retrieval sentences ")

    if os.path.exists(ret_trans_file) and os.path.isfile(ret_trans_file):
        logger.info(" Translated! Skip!")
    else:
        direct_translate(llm, tokenizer, sampling_params,
                        ret_src_file, ret_trans_file, args.src_lang, args.tgt_lang, args.batch_size)
        logger.info(" Translated!")

    logger.info(" Extracting topics of retrieval sentences ")

    if os.path.exists(ret_topic_file) and os.path.isfile(ret_topic_file):
        logger.info(" Extracted Topics! Skip!")
    else:
        extract_topic(ret_src_file, ret_topic_file, llm, sampling_params, tokenizer, args.batch_size)
        logger.info(" Extracted Topics!")

    if os.path.exists(ret_kw_file) and os.path.isfile(ret_kw_file):
        logger.info(" Extracted Keywords! Skip!")
    else:
        extract_keywords(ret_src_file, ret_kw_file, llm, sampling_params, tokenizer, args.batch_size)
        logger.info(" Extracted Keywords!")

    combine(os.path.dirname(ret_trans_file), os.path.dirname(ret_ref_file), comb_out_dir,
            f"{args.dataset}{'ret' if args.dataset in 'ntrex' else 'dev'}")
    logger.info(" Combined! ")

    del llm, tokenizer
    clean_gpu()

    logger.info(" Start Retrieval ")
    collection_name = f"dense_{args.src_lang}_src"

    kb_dir = f"./retrieval/kb/{model_name}/{collection_name}"
    os.makedirs(kb_dir, exist_ok=True)

    dense_kb = DenseKB(
        model_path=args.embedding_path,
        db_dir=kb_dir,
        collection_name=collection_name
    )
    if not (os.path.exists(f"{kb_dir}/checkpoint.json") and os.path.isfile(f"{kb_dir}/checkpoint.json")):
        logger.info(" Constructing DenseKB ")
        dense_kb.build(comb_out_dir, ret_topic_file, ret_kw_file)

    logger.info(" Constructed! ")

    sparse_kb = SparseKB(comb_out_dir, args.src_lang, ret_topic_file, ret_kw_file)

    retrieve(input_file=test_file, output_file=ret_res_file, src_lang=args.src_lang, tgt_lang=args.tgt_lang,
             dense_kb=dense_kb, sparse_kb=sparse_kb, dense_k=args.dense_k, sparse_k=args.sparse_k)

    logger.info(" Finished Retrieval! ")

    del dense_kb
    del sparse_kb

    gc.collect()
    torch.cuda.empty_cache()


    logger.info(f"Loading LLM: {args.llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    llm = LLM(
        model=args.llm_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        dtype='bfloat16',
        max_model_len=16384,
        enforce_eager=False
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id],
        skip_special_tokens=True,
        seed=args.seed
    )

    test_trans_file = (f"{args.output_dir}/{args.dataset}/direct/{model_name}/"
                       f"{args.src_lang}_{args.tgt_lang}_seed{args.seed}_translation.jsonl")
    os.makedirs(os.path.dirname(test_trans_file), exist_ok=True)

    logger.info(" Translating test sentences in 0-shot ")
    direct_translate(llm, tokenizer, sampling_params,
                     test_file, test_trans_file, args.src_lang, args.tgt_lang, args.batch_size)
    logger.info(" Translated in 0-shot !")

    test_topic_file = (f"./topics/{args.dataset}/{model_name}/"
                      f"{args.dataset}_{args.src_lang}_{args.tgt_lang}_seed{args.seed}_topics.jsonl")
    os.makedirs(os.path.dirname(test_topic_file), exist_ok=True)

    test_kw_file = (f"./keywords/{args.dataset}/{model_name}/"
                    f"{args.dataset}_{args.src_lang}_{args.tgt_lang}_seed{args.seed}_keywords.jsonl")
    os.makedirs(os.path.dirname(test_kw_file), exist_ok=True)

    logger.info(" Extracting topics of test sentences ")
    if os.path.exists(test_topic_file) and os.path.isfile(test_topic_file):
        logger.info(" Extracted! Skip!")
    else:
        extract_topic(test_file, test_topic_file, llm, sampling_params, tokenizer, args.batch_size)
        logger.info(" Extracted! ")

    logger.info(" Extracting keywords of test sentences ")
    if os.path.exists(test_kw_file) and os.path.isfile(test_kw_file):
        logger.info(" Extracted! Skip!")
    else:
        extract_keywords(test_file, test_kw_file, llm, sampling_params, tokenizer, args.batch_size)
        logger.info(" Extracted! ")


    logger.info(" Implicit refinement ")

    im_trans_file = (f"{args.output_dir}/{args.dataset}/im_ref/{model_name}/"
                       f"{args.src_lang}_{args.tgt_lang}_seed{args.seed}_translation.jsonl")
    os.makedirs(os.path.dirname(im_trans_file), exist_ok=True)

    im_refine(input_file=ret_res_file, out_file=im_trans_file, draft_file=test_trans_file, topic_file=test_topic_file,
              kw_file=test_kw_file, llm=llm, sampling_params=sampling_params, tokenizer=tokenizer,
              src_lang=args.src_lang, tgt_lang=args.tgt_lang, batchsize=args.batch_size)
    logger.info(" Finished Implicit refinement! ")
    last_ex_file = im_trans_file

    logger.info(" Explicit refinement ")
    for i in range(args.max_iterations):

        logger.info(f" Iteration {i+1} ")
        if i ==0:
            ex_trans_file = (f"{args.output_dir}/{args.dataset}/ex_ref_1/{model_name}/ref/"
                             f"{args.src_lang}_{args.tgt_lang}_seed{args.seed}_translation.jsonl")
            os.makedirs(os.path.dirname(ex_trans_file), exist_ok=True)

            est_file = (f"{args.output_dir}/{args.dataset}/ex_ref_1/{model_name}/est/"
                        f"{args.src_lang}_{args.tgt_lang}_seed{args.seed}_estimation.jsonl")
            os.makedirs(os.path.dirname(est_file), exist_ok=True)

            ex_refine(input_file=last_ex_file, est_file=est_file, out_file=ex_trans_file,
                      llm=llm, sampling_params=sampling_params, tokenizer=tokenizer,
                      src_lang=args.src_lang, tgt_lang=args.tgt_lang,batchsize=args.batch_size)
            last_ex_file = ex_trans_file
        else:
            ex_trans_file = (f"{args.output_dir}/{args.dataset}/ex_ref_{i+1}/{model_name}/"
                            f"{args.src_lang}_{args.tgt_lang}_seed{args.seed}_translation.jsonl")
            os.makedirs(os.path.dirname(ex_trans_file), exist_ok=True)

            est_file = (f"{args.output_dir}/{args.dataset}/ex_ref_{i+1}/{model_name}/est/"
                        f"{args.src_lang}_{args.tgt_lang}_seed{args.seed}_estimation.jsonl")
            os.makedirs(os.path.dirname(est_file), exist_ok=True)

            ex_refine(input_file=last_ex_file, est_file=est_file, out_file=ex_trans_file,
                      llm=llm, sampling_params=sampling_params, tokenizer=tokenizer,
                      src_lang=args.src_lang, tgt_lang=args.tgt_lang, batchsize=args.batch_size)

            last_ex_file = ex_trans_file
    logger.info(" Finished Explicit refinement! ")
    logger.info(" Finished all! ")

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)