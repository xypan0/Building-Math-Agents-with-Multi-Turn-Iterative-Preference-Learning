"""
This scrip evaluates the math data to check their correctiveness.
"""
import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import json
from eval.grader import *
from utils.parser import *
from utils.utils import load_jsonl
from utils.python_executor import PythonExecutor
from datasets import load_dataset

def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        #samples = list(load_jsonl(file_path))
        dsx = load_dataset(file_path, split='train')#.select(range(1000))        
        new_dsx = dsx.rename_column('idx', 'uidx')
        samples = [sample for sample in new_dsx] 
    # dedup by idx
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]
    
    # parse gt if not in the dataset
    if 'gt' in samples[0]:
        pass
    else:
        for sample in samples:
            sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)

    # execute
    if ('pred' not in samples[0]) or execute:
        if "pal" in prompt_type:
            executor = PythonExecutor(get_answer_expr="solution()")
        else:
            executor = PythonExecutor(get_answer_from_stdout=True)

        for sample in tqdm(samples, desc="Execute"):
            sample['code'] = sample['my_solu']
            sample['pred'] = []
            sample['report'] = []
            for code in sample['code']:
                pred, report = run_execute(executor, code, prompt_type, execute=True)
                sample['pred'].append(pred)
                sample['report'].append(report)

    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool() as pool:
        future = pool.map(math_equal_process, params, timeout=10)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_str = f"Num samples: {len(samples)}\n" \
        f"Num scores: {len(scores)}\n" \
        f"Timeout samples: {timeout_cnt}\n" \
        f"Empty samples: {len([s for s in samples if not s['pred'][-1]])}\n" \
        f"Mean score: {mean_score}\n"

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_str += f"Type scores: {type_scores}\n"

    print(result_str)
    return result_str, samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tora")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    args = parser.parse_args()
    return args

# data_name='gsm8k'
# prompt_type = 'cot' / 'tora'
# 
args = parse_args()
eval_result, all_data = evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path, execute=args.execute)

dict_data = {
            "idx": [d['uidx'] for d in all_data],
            
            "gt": [d['gt'] for d in all_data],
                    #"level": [d['level'] for d in all_data],
                        #"type": [d['type'] for d in all_data],
                    "my_solu": [d['my_solu'] for d in all_data],
                    'score': [d['score'] for d in all_data],
                    "pred": [d['pred'] for d in all_data],
                                }
output_dir = args.output_dir
from datasets import Dataset, DatasetDict
dataset = Dataset.from_dict(dict_data)
DatasetDict({'train': dataset}).push_to_hub(output_dir)

#with open(args.output_dir, "w", encoding="utf8") as f:
#    for i in range(len(all_samples)):
#        json.dump(all_samples[i], f, ensure_ascii=False)
#        f.write('\n')
