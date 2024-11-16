import os
import json
import random
from datasets import load_dataset, Dataset, concatenate_datasets
from utils.utils import load_jsonl, lower_keys

def load_data(data_name, split, data_dir='./data'):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset("competition_math", split=split, name="main", cache_dir=f"{data_dir}/temp")
        elif "RLHF4MATH/prompt_iter" in data_name:
            # if we use the pre-processed prompts
            dataset = load_dataset(data_name, split=split)
        elif data_name == "theorem-qa":
            dataset = load_dataset("wenhu/TheoremQA", split=split)
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "gsm-hard":
            dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        elif data_name == "svamp":
            # evaluate on training set + test set 
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets([dataset, load_dataset("ChilleD/SVAMP", split="test")])
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(lambda x: ";" not in x['answer']) # remove multi-answer examples
        elif data_name == "mawps":
            examples = []
            # four sub-tasks
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example['type'] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "finqa":
            dataset = load_dataset("dreamerdeo/finqa", split=split, name="main")
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "tabmwp":
            examples = []
            with open(f"{data_dir}/tabmwp/tabmwp_{split}.json", "r") as f:
                data_dict = json.load(f)
                examples.extend(data_dict.values())
            dataset = Dataset.from_list(examples)
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "bbh":
            examples = []
            for data_name in ["reasoning_about_colored_objects", "penguins_in_a_table",\
                            "date_understanding", "repeat_copy_logic", "object_counting"]:
                with open(f"{data_dir}/bbh/bbh/{data_name}.json", "r") as f:
                    sub_examples = json.load(f)["examples"]
                    for example in sub_examples:
                        example['type'] = data_name
                    examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if 'idx' not in examples[0]:
        examples = [{'idx': i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x['idx'])
    return examples
(base) wexiong_google_com@carl-spring2024-a100:~$ cat math_exp/RLHF4MATH_Dev/inference/utils/data_loader.py |wc -l
74
(base) wexiong_google_com@carl-spring2024-a100:~$ cat math_exp/RLHF4MATH_Dev/inference/utils/
__pycache__/        data_loader.py      parser.py           utils.py            
annotate_data.py    filter_data.py      python_executor.py  
(base) wexiong_google_com@carl-spring2024-a100:~$ cat math_exp/RLHF4MATH_Dev/inference/utils/parser.py 
import re
from typing import Any, Dict


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_answer(pred_str):
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif extract_program_output(pred_str) != "":
        # fall back to program
        pred = extract_program_output(pred_str)
    else:  # use the last number
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if len(pred) >= 1:
            pred = pred[-1]
        else:
            pred = ""

    # multiple line
    pred = pred.split("\n")[0]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred


'''
def extract_program(result: str, last_only=True):
    """
    extract the program after "```python", and before "```"
    """
    program = ""
    start = False
    for line in result.split("\n"):
        if line.startswith("```python"):
            if last_only:
