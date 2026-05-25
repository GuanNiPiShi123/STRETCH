from typing import List, Dict

from numpy import random
import pandas as pd
from transformers import AutoTokenizer

from absolute_zero_reasoner.data_construction.prompts import get_code_problem_generator_prompt, get_code_problem_predictor_prompt
from absolute_zero_reasoner.data_construction.process_data import boxed_instruction, instruction_following
from absolute_zero_reasoner.utils.code_utils.parsers import replace_main_function_name

#修补
def get_gen_code_io_data(
    io_data: List[Dict],
    target_data_len: int,
    problem_type: str,
    instruction_type: str,
    content_max_length: int,
    io_n: int,
    output_path: str,
    split: str,
    tokenizer: AutoTokenizer,
    banned_keywords: List[str],
    banned_assertion_keywords: List[str],
    weights: List[float] = None,
    enable_composite_function: bool = False,
    composite_function_n_min: int = -1,
    composite_function_n_max: int = -1,
    composite_chance: float = 0.5,
    remove_after_return: bool = False,
    num_inputs: int = 10,
    remove_input_from_snippet: bool = False,
    #include_references: bool = True,
):
    return_io_data = []
    if instruction_type.startswith('boxed'):
        instruction_template = boxed_instruction
    elif instruction_type.startswith('answer'):
        instruction_template = instruction_following
    elif instruction_type.startswith('none'):
        instruction_template = '{}'
    else:
        raise ValueError(f"Invalid instruction type: {instruction_type}")

    if weights is None:
        probabilities = [1.0 / len(io_data)] * len(io_data)
    else:
        # Normalize weights to form a probability distribution
        probabilities = [float(w)/sum(weights) for w in weights]
    
    idx = 0

    while len(return_io_data) < target_data_len:#真正的数据格式
        chosen_references = random.choice(io_data, size=min(io_n, len(io_data)), replace=False, p=probabilities)
        
        #修补
        io_prompt = instruction_template.format()
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            io_item = {
                "data_source": 'gen_' + problem_type,
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                #"problem": '',
                #"ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": '',
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'gen_' + problem_type,
                    'chosen_references': chosen_references,
                    #'composite_functions': composite_functions,
                    #'imports': imports,
                }
            }
            return_io_data.append(io_item)
            idx += 1

        if len(return_io_data) >= target_data_len:
            break

    # if io_data is not full, we sample upsample random data
    while len(return_io_data) < target_data_len:
        #修补
        io_item = io_data[random.randint(0, len(io_data))]
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)


def get_pred_code_io_data(
    io_data: List[Dict],
    target_data_len: int,
    problem_type: str,
    instruction_type: str,
    content_max_length: int,
    output_path: str,
    split: str,
    tokenizer: AutoTokenizer,
):
    return_io_data = []
    if instruction_type.startswith('boxed'):
        instruction_template = boxed_instruction
    elif instruction_type.startswith('answer'):
        instruction_template = instruction_following
    elif instruction_type.startswith('none'):
        instruction_template = '{}'
    else:
        raise ValueError(f"Invalid instruction type: {instruction_type}")

    for idx, io_item in enumerate(io_data):
        if problem_type == 'code_i':
            ground_truth = io_item['input']
        elif problem_type == 'code_o':
            ground_truth = io_item['output']
        elif problem_type == 'code_e':
            ground_truth = io_item['output']
        elif problem_type == 'code_f':
            ground_truth = io_item['snippet']
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")
        if problem_type == 'code_f':
            num_given_inputs = len(io_item['inputs']) // 2
            num_given_outputs = len(io_item['outputs']) // 2
            given_inputs = list(io_item['inputs'][:num_given_inputs])
            given_outputs = list(io_item['outputs'][:num_given_outputs])
            hidden_inputs = list(io_item['inputs'][num_given_inputs:])
            hidden_outputs = list(io_item['outputs'][num_given_outputs:])
            io_prompt = instruction_template.format(
                get_code_problem_predictor_prompt(
                    problem_type=problem_type,
                    snippet=io_item['snippet'],
                    message=io_item['message'],
                    input_output_pairs=zip(given_inputs, given_outputs),
                )
            )
        else:
            io_prompt = instruction_template.format(
                get_code_problem_predictor_prompt(#生成用户提问的提示词的另一种方式
                    problem_type=problem_type,
                    snippet=io_item['snippet'],
                    input_args=io_item['input'],
                    output=io_item['output'],
                )
            )
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            output_io_item = {
                "data_source": 'pred_' + problem_type,
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "problem": io_item['snippet'],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'pred_' + problem_type,
                    'imports': io_item['imports'],
                }
            }
            if problem_type == 'code_f': # for code_f, we need to split the inputs and outputs into given and hidden, only show part of the inputs and outputs to the model
                output_io_item['extra_info']['given_inputs'] = given_inputs
                output_io_item['extra_info']['given_outputs'] = given_outputs
                output_io_item['extra_info']['hidden_inputs'] = hidden_inputs
                output_io_item['extra_info']['hidden_outputs'] = hidden_outputs
                output_io_item['extra_info']['message'] = io_item['message']
            else:
                output_io_item['extra_info']['input'] = io_item['input']
                output_io_item['extra_info']['output'] = io_item['output']
            return_io_data.append(output_io_item)

        if len(return_io_data) >= target_data_len:
            break

    # if io_data is not full, we sample upsample random data
    while len(return_io_data) < target_data_len:
        io_item = return_io_data[random.randint(0, len(return_io_data))]
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)

