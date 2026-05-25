import os
from functools import partial
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import re
import uuid
from functools import partial

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.protocol import DataProtoItem
from verl.utils.dataset.rl_dataset import collate_fn
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

import absolute_zero_reasoner.rewards.custom_evaluate as custom_evaluate
from rewards.code_reward import (
    parse_inputs_message,
    parse_code_function,
    ast_edit_distance,
    get_code_complexity_reward,
    get_halstead_reward,
    get_type_counts_reward,
)
from rewards.custom_evaluate import get_format_reward, extract_thought, extract_answer, get_solution_format_reward
from data_construction.process_data import boxed_instruction, instruction_following
from data_construction.constructor import get_code_problem_predictor_prompt
from utils.dataset.rl_dataset import RLHFDataset
from utils.logging_utils.stdout import PrettyPrinter
from utils.code_utils.checks import check_composite_function


class ORRewardManager():
    """The reward manager."""
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        #num_examine: int,#0 or 1
        split: str,#'train'
        reward_fn_extraction_type: str,#answer_conditional
        math_metric: str,#math_verify
        splitter: str,#"Assistant:"
        output_path: str,#checkpoints/operation/azr/azr_7b/test_answer/Qwen2.5-7B/answer_conditional
        generation_reward_config: Dict[str, Any],
        max_prompt_length: int = 8192,
        valid_program_filter: str = 'all',
        num_inputs: int = 10,
        code_f_reward_type: str = 'accuracy',#binary
        boxed_retry: bool = False,
    ):
        self.tokenizer = tokenizer
        #self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = partial(custom_evaluate.get_reward, math_metric=math_metric, boxed_retry=boxed_retry)
        self.reward_fn_extraction_type = reward_fn_extraction_type
        self.split = split
        self.splitter = splitter
        self.output_path = output_path
        self.max_prompt_length = max_prompt_length
        self.generation_reward_config = generation_reward_config
        self.valid_program_filter = valid_program_filter
        self.use_original_code_as_ref = generation_reward_config.use_original_code_as_ref#True
        self.num_inputs = num_inputs
        self.code_f_reward_type = code_f_reward_type
        self.boxed_retry = boxed_retry

    @staticmethod
    def extract_solution_elements(self, extracted_content: str):
        thought = extract_thought(extracted_content)
        if self.generation_reward_config.format_reward:
            format_score, answer = get_solution_format_reward(
                solution_str=extracted_content, 
                extraction_type=self.reward_fn_extraction_type
            )
        else:
            format_score = 1
            _, answer = {"formula":extracted_content, "code":extracted_content}
        return thought, answer, format_score

    def _get_data_dict(self, data_item: DataProtoItem, problem_type: str, uid: str) -> Dict:
        prompt_ids = data_item.batch['prompts']#一维的
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]#左padding
        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]#右padding
        # decode
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences)
        
        data_source = data_item.non_tensor_batch['data_source']#数据集的名字吧
        extra_info = data_item.non_tensor_batch['extra_info']#数据的类型或者领域
        non_special_tokens_sequences_str = self.tokenizer.decode(self.tokenizer.encode(sequences_str), skip_special_tokens=True)
        generation = non_special_tokens_sequences_str.split(self.splitter)[1].strip().strip('\"\'')#模型response全部内容
        data_dict = {
            'generation': generation,#生成结果
            'data_source': data_source,
            'extra_info': extra_info,
            #'non_special_tokens_sequences_str': non_special_tokens_sequences_str,
            'valid_response_length': valid_response_length,
            #'thought': thought,
            'uid': uid}
        # first go through, we only checking the format
        if problem_type.startswith('gen'):
            if 'operation' in problem_type:
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                if format_score:
                    data_dict['thought'] = extract_thought(generation)
                    data_dict['answer'] = extract_answer(solution_str=generation)
                else:
                    data_dict['thought'] = extract_thought(generation)
                    data_dict['answer'] = generation
                return data_dict
            else:
                #TODO
                pass
        # if prediction is the task
        elif problem_type.startswith('pred'):
            # Check required blocks
            if problem_type.endswith('operation'): # parse input
                thought, answer, format_score = self.extract_solution_elements(generation)
                data_dict['thought'] = thought
                if not format_score:
                    data_dict['format_score'] = 0.
                    data_dict['answer'] = generation
                    return data_dict
                data_dict['format_score'] = format_score
                data_dict['answer'] = generation
                data_dict['answer_elements'] = answer 
                return data_dict
            elif problem_type.endswith('negotiation'):
                #TODO
                pass
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")

    def __call__(
        self,
        data: DataProto,
        problem_type: str = None,#pred_operation or gen_operation
        executor = None,
        rollout_actor_wg = None,
        banned_words: List[str] = [],
        banned_assertion_keywords: List[str] = [],#被禁止的断言关键词
        n_samples: int = 1,
        input_type_counters: Dict[str, Dict[str, int]] = None
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        """We will expand this function gradually based on the available datasets"""
        #修补
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)# shape: [2, 3]  2个回答，每个回答3个token
        all_scores = defaultdict(list)
        data_dicts = []
        valid_programs = [] # for gen tasks, we need to store the valid programs for later use, ignore this if prediction task
        correct_predictions = []
        uids = np.array([str(uuid.uuid4()) for _ in range(len(data))], dtype=object)#batch size
        
        if problem_type is None:
            problem_types = [d.non_tensor_batch['extra_info']['metric'] for d in data]
            problem_type = 'pred' # dummy set
        else:
            problem_types = [problem_type] * len(data)#每一条数据对应一个problem type
        
        PrettyPrinter.section_header("Getting Data Dicts")
        for i in range(len(data)): # get format score
            #对每一条数据产出的结果进行计算format score并且抽取出回复内容
            data_dict = self._get_data_dict(data[i], problem_types[i], uids[i])
            data_dicts.append(data_dict)

        if problem_type.startswith('gen') and rollout_actor_wg is not None: # get generation rewards
            PrettyPrinter.section_header("Generating Rewards for Generation Tasks")
            #修补
            rewards, valid_programs = self._get_problem_generator_rewards_and_valid_programs(
                data_dicts=data_dicts,
                problem_type=problem_type,
                n_samples=n_samples,
                rollout_actor_wg=rollout_actor_wg,
                executor=executor,
                input_type_counters=input_type_counters
            )
            PrettyPrinter.section_header("Combining Rewards for Generation Tasks")
            for i in range(len(data_dicts)):
                uid = data_dicts[i]['uid']
                valid_response_length = data_dicts[i]['valid_response_length']
                acc_reward = rewards[uid]['accuracy']
                format_reward = data_dicts[i]['format_score']
                if format_reward > 0:
                    if acc_reward > 0:
                        # Helper function for safe reward combination
                        def _combine_rewards(acc, intrinsic_components, method):
                            components = [c for c in intrinsic_components if c is not None]

                            if method == 'sum':
                                return acc + sum(components) if components else acc
                            elif method == 'multiply':
                                return acc * np.prod([c for c in components]) if components else acc
                            elif method == 'sum_multiply':
                                return acc + np.prod([c for c in components]) if components else acc
                            elif method == 'multiply_sum':
                                return acc * sum(components) if components else acc
                            else:
                                raise ValueError(f"Unknown combination method: {method}")

                        intrinsic_reward_components = []
                        #修补
                        if problem_type.endswith('operation'):
                            if self.generation_reward_config.f_input_answer_diversity_reward.enabled:#False
                                intrinsic_reward_components.append(min(self.generation_reward_config.f_input_answer_diversity_reward.coef * rewards[uid]['input_type_counts'],
                                    self.generation_reward_config.f_input_answer_diversity_reward.max))
                            if self.generation_reward_config.f_output_answer_diversity_reward.enabled:#False
                                intrinsic_reward_components.append(min(self.generation_reward_config.f_output_answer_diversity_reward.coef * rewards[uid]['output_type_counts'],
                                    self.generation_reward_config.f_output_answer_diversity_reward.max))
                        else:
                            if self.generation_reward_config.complexity_reward.enabled:#False
                                intrinsic_reward_components.append(min(self.generation_reward_config.complexity_reward.coef * rewards[uid]['complexity'],
                                    self.generation_reward_config.complexity_reward.max))
                            if self.generation_reward_config.mean_edit_distance_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.mean_edit_distance_reward.coef * rewards[uid]['mean_edit_distance'],
                                    self.generation_reward_config.mean_edit_distance_reward.max))
                            if self.generation_reward_config.halstead_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.halstead_reward.coef * rewards[uid]['halstead'],
                                    self.generation_reward_config.halstead_reward.max))
                            if self.generation_reward_config.answer_diversity_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.answer_diversity_reward.coef * rewards[uid]['type_counts'],
                                    self.generation_reward_config.answer_diversity_reward.max))

                        final_reward = _combine_rewards(acc_reward, intrinsic_reward_components, self.generation_reward_config.intrinsic_combine_method)#sum
                        reward_tensor[i, valid_response_length - 1] = final_reward
                    else:
                        reward_tensor[i, valid_response_length - 1] = -0.5
                else:
                    reward_tensor[i, valid_response_length - 1] = -1.0
            all_scores['accuracy'] = [rewards[uid]['accuracy'] for uid in rewards]
            all_scores['format_score'] = [data_dicts[i]['format_score'] for i in range(len(data))]
            #修补
            if 'code_f' not in problem_type:
                all_scores['answer_diversity'] = [rewards[uid]['type_counts'] for uid in rewards]
                all_scores['complexity'] = [rewards[uid]['complexity'] for uid in rewards]
                all_scores['mean_edit_distance'] = [rewards[uid]['mean_edit_distance'] for uid in rewards]
                all_scores['halstead'] = [rewards[uid]['halstead'] for uid in rewards]
            else:
                all_scores['input_answer_diversity'] = [rewards[uid]['input_type_counts'] for uid in rewards]
                all_scores['output_answer_diversity'] = [rewards[uid]['output_type_counts'] for uid in rewards]
        elif problem_type.startswith('pred'): # get prediction rewards
            PrettyPrinter.section_header("Getting Prediction Rewards")
            all_scores['none_count'] = 0
            #TODO process verification
            acc_rewards = []
            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                if not data_dicts[i]['format_score']:#五元素+代码格式 
                    acc_reward = 0.
                elif problem_types[i].endswith('operation'):
                    #修补
                    #acc_reward = executor.eval_prediction(scenario = scenario, formula = formula, code=program, agent_input=answer, imports=list(set(imports)))
                    acc_reward = executor.eval_prediction()
                    if acc_reward is None:
                        all_scores['none_count'] += 1
                        acc_reward = 0.
                        print(f"error in pred_operation, not in [0, 1], formula={data_dict['answer_elements']['formula']}\nprogram:\n{data_dict['answer_elements']['code']}\n---\n")
                    if acc_reward > 0.0:
                        correct_predictions.append(data_dict)
                elif problem_types[i].endswith('negotaition'):
                    #TODO
                    pass
                else:
                    raise ValueError(f"Invalid problem type: {problem_types[i]}")

                if self.split == 'train':
                    if data_dicts[i]['format_score'] > 0:
                        if acc_reward > 0:
                            reward_tensor[i, valid_response_length - 1] = acc_reward
                        else:
                            reward_tensor[i, valid_response_length - 1] = -0.5
                    else:
                        reward_tensor[i, valid_response_length - 1] = -1.0
                elif self.split == 'test': # only acc reward for eval
                    if acc_reward > 0:
                        reward_tensor[i, valid_response_length - 1] = 1.0
                    else:
                        reward_tensor[i, valid_response_length - 1] = 0.0
                acc_rewards.append(acc_reward)
            all_scores['accuracy'] = acc_rewards
            all_scores['format_score'] = [data_dicts[i]['format_score'] for i in range(len(data))]
            all_scores['none_ratio'] = all_scores['none_count'] / len(data)#scalar
        return reward_tensor, all_scores, valid_programs, correct_predictions

    def _get_problem_generator_rewards_and_valid_programs(
        self,
        data_dicts: List[Dict],
        problem_type: str,
        n_samples: int,
        rollout_actor_wg,
        executor,
        type_counters: Dict[str, Dict[str, int]] = None,#修补
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, str]]]:
        """This function uses samples to estimate the accuracy reward for each program, also computes the code complexity and mean edit distance of generated programs.
            Also returns the valid programs using filters.
            Args:
                data_dicts: List[Dict]: A list of data dictionaries.
                problem_type: str: The type of problem.
                n_samples: int: The number of samples to use.
                rollout_actor_wg: RolloutActorWG: The rollout actor.
                executor: PythonExecutor/CodeBoxExecutor: The executor.
                type_counters: Dict[str, Dict[str, int]]: The type counters.
            Returns:
               rewards: Dict[str, Dict[str, float]]: A dictionary of rewards for each program.
               valid_programs: List[Dict[str, str]]: A list of valid programs.
        """

        valid_data_dicts = [data_dict for data_dict in data_dicts if data_dict['format_score']]
        uid2valid_dict_idx = {data_dict['uid']: i for i, data_dict in enumerate(valid_data_dicts)}
        valid_uids = [data_dict['uid'] for data_dict in data_dicts if data_dict['format_score']]
        invalid_uids = [data_dict['uid'] for data_dict in data_dicts if not data_dict['format_score']]
        assert len(valid_uids) + len(invalid_uids) == len(data_dicts)
        accuracies = {uid: 0.0 for uid in invalid_uids}

        rewards = defaultdict(dict)
        valid_programs = []
        if len(valid_uids) > 0:
            if self.reward_fn_extraction_type.startswith('boxed'):
                instruction_template = boxed_instruction
            elif self.reward_fn_extraction_type.startswith('answer'):
                instruction_template = instruction_following
            elif self.reward_fn_extraction_type.startswith('none'):
                instruction_template = '{}'
            else:
                raise ValueError(f"Invalid instruction type: {self.reward_fn_extraction_type}")
            prompts = []
            if problem_type.endswith('operation'):
                pt = 'operation'
            elif problem_type.endswith('negotiation'):
                pt = 'negotiation'
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            for data_dict in valid_data_dicts:
                if pt == 'operation':
                    #修补
                    io_prompt = ""
                else:
                    #TODO
                    pass
                prompts_dict = {
                    'prompt': [{'role': 'user', 'content': io_prompt}],
                    'uid': data_dict['uid']
                }
                prompts.append(prompts_dict)

            # sampling to estimate the accuracy
            PrettyPrinter.section_header("Sampling to Estimate Accuracy")
            prompts = prompts * n_samples # repeat the prompts n_samples times
            pd.DataFrame(prompts).to_parquet(f'{self.output_path}/temp.parquet') # RLHFDataset expects parquet
            
            temp_data = RLHFDataset(
                parquet_files=f'{self.output_path}/temp.parquet',
                tokenizer=self.tokenizer,
                prompt_key='prompt',
                max_prompt_length=self.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=False,
                truncation='error'
            )
            os.remove(f'{self.output_path}/temp.parquet') # we do not need this file after we load in the dataset
            sampler = torch.utils.data.SequentialSampler(data_source=temp_data)

            dataloader = torch.utils.data.DataLoader(
                dataset=temp_data,
                batch_size=len(temp_data),
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            assert len(dataloader) == 1
            data = next(iter(dataloader))
            batch = DataProto.from_single_dict(data)
            gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': True,
                'validate': False,
            }
            # pad to be divisible by dp_size
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
            output_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
            # unpad
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            batch = batch.union(output_gen_batch)
            batched_responses = []
            for b in batch:
                #修补
                batch_dict = {
                        'extracted_answers': extract_answer(
                            self.tokenizer.decode(b.batch['responses'], skip_special_tokens=True),
                            self.reward_fn_extraction_type,
                            boxed_retry=self.boxed_retry,
                        ),
                        'uid': b.non_tensor_batch['uid'],
                        'problem': b.non_tensor_batch['problem'],
                        'data_source': b.non_tensor_batch['data_source'],
                        'extra_info': b.non_tensor_batch['extra_info'],
                        'program': b.non_tensor_batch['program'],
                        'references': b.non_tensor_batch['references'],
                        'imports': b.non_tensor_batch['imports'],
                    }
                if pt == 'code_f':
                    batch_dict.update({
                        'given_inputs': b.non_tensor_batch['given_inputs'],
                        'given_outputs': b.non_tensor_batch['given_outputs'],
                        'hidden_inputs': b.non_tensor_batch['hidden_inputs'],
                        'hidden_outputs': b.non_tensor_batch['hidden_outputs'],
                        'message': b.non_tensor_batch['message'],
                    })
                else:
                    batch_dict.update({
                        'input': b.non_tensor_batch['input'],
                        'output': b.non_tensor_batch['output'],
                        'original_program': b.non_tensor_batch['original_program'],
                        'composite_functions': b.non_tensor_batch['composite_functions'].tolist(),
                    })
                batched_responses.append(batch_dict)
            df = pd.DataFrame(batched_responses)

            # estimating accuracy using python executor
            PrettyPrinter.section_header("Estimating Accuracy Using Python Executor")
            for valid_uid in valid_uids:
                df_valid = df[df['uid'] == valid_uid]
                if df_valid.empty: # the prompt got filtered out TODO: check
                    accuracies[valid_uid] = 0.0
                    continue
                
                if pt == 'operation':
                    #修补
                    answers = [self.extract_input_output(
                        answer,
                        return_type=problem_type.endswith('operation'),
                    ) for answer in df_valid['extracted_answers'].tolist()]#要算出结果
                else:
                    #TODO
                    pass
                
                answer_cache = {} # for the same uid, the answer is the same and the program is assumed to be deterministic, therefore we cache the answer -> accuracy mapping
                problem_accuracies = []
                if problem_type.endswith('operation'):
                    #修补
                    for answer in answers:
                        acc_reward = 0.0
                        if acc_reward is not None:
                            problem_accuracies.append(acc_reward)
                            answer_cache[answer] = acc_reward
                            
                accuracies[valid_uid] = sum(problem_accuracies) / len(problem_accuracies) if problem_accuracies else 0.0

                # filtering valid programs
                #修补
                #中等难度的保留 non_extremes
                if self.valid_program_filter == 'all':
                    valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])#proposed problem
                elif self.valid_program_filter == 'non_one':
                    if accuracies[valid_uid] < 1.0:
                        valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                elif self.valid_program_filter == 'non_extremes':
                    if accuracies[valid_uid] > 0.0 and accuracies[valid_uid] < 1.0:
                        valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                else:
                    raise ValueError(f"Invalid valid program filter: {self.valid_program_filter}")

        # getting other rewards
        PrettyPrinter.section_header("Getting Other Rewards")
        # outputting rewards
        #修补
        for d in data_dicts:
            uid = d['uid']
            #修补 中等难度的reward概率转换
            if self.generation_reward_config.generation_accuracy_convertion == 'one_minus':
                rewards[uid]['accuracy'] = (1 - accuracies[uid]) if accuracies[uid] > 0 else 0.0
            elif self.generation_reward_config.generation_accuracy_convertion == 'inverse':
                rewards[uid]['accuracy'] = 1 - accuracies[uid]
            else:
                raise ValueError(f"Invalid generation accuracy convertion: {self.generation_reward_config.generation_accuracy_convertion}")
            rewards[data_dict['uid']]['type_counts'] = get_type_counts_reward(
                data_dict['answer'][type_counter_key],
                type_counters,
                hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
            ) if 'answer' in data_dict else 0.0
            
        if self.debug:
            for data_dict in data_dicts:
                if 'answer' in data_dict:
                    continue

        rewards = dict(rewards)
        return rewards, valid_programs
