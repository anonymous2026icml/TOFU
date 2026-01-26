from datasets import load_dataset
from data import prompt_formats

class FormatAlpaca:
    def __init__(self, split, tokenizer):
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("tatsu-lab/alpaca", split=self.split)

        self.data = self.data.filter(lambda x: x["output"].strip() != "")
        self.data = self.data.filter(lambda x: x["instruction"].strip() != "")

    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(x["instruction"].strip(), x["input"].strip()),
        "completion": prompt_formats.ANSWER.format(x["output"].strip()) + self.tokenizer.eos_token
        },
        remove_columns=["instruction", "input", "output", "text"])
        
        return data
    

class FormatUltraFeedback:
    def __init__(self, split, tokenizer, thresh_score=7):
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("openbmb/UltraFeedback", split=self.split)
        self.thresh_score = thresh_score
        
        self.data = self.data.filter(lambda x: len(x["completions"]) > 0)
        self.data = self.data.filter(lambda x: self.extract_best_score(x) >= self.thresh_score)

    def extract_best_score(self, sample):
        return max(r["overall_score"] for r in sample["completions"])

    def extract_best_answer(self, sample):
        return max(sample["completions"], key=lambda r: r["overall_score"])["response"]
    
    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(x["instruction"].strip(), ""),
        "completion": prompt_formats.ANSWER.format(self.extract_best_answer(x).strip()) + self.tokenizer.eos_token
        },
        remove_columns=["source", "instruction", "models", "completions", "correct_answers", "incorrect_answers"])

        return data

        
    
class FormatMMLU:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("cais/mmlu", "all", split=self.split)
    
    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(prompt_formats.build_mmlu_prompt(x["question"].strip(), x["choices"]), ""),
        "correct_response": "ABCD"[x["answer"]]
        },
        remove_columns=["subject", "choices", "answer", "question"])

        return data
    

class FormatARC:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=self.split)
    
    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(prompt_formats.build_mmlu_prompt(x["question"].strip(), x["choices"]["text"], x["choices"]["label"]), ""),
        "correct_response": x["answerKey"]
        },
        remove_columns=["id", "choices", "answerKey", "question"])

        return data


class FormatShortStories:
    def __init__(self, split):
        ValueError(f"Will be released upon publication")


class FormatSmallPrompts():
    def __init__(self, split):
        ValueError(f"Will be released upon publication")


class FormatNoveltyBench:
    def __init__(self, split):
        self.split = split
        
        self.data = load_dataset("yimingzhang/novelty-bench", split=self.split)

        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(x["prompt"].strip(), "")
        },
        remove_columns=["id"])
        
        return data
    
    @property
    def base_format(self):

        data = self.data.map(lambda x: {
        "prompt": x["prompt"].strip()
        },
        remove_columns=["id"])
        
        return data


class FormatEval:
    def __init__(self, data):
        self.data = data

    def clean_instruction(self, prompt):
        if "<|instruct|>" in prompt and "<|/instruct|>" in prompt:
            instruction = prompt.split("<|instruct|>")[1].split("<|/instruct|>")[0].strip()
            return instruction
        return prompt

    def clean_response(self, response):
        if "<|response|>" in response:
            response = response.split("<|response|>")[1]
        if "<|/response|>" in response:
            response = response.split("<|/response|>")[0]
        return response.strip()

    @property
    def judge_instruct_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.JUDGE_INSTRUCT.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data
    
    @property
    def bleu_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            formatted_data.append({'instruction': cleaned_instruction, 'responses': cleaned_responses})
        return formatted_data
    

    @property
    def judge_story_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.JUDGE_STORY.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data

    
    @property
    def judge_poem_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.JUDGE_POEM.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data
    


class FormatOpenCode:
    def __init__(self, split, tokenizer):
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("nvidia/OpenCodeInstruct", split=self.split)

        self.data = self.data.filter(lambda x: x["output"].strip() != "")
        self.data = self.data.filter(lambda x: x["input"].strip() != "")


    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(x["input"].strip(), ""),
        "completion": prompt_formats.ANSWER.format(x["output"].strip()) + self.tokenizer.eos_token
        },
        remove_columns=["id", "domain", "generation_algorithm", "llm_judgement", "unit_tests", "tests_execution_status", "average_test_score"])

        return data
    

class FormatTruthfulQA:
    def __init__(self, split, tokenizer):
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("domenicrosati/TruthfulQA", split=self.split)

        self.data = self.data.filter(lambda x: x["Best Answer"].strip() != "")
        self.data = self.data.filter(lambda x: x["Question"].strip() != "")


    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(x["Question"].strip(), ""),
        "completion": prompt_formats.ANSWER.format(x["Best Answer"].strip()) + self.tokenizer.eos_token
        },
        remove_columns=["Type", "Category", "Correct Answers", "Incorrect Answers", "Source", "Question", "Best Answer"])

        return data
