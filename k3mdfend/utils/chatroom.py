import os
from openai import OpenAI
import pickle
import pandas as pd
import json
def text_extract(json_str):
    json_dict = json.loads(json_str)
    # # 提取 discussion 的值
    discussion = json_dict['discussion']
    return discussion
class CommentGenerator:
    def __init__(self, source_text, prompts_folder, dataset):
        self.source_text = source_text
        self.prompts_folder = prompts_folder
        self.dataset = dataset
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        if self.dataset == 'ch1':
            self.prompts_folder = 'prompts/ch1/'
        else:
            self.prompts_folder = 'prompts/en/'

    def load_prompt(self, file_path):
        """加载指定路径的.md文件"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def first_chat(self, prompt, text, max_tokens=1000, temperature=0.1):
        """调用 Qwen-max 对给定的文本和提示词生成评论"""
        prompt_with_text = prompt.replace("{TEXT}", text)
        # prompt_with_text = prompt.apply(
        #     lambda x: x.replace("{TEXT}", text))
        completion = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[{'role': 'user', 'content': prompt_with_text}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def second_chat(self, prompt, text, comment, max_tokens=1000, temperature=0.1):
        """调用 Qwen-max 对给定的文本和提示词生成评论"""
        prompt_with_text = prompt.replace("{TEXT}", text).replace("{COMMENT TEXT}", comment)
        # 对每个元素进行替换操作
        # prompt_with_text = prompt.apply(
        #     lambda x: x.replace("{TEXT}", text).replace("{COMMENT TEXT}", comment))

        completion = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[{'role': 'user', 'content': prompt_with_text}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def generate_comments(self):
        file_path = 'data/ch1/train_update.pkl'
        if os.path.exists(file_path):
            print('Loading the existing data...')
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            # 1. 加载所有的提示词
            prompts = {}
            prompt_files = [
                'decision.md',
                'expert.md', 'user.md', 'reporter.md',
                'expert_to_reporter.md', 'expert_to_user.md',
                'user_to_expert.md', 'user_to_reporter.md'
            ]

            for file_name in prompt_files:
                file_path = os.path.join(self.prompts_folder, file_name)
                prompts[file_name] = self.load_prompt(file_path)

            # 2. 执行 Qwen-Plus 生成各角色的评论
            results , data = {}, {}
            results['expert'], data['expert'] = [], []
            results['user'], data['user'] = [], []
            results['reporter'], data['reporter'] = [], []
            results['expert_to_reporter'], data['expert_to_reporter'] = [], []
            results['expert_to_user'], data['expert_to_user'] = [], []
            results['user_to_expert'], data['user_to_expert'] = [], []
            results['user_to_reporter'], data['user_to_reporter'] = [], []

            output_dir = 'data/ch1/json'
            os.makedirs(output_dir, exist_ok=True)

            # for idx, line in enumerate(self.source_text):
            # 1. expert.md
            results['expert'].append(self.first_chat(prompts['expert.md'], self.source_text))
            expert_text = text_extract(results['expert'][0])
            # 2. user.md
            results['user'].append(self.first_chat(prompts['user.md'], self.source_text))
            print(results['user'])
            user_text = text_extract(results['user'][0])
            # 3. reporter.md
            results['reporter'].append(self.first_chat(prompts['reporter.md'], self.source_text))
            reporter_text = text_extract(results['reporter'][0])
            # 4. expert_to_reporter
            results['expert_to_reporter'].append(self.second_chat(prompts['expert_to_reporter.md'],
                                                              self.source_text, reporter_text))
            expert_to_reporter_text = text_extract(results['expert_to_reporter'][0])
            # 5. expert_to_user
            results['expert_to_user'].append(self.second_chat(prompts['expert_to_user.md'], self.source_text, user_text))
            expert_to_user_text = text_extract(results['expert_to_user'][0])
            # 6. user_to_expert
            results['user_to_expert'].append(self.second_chat(prompts['user_to_expert.md'], self.source_text, expert_text))
            user_to_expert_text = text_extract(results['user_to_expert'][0])
            # 7. user_to_reporter
            results['user_to_reporter'].append(self.second_chat(prompts['user_to_reporter.md'], self.source_text, reporter_text))
            user_to_reporter_text = text_extract(results['user_to_reporter'][0])

            data['expert'].append(expert_text)
            data['user'].append(user_text)
            data['reporter'].append(reporter_text)
            data['expert_to_reporter'].append(expert_to_reporter_text)
            data['expert_to_user'].append(expert_to_user_text)
            data['user_to_expert'].append(user_to_expert_text)
            data['user_to_reporter'].append(user_to_reporter_text)

            return data
            # # Load the existing data from the pickle file
            # with open('data/ch1/train.pkl', 'rb') as f:
            #     data = pickle.load(f)
            #
            # # Convert the data to a DataFrame
            # df = pd.DataFrame(data)
            #
            # # Add the new columns
            # df['expert'] = results['expert']
            # df['user'] = results['user']
            # df['reporter'] = results['reporter']
            # df['expert_to_reporter'] = results['expert_to_reporter']
            # df['expert_to_user'] = results['expert_to_user']
            # df['user_to_expert'] = results['user_to_expert']
            # df['user_to_reporter'] = results['user_to_reporter']
            #
            # # Save the updated DataFrame to a new pickle file
            # with open('data/ch1/train_update.pkl', 'wb') as f:
            #     pickle.dump(df, f)
            # with open('data/ch1/train_update.pkl', 'rb') as f:
            #     data = pickle.load(f)
            #     print(data)
            # return data
        # # 3. 保存所有结果
        # output_folder = 'output'
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)
        #
        # for role, comment in results.items():
        #     with open(os.path.join(output_folder, f'{role}_comment.txt'), 'w', encoding='utf-8') as file:
        #         file.write(comment)

