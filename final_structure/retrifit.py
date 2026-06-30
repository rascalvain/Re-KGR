"""
使用GPT-3.5对幻觉答案进行修正（Few-Shot Prompt版本）
修正gpt_sentence字段，并将结果作为新字段添加到JSON中
"""
import json
import os
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import time
import re


class HallucinationCorrector:
    """幻觉修正器（Few-Shot版本）"""

    def __init__(self, api_key, model="gpt-3.5-turbo"):
        """
        初始化
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"✓ OpenAI客户端初始化成功 (模型: {model})")

    def format_triples(self, triples_list, max_count=20):
        """将三元组列表转换为易读的字符串格式"""
        if not triples_list:
            return "None"

        formatted_lines = []
        for item in triples_list[:max_count]:  # 限制数量避免token过多
            content = item.get("triple", str(item)) if isinstance(item, dict) else str(item)
            formatted_lines.append(f"- {content}")

        return "\n".join(formatted_lines)

    def get_hallucinated_triples(self, gpt_triples, labels, threshold=1):
        """
        根据标签提取被判定为幻觉的三元组
        Args:
            gpt_triples: GPT生成的三元组列表
            labels: 对应的幻觉标签（0=事实，1=可能幻觉，2=幻觉）
            threshold: 阈值（>=threshold的标签视为幻觉）
        Returns:
            格式化的幻觉三元组字符串
        """
        hallucinations = []
        for idx, triple in enumerate(gpt_triples):
            if idx < len(labels) and labels[idx] >= threshold:
                content = triple.get("triple", str(triple)) if isinstance(triple, dict) else str(triple)
                hallucinations.append(f"- {content}")

        return "\n".join(hallucinations) if hallucinations else "None"

    def generate_correction_prompt(self, question, draft_response,
                                   context_triples, gpt_triples, labels):
        """
        构建包含Few-Shot示例的提示词
        """
        context_str = self.format_triples(context_triples)
        hallucinations_str = self.get_hallucinated_triples(gpt_triples, labels)

        prompt = f"""You are a Fact Correction Assistant. Your task is to correct a Draft Response based strictly on the provided Trusted Context Triples.
You will be given the Question, Trusted Context, the Draft Response, and a list of specific Hallucinations (errors) found in the draft.

### Rules:
1. **Remove Hallucinations**: Eliminate all information marked in the "Identified Hallucinations" list.
2. **Strict Grounding**: Only use information present in the "Trusted Context Triples".
3. **Handle Missing Info**: If the Trusted Context does not contain the answer to the question, specifically state that the information is missing. DO NOT make up external knowledge.
4. **Coherence**: Ensure the corrected response is fluent and grammatically correct.
5. **Keep Reasoning Structure**: If the draft uses step-by-step reasoning, maintain that structure but correct the facts.

---
### Example 1 (Correction Case)
**Question**: Who directed the movie 'Inception'?
**Trusted Context Triples**:
- (Inception, director, Christopher Nolan)
- (Inception, release year, 2010)
**Draft Response**: The movie Inception was directed by Steven Spielberg in 2010.
**Identified Hallucinations**:
- (Inception, director, Steven Spielberg)
**Corrected Response**: The movie Inception was directed by Christopher Nolan in 2010.

---
### Example 2 (Missing Info Case - Important!)
**Question**: What is the height of Mount Everest?
**Trusted Context Triples**:
- (Mount Everest, location, Himalayas)
- (Mount Everest, first ascent, 1953)
**Draft Response**: Mount Everest is located in the Himalayas and is 8,848 meters tall.
**Identified Hallucinations**:
- (Mount Everest, height, 8,848 meters)
**Corrected Response**: Mount Everest is located in the Himalayas. However, the provided context does not state its specific height.

---
### Example 3 (Step-by-Step Reasoning)
**Question**: What is the population of the city where Team X is based?
**Trusted Context Triples**:
- (Team X, based in, City A)
- (City A, country, Country B)
**Draft Response**: Step 1: Team X is based in City A. Step 2: According to 2020 census, City A has 500,000 people. So the answer is: 500,000
**Identified Hallucinations**:
- (City A, population, 500,000)
**Corrected Response**: Step 1: Team X is based in City A. Step 2: However, the provided context does not contain population information for City A. So the answer is: The population information is not available in the provided context.

---
### Current Task
**Question**: {question}

**Trusted Context Triples**:
{context_str}

**Draft Response**: 
{draft_response}

**Identified Hallucinations**:
{hallucinations_str}

**Corrected Response**:"""

        return prompt

    def extract_answer(self, corrected_response):
        """
        从修正后的响应中提取最终答案
        尝试匹配 "So the answer is: XXX" 或 "answer is XXX" 等模式
        """
        # 模式1: "So the answer is: XXX" 或 "The answer is: XXX"
        match = re.search(r'(?:So |The )?answer is:?\s*(.+?)(?:\.|$)', corrected_response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 模式2: 最后一句话（如果没有明确的answer标记）
        sentences = corrected_response.split('.')
        if sentences:
            last_sentence = sentences[-1].strip()
            if last_sentence:
                return last_sentence

        # 默认返回整个响应的最后部分
        return corrected_response.split('\n')[-1].strip()

    def correct_sample(self, question, draft_response,
                       context_triples, gpt_triples, labels,
                       temperature=0.0, max_tokens=400):
        """
        修正单个样本
        Returns:
            corrected_response: 修正后的完整响应
            corrected_answer: 提取的最终答案
            usage: token使用情况
        """
        try:
            # 构建prompt
            prompt = self.generate_correction_prompt(
                question, draft_response,
                context_triples, gpt_triples, labels
            )

            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that corrects factual errors based on provided knowledge graphs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0
            )

            # 提取修正后的响应
            corrected_response = response.choices[0].message.content.strip()

            # 提取最终答案
            corrected_answer = self.extract_answer(corrected_response)

            # 提取使用信息
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            return corrected_response, corrected_answer, usage

        except Exception as e:
            print(f"\n⚠️ API调用失败: {e}")
            return None, None, None


def correct_hallucinated_samples(input_json_path,
                                 output_json_path,
                                 api_key,
                                 model="gpt-3.5-turbo",
                                 hallucination_threshold=1,
                                 delay=1.0):
    """
    批量修正幻觉样本，并将结果作为新字段添加到JSON中

    Args:
        input_json_path: 包含检测结果的JSON文件
        output_json_path: 输出修正结果的JSON文件
        api_key: OpenAI API密钥
        model: 使用的模型
        hallucination_threshold: 三元组幻觉标签阈值（>=该值认为是幻觉）
        delay: API调用间隔（秒）
    """
    print(f"\n{'=' * 80}")
    print("幻觉答案修正（Few-Shot版本）")
    print(f"{'=' * 80}\n")

    # 1. 加载数据
    print(f"📊 加载检测结果...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ 加载完成: {len(data)} 条记录\n")

    # 2. 筛选需要修正的样本（幻觉样本）
    hallucination_samples = [
        (idx, item) for idx, item in enumerate(data)
        if item.get('hallucination_prediction') == 1
    ]

    print(f"🔍 筛选幻觉样本:")
    print(f"   总样本: {len(data)}")
    print(f"   幻觉样本: {len(hallucination_samples)}")
    print(f"   事实样本: {len(data) - len(hallucination_samples)}\n")

    if len(hallucination_samples) == 0:
        print("✓ 没有需要修正的幻觉样本")
        # 仍然保存原始数据
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    # 3. 初始化修正器
    corrector = HallucinationCorrector(api_key=api_key, model=model)
    print()

    # 4. 批量修正
    print(f"{'=' * 80}")
    print("开始修正...")
    print(f"{'=' * 80}\n")

    total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    success_count = 0
    fail_count = 0

    # 创建输出数据（从原始数据复制）
    output_data = [item.copy() for item in data]

    for idx, item in tqdm(hallucination_samples, desc="修正进度"):
        try:
            # 提取信息
            question = item.get('question', '')
            draft_response = item.get('gpt_sentence', '')
            context_triples = item.get('context_triples', [])
            gpt_triples = item.get('gpt_sentence_triples', [])
            labels = item.get('triple_hallucination_labels', [])

            # 调用API修正
            corrected_response, corrected_answer, usage = corrector.correct_sample(
                question=question,
                draft_response=draft_response,
                context_triples=context_triples,
                gpt_triples=gpt_triples,
                labels=labels
            )

            if corrected_response is not None:
                # 🔥 添加新字段到输出数据
                output_data[idx]['corrected_gpt_sentence'] = corrected_response
                output_data[idx]['corrected_answer'] = corrected_answer
                output_data[idx]['correction_metadata'] = {
                    'corrected': True,
                    'token_usage': usage,
                    'model': model,
                    'timestamp': datetime.now().isoformat()
                }

                # 累计token使用
                if usage:
                    total_usage['prompt_tokens'] += usage['prompt_tokens']
                    total_usage['completion_tokens'] += usage['completion_tokens']
                    total_usage['total_tokens'] += usage['total_tokens']

                success_count += 1
            else:
                # 修正失败，标记但保留原始数据
                output_data[idx]['corrected_gpt_sentence'] = None
                output_data[idx]['corrected_answer'] = None
                output_data[idx]['correction_metadata'] = {
                    'corrected': False,
                    'error': 'API call failed'
                }
                fail_count += 1

            # 延迟，避免超过rate limit
            time.sleep(delay)

        except Exception as e:
            print(f"\n⚠️ 处理样本 {item.get('_id', 'unknown')} 时出错: {e}")
            output_data[idx]['corrected_gpt_sentence'] = None
            output_data[idx]['corrected_answer'] = None
            output_data[idx]['correction_metadata'] = {
                'corrected': False,
                'error': str(e)
            }
            fail_count += 1
            continue

    # 5. 保存结果
    print(f"\n{'=' * 80}")
    print("保存修正结果...")
    print(f"{'=' * 80}\n")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 6. 统计
    print(f"✓ 修正完成！")
    print(f"\n统计信息:")
    print(f"  总样本数: {len(data)}")
    print(f"  幻觉样本数: {len(hallucination_samples)}")
    print(f"  成功修正: {success_count}")
    print(f"  失败: {fail_count}")

    print(f"\nToken使用情况:")
    print(f"  Prompt tokens: {total_usage['prompt_tokens']:,}")
    print(f"  Completion tokens: {total_usage['completion_tokens']:,}")
    print(f"  总计: {total_usage['total_tokens']:,}")

    # 估算费用
    cost = (total_usage['prompt_tokens'] * 0.0015 +
            total_usage['completion_tokens'] * 0.002) / 1000
    print(f"  估算费用: ${cost:.4f}")

    # 7. 显示示例
    corrected_samples = [(idx, output_data[idx]) for idx, _ in hallucination_samples
                         if output_data[idx].get('corrected_gpt_sentence')]

    if corrected_samples:
        print(f"\n{'=' * 80}")
        print("修正示例（前3个）")
        print(f"{'=' * 80}\n")

        for i, (idx, item) in enumerate(corrected_samples[:3], 1):
            print(f"【示例 {i}】")
            print(f"ID: {item.get('_id', 'N/A')}")
            print(f"问题: {item.get('question', 'N/A')[:80]}...")

            print(f"\n原始响应:")
            print(f"  {item.get('gpt_sentence', '')[:150]}...")

            print(f"\n修正后响应:")
            print(f"  {item.get('corrected_gpt_sentence', '')[:150]}...")

            print(f"\n修正后答案:")
            print(f"  {item.get('corrected_answer', '')}")

            metadata = item.get('correction_metadata', {})
            if 'token_usage' in metadata:
                print(f"\nToken使用: {metadata['token_usage']['total_tokens']}")
            print()

    print(f"{'=' * 80}")
    print(f"✓ 输出文件: {output_json_path}")
    print(f"  - 包含全部 {len(output_data)} 条记录")
    print(f"  - 其中 {success_count} 条已添加修正字段")
    print(f"{'=' * 80}\n")

    return output_data


def main():
    """主函数"""

    # 🔥 配置
    # OpenAI API密钥
    API_KEY = "sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN"  # 🔥 请替换为你的API密钥

    # 或者从环境变量读取
    # API_KEY = os.environ.get('OPENAI_API_KEY')

    if not API_KEY or API_KEY == "your-openai-api-key-here":
        print("❌ 错误: 请设置有效的 OpenAI API 密钥！")
        print("   方式1: 修改代码中的 API_KEY 变量")
        print("   方式2: 设置环境变量 export OPENAI_API_KEY='your-key'")
        return

    # 输入文件（带有检测结果的JSON）
    input_json_path = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/rgat_output/sampled_50_with_detection_20260119_110409.json'

    # 输出文件
    output_dir = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/rgat_output'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_json_path = os.path.join(
        output_dir,
        f'sampled_50_with_correction_{timestamp}.json'
    )

    device = 'cuda' if os.path.exists('/usr/bin/nvidia-smi') else 'cpu'
    print(f"运行环境: {device}\n")

    # 🎯 运行修正
    output_data = correct_hallucinated_samples(
        input_json_path=input_json_path,
        output_json_path=output_json_path,
        api_key=API_KEY,
        model="gpt-3.5-turbo",
        hallucination_threshold=1,  # >=1的标签（可能幻觉+幻觉）
        delay=1.0  # API调用间隔
    )

    print(f"\n🎉 全部完成！")
    if output_data:
        corrected_count = sum(1 for item in output_data
                              if item.get('corrected_gpt_sentence') is not None)
        print(f"   已修正 {corrected_count} 个幻觉样本")


if __name__ == '__main__':
    main()