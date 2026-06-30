from openai import OpenAI
import openai
import time
import json
import random

client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",
)

MODEL = "gpt-4.1-mini"

# 统一的答案生成提示词
ANSWER_PROMPT = """You are answering a question based on the provided context. Please provide an accurate and factual answer based ONLY on the information given in the context.

Context:
{context}

Question: {question}

For the following question, reason step by step and end with "So the answer is: ...".

IMPORTANT: 
- Provide your reasoning process step by step
- After your reasoning, end with "So the answer is: X" where X is a SHORT, CONCISE answer
- The final answer should be brief (e.g., "yes", "no", a name, a date, a short phrase)
- DO NOT write a full sentence as the final answer
- Examples of good final answers: "yes", "no", "John Smith", "1995", "Paris", "American"

Please analyze the context carefully and provide your reasoning process, then give a short final answer."""


def load_progress(save_path):
    """加载已保存的进度"""
    try:
        with open(save_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_data(data, save_path):
    """保存数据"""
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def request_api(prompt, model, temperature):
    """调用OpenAI API - 适配GPT-5-mini"""
    flag = True
    retry_count = 0
    max_retries = 5

    while flag and retry_count < max_retries:
        try:
            message = [{'role': 'user', 'content': prompt}]

            # GPT-5 系列的 API 调用参数
            api_params = {
                "model": model,
                "messages": message,
                # "temperature": temperature,
                "n": 1
            }

            # 尝试使用 max_tokens (GPT-5 标准参数)
            try:
                api_params["max_tokens"] = 500
                response = client.chat.completions.create(**api_params)
            except (TypeError, openai.BadRequestError) as param_error:
                # 如果 max_tokens 不支持，尝试 max_completion_tokens
                print(f"  尝试使用 max_completion_tokens 参数...")
                api_params.pop("max_tokens", None)
                api_params["max_completion_tokens"] = 1000
                response = client.chat.completions.create(**api_params)

            text_response = response.choices[0].message.content.strip()
            flag = False
            return text_response

        except openai.RateLimitError as e:
            print(f"  速率限制超出，等待 {2 ** retry_count} 秒...")
            wait_time = 2 ** retry_count
            time.sleep(wait_time)
            retry_count += 1

        except openai.APIError as e:
            print(f"  API 错误: {e}")
            if "timeout" in str(e).lower():
                print(f"  请求超时，等待后重试...")
                time.sleep(2)
            retry_count += 1

        except Exception as e:
            print(f"  API调用错误: {type(e).__name__}: {e}")
            time.sleep(1)
            retry_count += 1

    if retry_count >= max_retries:
        raise Exception(f"API调用失败，超过最大重试次数 ({max_retries})")


def format_context(context_list):
    """格式化context为可读文本"""
    formatted_text = ""
    for title, sentences in context_list:
        formatted_text += f"\n{title}:\n"
        if isinstance(sentences, list):
            for sentence in sentences:
                formatted_text += f"  {sentence}\n"
        else:
            formatted_text += f"  {sentences}\n"
    return formatted_text


def generate_answer(question, context, temperature=0.7):
    """生成答案

    Args:
        question: 问题文本
        context: 上下文信息
        temperature: 温度参数，控制随机性

    Returns:
        生成的答案文本
    """
    context_text = format_context(context)

    prompt = ANSWER_PROMPT.format(
        question=question,
        context=context_text
    )

    answer = request_api(prompt, model=MODEL, temperature=temperature)
    return answer


def extract_final_answer(gpt_response):
    """从GPT响应中提取最终答案（"So the answer is:"之后的部分）
    并清理掉标点符号，返回简短答案
    """
    if "So the answer is:" in gpt_response:
        final_answer = gpt_response.split("So the answer is:")[-1].strip()
        # 移除末尾的句号、引号等标点符号
        final_answer = final_answer.rstrip('.,!?;:"\' ')
        # 移除开头的引号等
        final_answer = final_answer.lstrip('"\' ')
        return final_answer
    return None


def normalize_answer(answer):
    """标准化答案以便比较"""
    if answer is None:
        return ""
    # 转小写
    answer = str(answer).lower().strip()
    # 移除多余的空格
    answer = ' '.join(answer.split())
    # 移除标点符号
    import string
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    return answer


def is_answer_correct(generated_answer, correct_answer):
    """判断生成的答案是否正确

    Args:
        generated_answer: 生成的答案
        correct_answer: 正确答案

    Returns:
        bool: 是否正确
    """
    gen_norm = normalize_answer(generated_answer)
    cor_norm = normalize_answer(correct_answer)

    if not gen_norm or not cor_norm:
        return False

    # 完全匹配
    if gen_norm == cor_norm:
        return True

    # 检查是否包含关系（处理答案可能更详细的情况）
    if cor_norm in gen_norm or gen_norm in cor_norm:
        return True

    return False


def process_hotpotqa_file(input_path, output_path, start_index=0, temperature=0.7, target_samples=None):
    """处理HotpotQA验证集文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        start_index: 开始处理的索引（用于断点续传）
        temperature: 生成温度，控制随机性和错误率
        target_samples: 目标处理样本数，None表示处理全部
    """
    # 加载数据
    print(f"正在加载数据文件: {input_path}")
    print(f"使用模型: {MODEL}")
    print(f"Temperature: {temperature}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果输出文件已存在，加载进度
    processed_data = load_progress(output_path)
    if processed_data:
        start_index = len(processed_data)
        print(f"从第 {start_index} 条记录继续处理...")

    # 确保data是列表
    if not isinstance(data, list):
        data = [data]

    # 确定处理范围
    if target_samples:
        total = min(start_index + target_samples, len(data))
    else:
        total = len(data)

    print(f"将处理 {start_index} 到 {total} 条数据（共 {total - start_index} 条）")

    # 统计计数器
    correct_count = sum(1 for item in processed_data if item.get('generation_label') == 'correct')
    hallucination_count = sum(1 for item in processed_data if item.get('generation_label') == 'hallucination')

    # 记录处理时间
    start_time = time.time()

    # 处理每条数据
    for idx in range(start_index, total):
        item = data[idx].copy()  # 创建副本避免修改原数据

        print(f"\n{'=' * 80}")
        print(f"处理第 {idx + 1}/{total} 条数据")
        print(f"问题: {item['question']}")
        print(f"正确答案: {item.get('answer', 'N/A')}")

        try:
            # 生成答案
            answer = generate_answer(
                item['question'],
                item['context'],
                temperature=temperature
            )

            # 提取最终答案
            final_answer = extract_final_answer(answer)

            # 判断答案是否正确
            correct_answer = item.get('answer', '')
            is_correct = is_answer_correct(final_answer, correct_answer)

            # 根据判断结果设置标签
            if is_correct:
                label = "correct"
                correct_count += 1
            else:
                label = "hallucination"
                hallucination_count += 1

            # 添加生成的答案和标签
            item['gpt_sentence'] = answer  # 完整的推理过程+答案
            item['gpt_final_answer'] = final_answer  # 简短的最终答案
            item['generation_label'] = label  # 标签：correct 或 hallucination
            item['model_used'] = MODEL  # 记录使用的模型
            item['temperature'] = temperature  # 记录温度参数

            print(f"\n生成的完整回答:")
            print(f"{answer}")
            print(f"\n{'─' * 40}")
            print(f"提取的最终答案: [{final_answer}]")
            print(f"数据集答案: [{correct_answer}]")
            print(f"答案匹配: {'✓ 正确' if is_correct else '✗ 错误（幻觉）'}")
            print(f"标签: {label}")

            # 添加到处理后的数据中
            processed_data.append(item)

            # 每处理5条数据保存一次并显示统计
            if (idx + 1) % 5 == 0:
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / (idx + 1 - start_index)
                remaining = (total - idx - 1) * avg_time

                current_total = len(processed_data)
                correct_ratio = correct_count / current_total * 100 if current_total > 0 else 0
                hallucination_ratio = hallucination_count / current_total * 100 if current_total > 0 else 0

                print(f"\n💾 保存进度... (已处理 {idx + 1} 条)")
                print(
                    f"   当前统计: 正确 {correct_count} ({correct_ratio:.1f}%) | 幻觉 {hallucination_count} ({hallucination_ratio:.1f}%)")
                print(f"   平均耗时: {avg_time:.2f}秒/条")
                print(f"   预计剩余时间: {remaining / 60:.1f}分钟")
                save_data(processed_data, output_path)

            # 添加短暂延迟以避免API限制
            time.sleep(0.2)

        except Exception as e:
            print(f"❌ 处理第 {idx + 1} 条数据时出错: {e}")
            print(f"   错误类型: {type(e).__name__}")
            # 保存当前进度
            print("   正在保存当前进度...")
            save_data(processed_data, output_path)

            # 询问是否继续
            import traceback
            traceback.print_exc()
            print("\n将在5秒后继续处理下一条...")
            time.sleep(5)
            continue

    # 最终保存
    print(f"\n{'=' * 80}")
    print(f"处理完成！正在保存最终结果...")
    save_data(processed_data, output_path)

    # 最终统计信息
    total_time = time.time() - start_time
    current_total = len(processed_data)

    print(f"\n结果已保存到: {output_path}")
    print(f"\n📊 最终统计信息:")
    print(f"  - 总数据量: {current_total}")
    print(f"  - 正确答案: {correct_count} ({correct_count / current_total * 100:.1f}%)")
    print(f"  - 幻觉答案: {hallucination_count} ({hallucination_count / current_total * 100:.1f}%)")
    print(f"  - 总耗时: {total_time / 60:.1f} 分钟")
    print(f"  - 平均耗时: {total_time / (total - start_index):.2f} 秒/条")
    print(f"  - 使用模型: {MODEL}")
    print(f"  - Temperature: {temperature}")

    return processed_data


if __name__ == "__main__":
    # 配置输入输出路径
    input_file = "hotpot_dev_fullwiki_v1.json"
    output_file = "./hotpot_dev_with_gpt_answers_new.json"

    # 配置生成参数
    # Temperature 控制随机性：
    # - 0.0-0.3: 更确定性，更少幻觉
    # - 0.5-0.7: 平衡，自然的幻觉率
    # - 0.8-1.0: 更随机，更多幻觉
    TEMPERATURE = 0.7

    # 可选：设置只处理部分数据（用于测试）
    # TARGET_SAMPLES = 100  # 只处理100条
    TARGET_SAMPLES = None  # 处理全部数据

    print(f"""
{'=' * 80}
HotpotQA 数据集处理脚本 (自动幻觉检测版)
{'=' * 80}
配置信息:
  - 输入文件: {input_file}
  - 输出文件: {output_file}
  - 模型: {MODEL}
  - Temperature: {TEMPERATURE}
  - 处理策略: 统一生成，根据答案匹配自动判定幻觉
  - 目标样本数: {'全部' if TARGET_SAMPLES is None else TARGET_SAMPLES}
{'=' * 80}
    """)

    # 处理文件
    try:
        result = process_hotpotqa_file(
            input_file,
            output_file,
            temperature=TEMPERATURE,
            target_samples=TARGET_SAMPLES
        )
        print(f"\n✅ 成功处理了 {len(result)} 条数据")

        # 显示标签分布
        correct = sum(1 for item in result if item.get('generation_label') == 'correct')
        hallucination = sum(1 for item in result if item.get('generation_label') == 'hallucination')
        print(f"\n标签分布:")
        print(f"  正确: {correct} ({correct / len(result) * 100:.1f}%)")
        print(f"  幻觉: {hallucination} ({hallucination / len(result) * 100:.1f}%)")

    except KeyboardInterrupt:
        print("\n⏸️  用户中断，进度已保存")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback

        traceback.print_exc()