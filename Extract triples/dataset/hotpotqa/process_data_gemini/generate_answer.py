from google import genai
import time
import json
import random

# ================= Gemini API 配置 =================
client = genai.Client(
    api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",
    http_options={"base_url": "https://api.openai-proxy.org/google"}
)

MODEL = "gemini-2.5-flash-lite-preview-09-2025"

# ================= 生成参数配置 =================
MAX_OUTPUT_TOKENS = 400  # 最大输出token数，可根据需要调整：500, 1000, 2000, 4000等
TEMPERATURE = 0.7  # 温度参数
API_DELAY = 0.5  # API调用延迟（秒）
MAX_GENERATION_RETRIES = 3  # 答案生成重试次数（当未找到标准格式"So the answer is:"时）

# 优化后的答案生成提示词 - 更强调基于context回答
ANSWER_PROMPT = """You are answering a question based on the provided context. Please provide an accurate and factual answer based on the information given in the context and your own knowledge.

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
- You are STRICTLY FORBIDDEN from providing non-answers such as "Not mentioned", "No information", "None", "Not specified", "N/A", or "Not provided".
- If the answer is not explicitly in the context, you MUST use your internal knowledge or make a reasonable inference to provide a concrete, specific entity or value.
- You **MUST** provide a valid entity, name, date, or value.
- **NEVER** return a non-answer. The following outputs are **BANNED**: "(No answer available)", "Not mentioned", "No information", "None", "Not specified", "N/A".
- If the answer is not in the context, you **MUST** make a reasonable inference or use your world knowledge to provide the best possible specific answer.
- It is better to provide a likely guess than to say "not stated".
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


def request_api(prompt, model, temperature, max_tokens=MAX_OUTPUT_TOKENS):
    """调用Gemini API - 带重试机制和token控制"""
    flag = True
    retry_count = 0
    max_retries = 5

    while flag and retry_count < max_retries:
        try:
            # Gemini API 调用
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                }
            )

            text_response = response.text.strip()
            flag = False
            return text_response

        except Exception as e:
            error_msg = str(e).lower()
            print(f"  API调用错误: {type(e).__name__}: {e}")

            # 处理速率限制
            if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                wait_time = 2 ** retry_count
                print(f"  速率限制，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            # 处理超时
            elif "timeout" in error_msg:
                print(f"  请求超时，等待后重试...")
                time.sleep(2)
            else:
                time.sleep(1)

            retry_count += 1

    if retry_count >= max_retries:
        raise Exception(f"API调用失败，超过最大重试次数 ({max_retries})")


def format_context(context_list):
    """格式化context为更清晰的可读文本，增强可读性"""
    formatted_text = ""

    for idx, (title, sentences) in enumerate(context_list, 1):
        # 添加文档编号和标题
        formatted_text += f"\n{'=' * 60}\n"
        formatted_text += f"Document {idx}: {title}\n"
        formatted_text += f"{'=' * 60}\n"

        if isinstance(sentences, list):
            for sent_idx, sentence in enumerate(sentences, 1):
                formatted_text += f"{sent_idx}. {sentence}\n"
        else:
            formatted_text += f"{sentences}\n"

    formatted_text += f"\n{'=' * 60}\n"
    return formatted_text


def generate_answer(question, context, temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS):
    """生成答案

    Args:
        question: 问题文本
        context: 上下文信息（列表格式）
        temperature: 温度参数，控制随机性
        max_tokens: 最大输出token数

    Returns:
        生成的答案文本
    """
    # 格式化context为清晰的文档格式
    context_text = format_context(context)

    # 构建完整的prompt
    prompt = ANSWER_PROMPT.format(
        context=context_text,
        question=question
    )

    # 调用API
    answer = request_api(prompt, model=MODEL, temperature=temperature, max_tokens=max_tokens)
    return answer


def extract_final_answer(gpt_response):
    """从GPT响应中提取最终答案（"So the answer is:"之后的部分）
    并清理掉标点符号，返回简短答案
    
    Returns:
        str: 提取的答案
        None: 未找到标准格式，需要重新生成
    """
    if "So the answer is:" in gpt_response:
        final_answer = gpt_response.split("So the answer is:")[-1].strip()
        # 移除末尾的句号、引号等标点符号
        final_answer = final_answer.rstrip('.,!?;:"\' ')
        # 移除开头的引号等
        final_answer = final_answer.lstrip('"\' ')
        return final_answer
    
    # 未找到标准格式，返回None表示需要重试
    print(f"      ⚠️ 未找到标准答案格式 'So the answer is:'，需要重新生成")
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


def process_hotpotqa_file(input_path, output_path, start_index=0, end_index=None, temperature=TEMPERATURE, 
                          target_samples=None, max_tokens=MAX_OUTPUT_TOKENS, resume=False):
    """处理HotpotQA验证集文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        start_index: 开始处理的索引（从0开始）
        end_index: 结束处理的索引（不包含该索引），None表示处理到末尾
        temperature: 生成温度，控制随机性和错误率
        target_samples: 目标处理样本数，优先级低于end_index
        max_tokens: 最大输出token数
        resume: 是否从上次中断处继续（True时会忽略start_index参数）
    """
    # 加载数据
    print(f"正在加载数据文件: {input_path}")
    print(f"使用模型: {MODEL}")
    print(f"Temperature: {temperature}")
    print(f"Max Output Tokens: {max_tokens}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 确保data是列表
    if not isinstance(data, list):
        data = [data]

    # 如果输出文件已存在且resume=True，加载进度并继续
    processed_data = load_progress(output_path)
    if resume and processed_data:
        start_index = len(processed_data)
        print(f"📌 断点续传模式：从第 {start_index} 条记录继续处理...")
    else:
        processed_data = []

    # 确定处理范围
    if end_index is not None:
        # 如果指定了结束索引，使用它
        total = min(end_index, len(data))
    elif target_samples is not None:
        # 如果指定了样本数，计算结束位置
        total = min(start_index + target_samples, len(data))
    else:
        # 否则处理到末尾
        total = len(data)

    # 验证范围有效性
    if start_index >= len(data):
        print(f"❌ 错误: 起始索引 {start_index} 超出数据范围 (总共 {len(data)} 条)")
        return processed_data
    
    if start_index >= total:
        print(f"❌ 错误: 起始索引 {start_index} 大于等于结束索引 {total}")
        return processed_data

    print(f"\n📊 数据范围信息:")
    print(f"  - 数据集总量: {len(data)} 条")
    print(f"  - 处理范围: [{start_index}, {total}) (索引从0开始)")
    print(f"  - 将处理: {total - start_index} 条数据")

    # 统计计数器（统计已处理数据中的标签分布）
    existing_correct = sum(1 for item in processed_data if item.get('generation_label') == 'correct')
    existing_hallucination = sum(1 for item in processed_data if item.get('generation_label') == 'hallucination')
    
    if processed_data:
        print(f"\n📝 已有数据统计: 正确 {existing_correct} | 幻觉 {existing_hallucination}")
    
    # 当前批次计数器（只统计本次新处理的数据）
    correct_count = 0
    hallucination_count = 0
    skipped_count = 0  # 跳过的数据数量（无法获得标准格式答案）

    # 记录处理时间
    start_time = time.time()

    # 处理每条数据
    for idx in range(start_index, total):
        item = data[idx].copy()  # 创建副本避免修改原数据

        print(f"\n{'=' * 80}")
        print(f"处理第 {idx + 1}/{total} 条数据")
        print(f"问题: {item['question']}")
        print(f"正确答案: {item.get('answer', 'N/A')}")

        # 显示context信息
        if 'context' in item:
            print(f"Context文档数: {len(item['context'])}")
            for i, (title, _) in enumerate(item['context'][:2], 1):  # 显示前2个文档标题
                print(f"  - 文档{i}: {title}")
            if len(item['context']) > 2:
                print(f"  - ... 共 {len(item['context'])} 个文档")

        try:
            # 生成答案，带重试机制
            answer = None
            final_answer = None
            
            for retry_attempt in range(MAX_GENERATION_RETRIES):
                # 生成答案（将question和context一起传入）
                answer = generate_answer(
                    question=item['question'],
                    context=item['context'],  # 传入完整的context
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # 提取最终答案
                final_answer = extract_final_answer(answer)
                
                # 如果成功提取到答案，跳出重试循环
                if final_answer is not None:
                    if retry_attempt > 0:
                        print(f"      ✓ 第 {retry_attempt + 1} 次尝试成功获取标准格式答案")
                    break
                
                # 如果未能提取答案且不是最后一次尝试，等待后重试
                if retry_attempt < MAX_GENERATION_RETRIES - 1:
                    print(f"      ⟳ 第 {retry_attempt + 1} 次尝试失败，等待1秒后重试...")
                    time.sleep(1)
                else:
                    print(f"      ✗ 已重试 {MAX_GENERATION_RETRIES} 次仍未获得标准格式答案，跳过此条数据")
            
            # 如果所有重试都失败，跳过这条数据
            if final_answer is None:
                print(f"      ⚠️ 跳过索引 {idx} 的数据")
                skipped_count += 1
                continue

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
            item['max_tokens'] = max_tokens  # 记录最大token数

            print(f"\n生成的完整回答:")
            print(f"{answer[:500]}..." if len(answer) > 500 else answer)  # 限制显示长度
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
                current_batch_count = idx + 1 - start_index
                avg_time = elapsed_time / current_batch_count if current_batch_count > 0 else 0
                remaining = (total - idx - 1) * avg_time

                # 当前批次统计
                batch_total = current_batch_count
                batch_correct_ratio = correct_count / batch_total * 100 if batch_total > 0 else 0
                batch_hallucination_ratio = hallucination_count / batch_total * 100 if batch_total > 0 else 0

                # 全部数据统计
                all_correct = existing_correct + correct_count
                all_hallucination = existing_hallucination + hallucination_count
                all_total = len(processed_data)
                all_correct_ratio = all_correct / all_total * 100 if all_total > 0 else 0
                all_hallucination_ratio = all_hallucination / all_total * 100 if all_total > 0 else 0

                print(f"\n💾 保存进度... (已处理 {idx + 1}/{total} 条)")
                print(f"   本批次统计: 正确 {correct_count} ({batch_correct_ratio:.1f}%) | 幻觉 {hallucination_count} ({batch_hallucination_ratio:.1f}%) | 跳过 {skipped_count}")
                if existing_correct + existing_hallucination > 0:
                    print(f"   全部数据统计: 正确 {all_correct} ({all_correct_ratio:.1f}%) | 幻觉 {all_hallucination} ({all_hallucination_ratio:.1f}%)")
                print(f"   平均耗时: {avg_time:.2f}秒/条")
                print(f"   预计剩余时间: {remaining / 60:.1f}分钟")
                save_data(processed_data, output_path)

            # 添加延迟以避免API限制
            time.sleep(API_DELAY)

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
    current_batch_count = total - start_index
    
    # 本批次统计
    batch_correct_ratio = correct_count / current_batch_count * 100 if current_batch_count > 0 else 0
    batch_hallucination_ratio = hallucination_count / current_batch_count * 100 if current_batch_count > 0 else 0
    
    # 全部数据统计
    all_correct = existing_correct + correct_count
    all_hallucination = existing_hallucination + hallucination_count
    all_total = len(processed_data)
    all_correct_ratio = all_correct / all_total * 100 if all_total > 0 else 0
    all_hallucination_ratio = all_hallucination / all_total * 100 if all_total > 0 else 0

    print(f"\n结果已保存到: {output_path}")
    print(f"\n📊 本批次统计信息 (处理范围: [{start_index}, {total})):")
    print(f"  - 尝试处理: {current_batch_count} 条")
    print(f"  - 成功生成: {correct_count + hallucination_count} 条")
    print(f"  - 正确答案: {correct_count} ({batch_correct_ratio:.1f}%)")
    print(f"  - 幻觉答案: {hallucination_count} ({batch_hallucination_ratio:.1f}%)")
    print(f"  - 跳过数据: {skipped_count} 条 (无法获得标准格式答案)")
    print(f"  - 总耗时: {total_time / 60:.1f} 分钟")
    print(f"  - 平均耗时: {total_time / current_batch_count:.2f} 秒/条")
    
    if existing_correct + existing_hallucination > 0:
        print(f"\n📊 全部数据统计信息:")
        print(f"  - 总数据量: {all_total}")
        print(f"  - 正确答案: {all_correct} ({all_correct_ratio:.1f}%)")
        print(f"  - 幻觉答案: {all_hallucination} ({all_hallucination_ratio:.1f}%)")
    
    print(f"\n🔧 模型配置:")
    print(f"  - 使用模型: {MODEL}")
    print(f"  - Temperature: {temperature}")
    print(f"  - Max Tokens: {max_tokens}")

    return processed_data


if __name__ == "__main__":
    # ================= 文件路径配置 =================
    input_file = "./hotpot_dev_fullwiki_v1.json"
    output_file = "./hotpot_dev_with_gpt_answers_new.json"

    # ================= 生成参数配置 =================
    # Temperature 控制随机性：
    # - 0.0-0.3: 更确定性，更少幻觉
    # - 0.5-0.7: 平衡，自然的幻觉率
    # - 0.8-1.0: 更随机，更多幻觉
    GENERATION_TEMPERATURE = 0.7

    # 输出token数控制：
    # - 500: 简短回答
    # - 1000: 中等长度
    # - 2000: 详细回答
    # - 4000: 非常详细
    OUTPUT_MAX_TOKENS = 1000

    # ================= 数据范围配置 =================
    # 方式1: 使用起始和结束索引（精确控制）
    START_INDEX = 1500        # 起始索引（从0开始，包含该索引）
    END_INDEX = None       # 结束索引（不包含该索引），None表示处理到末尾
    
    # 方式2: 使用目标样本数（从START_INDEX开始处理指定数量）
    TARGET_SAMPLES = 1500  # 目标处理样本数，None表示处理全部
    
    # 断点续传模式（True时会忽略START_INDEX，从上次中断处继续）
    RESUME_MODE = False
    
    # 使用示例：
    # 1. 处理前100条：START_INDEX=0, END_INDEX=100 或 START_INDEX=0, TARGET_SAMPLES=100
    # 2. 处理100-200条：START_INDEX=100, END_INDEX=200
    # 3. 处理200条之后的所有数据：START_INDEX=200, END_INDEX=None
    # 4. 断点续传：RESUME_MODE=True

    print(f"""
{'=' * 80}
HotpotQA 数据集处理脚本 (基于Context的问答 - Gemini API)
{'=' * 80}
配置信息:
  - 输入文件: {input_file}
  - 输出文件: {output_file}
  - 模型: {MODEL}
  - Temperature: {GENERATION_TEMPERATURE}
  - Max Output Tokens: {OUTPUT_MAX_TOKENS}
  - API延迟: {API_DELAY}秒
  - 答案格式重试: {MAX_GENERATION_RETRIES}次
  - 处理策略: 基于Context文档生成答案，自动判定幻觉
  
数据范围:
  - 起始索引: {START_INDEX if not RESUME_MODE else '自动(断点续传)'}
  - 结束索引: {END_INDEX if END_INDEX is not None else '末尾'}
  - 目标样本数: {TARGET_SAMPLES if TARGET_SAMPLES is not None else '未限制'}
  - 断点续传: {'是' if RESUME_MODE else '否'}
{'=' * 80}
    """)

    # 处理文件
    try:
        result = process_hotpotqa_file(
            input_path=input_file,
            output_path=output_file,
            start_index=START_INDEX,
            end_index=END_INDEX,
            temperature=GENERATION_TEMPERATURE,
            target_samples=TARGET_SAMPLES,
            max_tokens=OUTPUT_MAX_TOKENS,
            resume=RESUME_MODE
        )
        total_count = len(result)
        print(f"\n✅ 处理完成！输出文件共包含 {total_count} 条数据")

        # 显示最终标签分布（全部数据）
        if total_count > 0:
            correct = sum(1 for item in result if item.get('generation_label') == 'correct')
            hallucination = sum(1 for item in result if item.get('generation_label') == 'hallucination')
            print(f"\n📈 输出文件标签分布:")
            print(f"  正确: {correct} ({correct / total_count * 100:.1f}%)")
            print(f"  幻觉: {hallucination} ({hallucination / total_count * 100:.1f}%)")

    except KeyboardInterrupt:
        print("\n⏸️  用户中断，进度已保存")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback

        traceback.print_exc()