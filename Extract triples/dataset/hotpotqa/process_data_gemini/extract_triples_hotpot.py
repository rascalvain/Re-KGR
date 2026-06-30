from google import genai
import time
import json

# ================= Gemini API 配置 =================
client = genai.Client(
    api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",
    http_options={"base_url": "https://api.openai-proxy.org/google"}
)

MODEL = "gemini-2.5-flash-lite-preview-09-2025"  # 或使用其他 Gemini 模型

GPT_TRIPLE_EXTRACTION_PROMPT = \
    """In the knowledge graph, knowledge triples are a basic data structure used to represent and store information, and each triple is an expression of a fact. Given a piece of text, please extract all knowledge triples contained in the text, and represent the triples in the form of ("head entity", "relationship", "tail entity").\
    Note that the extracted triples need to be as fine-grained as possible. It is necessary to ensure that the semantics of the triple are consistent with the information in the corresponding part of the text and there is not a pronoun in the triple. All knowledge triples in the text need to be extracted\
    Here is an in-context example:
    <text>:Paris, the capital of France, is a city with a long history and full of romance. Not only is there the world-famous Eiffel Tower and Louvre Museum, but it also has a unique artistic atmosphere and rich cultural heritage.
    <response>:
    Triple: (Paris, is, the capital of France)
    Triple: (Paris, possession, long history)
    Triple: (Paris, full, romantic)
    Triple: (Paris, possession, Eiffel Tower)
    Triple: (Paris, possession, Louvre)
    Triple: (Paris, possessions, unique artistic atmosphere)
    Triple: (Paris, possessions, rich cultural heritage)
    <text>:"\"The Girl Who Loved Tom Gordon\" is a novel by Stephen King, published in 1999. The story follows a young girl named Trisha McFarland who becomes lost in the woods while on a family hike. As she struggles to survive, she turns to her favorite baseball player, Tom Gordon, for comfort and guidance. The novel explores themes of isolation, fear, and the power of imagination. It was a critical and commercial success, and has been adapted into a comic book and a stage play."
    <response>:
    Triple: ("The Girl Who Loved Tom Gordon", is, a novel by Stephen King)
    Triple: ("The Girl Who Loved Tom Gordon", published in, 1999)
    Triple: ("The Girl Who Loved Tom Gordon", follows, Trisha McFarland)
    Triple: ("The Girl Who Loved Tom Gordon" protagonist: Trisha McFarland, becomes, lost in the woods)
    Triple: ("The Girl Who Loved Tom Gordon" protagonist: Trisha McFarland, turns to, Tom Gordon for comfort and guidance)
    Triple: ("The Girl Who Loved Tom Gordon", explores themes of, "isolation, fear, and the power of imagination")
    Triple: ("The Girl Who Loved Tom Gordon", was, a critical and commercial success)
    Triple: ("The Girl Who Loved Tom Gordon", has been adapted into, a comic book)
    Triple: ("The Girl Who Loved Tom Gordon", has been adapted into, a stage play)
    <text>{init_text}
    """

GPT_TRIPLE_EXTRACTION_PROMPT_REVISE = \
    """Below are the knowledge Triples you extracted based on the text, but there are still some errors in it. For example, the semantics of the triple are different from the semantics of the corresponding part in the original text or there is a pronoun in the triple. Please check and correct
    <initial prompt>{p}
    <triples>{t}\
    Please output all corrected triples directly, including changed and unmodified ones. Don't output any other words. """


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


def request_api(prompt, model, temperature, max_tokens=800):
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


def format_context_to_text(context):
    """将context列表转换为文本"""
    text_parts = []
    for title, sentences in context:
        if isinstance(sentences, list):
            # 合并句子列表
            full_text = ' '.join(sentences)
        else:
            full_text = sentences
        text_parts.append(f"{title}: {full_text}")

    return '\n'.join(text_parts)


def init_triples_response(text):
    """初始三元组提取"""
    prompts = GPT_TRIPLE_EXTRACTION_PROMPT.format(init_text=text)
    init_triples = request_api(prompts, model=MODEL, temperature=0.0)
    return init_triples


def update_triples_response(text, response):
    """修正三元组"""
    update_prompt = GPT_TRIPLE_EXTRACTION_PROMPT_REVISE.format(
        p=GPT_TRIPLE_EXTRACTION_PROMPT.format(init_text=text),
        t=response
    )
    update_triples = request_api(update_prompt, model=MODEL, temperature=0.0)
    return update_triples


def process_triples_response(response):
    """处理API返回的三元组响应"""
    Triple_start = response.find("Triple")
    if Triple_start != -1:
        response = response[Triple_start:]
    else:
        print("  响应中未找到 'Triple'")
        return []

    processed_data = []
    lines = response.split('\n')

    for ts in lines:
        if ts.strip():
            try:
                # 提取 "Triple: " 后面的内容
                if ': ' in ts:
                    Triple = ts.split(': ', 1)[1].strip()
                    temp_data = {'triple': Triple}
                    processed_data.append(temp_data)
            except Exception as e:
                print(f"  处理三元组时出错: {e}")
                continue

    return processed_data


def extract_triples_with_verification(text, text_type="text"):
    """提取三元组并验证（两阶段）"""
    if not text or not text.strip():
        return []

    try:
        # 第一阶段：初始提取
        print(f"    [阶段1/2] 初始提取 {text_type} 的三元组...")
        init_response = init_triples_response(text)

        # 第二阶段：验证和修正
        print(f"    [阶段2/2] 验证和修正 {text_type} 的三元组...")
        revised_response = update_triples_response(text, init_response)

        # 处理修正后的三元组
        triples = process_triples_response(revised_response)
        print(f"    提取到 {len(triples)} 个三元组")

        return triples

    except Exception as e:
        print(f"    ❌ 提取 {text_type} 三元组时出错: {e}")
        return []


def process_hotpotqa_file(input_path, output_path, start_index=0, enable_verification=True):
    """处理HotpotQA数据集，提取context和gpt_sentence的三元组

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        start_index: 开始处理的索引（用于断点续传）
        enable_verification: 是否启用验证和修正流程
    """
    # 加载数据
    print(f"正在加载数据文件: {input_path}")
    print(f"验证模式: {'启用' if enable_verification else '禁用'}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果输出文件已存在，加载进度
    processed_data = load_progress(output_path)
    if processed_data:
        start_index = len(processed_data)
        print(f"从第 {start_index} 条记录继续处理...")

    total = len(data)
    print(f"总共 {total} 条数据")

    # 记录处理时间
    start_time = time.time()

    # 处理每条数据
    for idx in range(start_index, len(data)):
        item = data[idx].copy()

        print(f"\n{'=' * 80}")
        print(f"处理第 {idx + 1}/{total} 条数据")
        print(f"问题: {item.get('question', 'N/A')[:80]}...")

        try:
            # 1. 提取context的三元组
            context = item.get('context', [])
            if context:
                print(f"  - 正在提取 context 的三元组...")
                context_text = format_context_to_text(context)

                # 限制context文本长度（如果太长可能超过token限制）
                max_context_length = 10000  # 字符数限制
                if len(context_text) > max_context_length:
                    print(f"    ⚠️  Context过长({len(context_text)}字符)，截取前{max_context_length}字符")
                    context_text = context_text[:max_context_length]

                if enable_verification:
                    context_triples = extract_triples_with_verification(context_text, "context")
                else:
                    init_response = init_triples_response(context_text)
                    context_triples = process_triples_response(init_response)

                item['context_triples'] = context_triples
            else:
                item['context_triples'] = []
                print(f"    ⚠️  context 为空")

            # 2. 提取gpt_sentence的三元组
            gpt_sentence = item.get('gpt_sentence', '')
            if gpt_sentence and gpt_sentence.strip():
                print(f"  - 正在提取 gpt_sentence 的三元组...")

                if enable_verification:
                    gpt_triples = extract_triples_with_verification(gpt_sentence, "gpt_sentence")
                else:
                    init_response = init_triples_response(gpt_sentence)
                    gpt_triples = process_triples_response(init_response)

                item['gpt_sentence_triples'] = gpt_triples
            else:
                item['gpt_sentence_triples'] = []
                print(f"    ⚠️  gpt_sentence 为空")

            # 添加到处理后的数据中
            processed_data.append(item)

            # 每处理5条数据保存一次
            if (idx + 1) % 5 == 0:
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / (idx + 1 - start_index)
                remaining = (total - idx - 1) * avg_time

                print(f"\n💾 保存进度... (已处理 {idx + 1} 条)")
                print(f"   平均耗时: {avg_time:.2f}秒/条")
                print(f"   预计剩余时间: {remaining / 60:.1f}分钟")
                save_data(processed_data, output_path)

            # 添加延迟避免API限制
            time.sleep(0.5)  # 使用0.5秒延迟

        except Exception as e:
            print(f"❌ 处理第 {idx + 1} 条数据时出错: {e}")
            print(f"   错误类型: {type(e).__name__}")
            # 保存当前进度
            print("   正在保存当前进度...")
            save_data(processed_data, output_path)

            import traceback
            traceback.print_exc()
            print("\n将在5秒后继续处理下一条...")
            time.sleep(5)
            continue

    # 最终保存
    print(f"\n{'=' * 80}")
    print(f"处理完成！正在保存最终结果...")
    save_data(processed_data, output_path)

    # 统计信息
    total_time = time.time() - start_time

    # 统计三元组数量
    total_context_triples = sum(len(item.get('context_triples', [])) for item in processed_data)
    total_gpt_triples = sum(len(item.get('gpt_sentence_triples', [])) for item in processed_data)

    print(f"\n结果已保存到: {output_path}")
    print(f"\n📊 统计信息:")
    print(f"  - 总数据量: {len(processed_data)}")
    print(f"  - Context三元组总数: {total_context_triples}")
    print(f"  - GPT句子三元组总数: {total_gpt_triples}")
    print(f"  - 总三元组数: {total_context_triples + total_gpt_triples}")
    print(f"  - 平均Context三元组/条: {total_context_triples / len(processed_data):.1f}")
    print(f"  - 平均GPT三元组/条: {total_gpt_triples / len(processed_data):.1f}")
    print(f"  - 总耗时: {total_time / 60:.1f} 分钟")
    print(f"  - 平均耗时: {total_time / len(processed_data):.2f} 秒/条")

    return processed_data


if __name__ == "__main__":
    # 配置输入输出路径
    input_file = "hotpot_dev_merged.json"
    output_file = "hotpot_dev_merged_triples.json"

    # 配置是否启用三元组验证和修正流程
    # True: 启用两阶段提取（初始提取 + 验证修正），质量更高但耗时更长
    # False: 仅进行初始提取，速度快但可能包含错误
    ENABLE_VERIFICATION = True

    print(f"""
{'=' * 80}
HotpotQA 知识图谱三元组抽取脚本 (Gemini API)
{'=' * 80}
配置信息:
  - 输入文件: {input_file}
  - 输出文件: {output_file}
  - 模型: {MODEL}
  - 验证模式: {'启用' if ENABLE_VERIFICATION else '禁用'}
  - 处理内容: context + gpt_sentence
{'=' * 80}
    """)

    # 处理文件
    try:
        result = process_hotpotqa_file(
            input_file,
            output_file,
            enable_verification=ENABLE_VERIFICATION
        )
        print(f"\n✅ 成功处理了 {len(result)} 条数据")
    except KeyboardInterrupt:
        print("\n⏸️  用户中断，进度已保存")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback

        traceback.print_exc()