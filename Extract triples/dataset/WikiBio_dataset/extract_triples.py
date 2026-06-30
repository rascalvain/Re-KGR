from openai import OpenAI
import openai
import time
import json

client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",
)

MODEL = "gpt-4.1-mini"

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


def request_api(prompt, model, temperature):
    """调用OpenAI API"""
    flag = True
    while flag:
        try:
            message = [{'role': 'user', 'content': prompt}]
            response = client.chat.completions.create(
                model=model,
                messages=message,
                max_completion_tokens=1000,
                n=1
            )
            text_response = response.choices[0].message.content.strip()
            flag = False
            return text_response
        except openai.RateLimitError as e:
            print("速率限制超出，等待中...")
            time.sleep(0.01)
        except Exception as e:
            print(f"API调用错误: {e}")
            time.sleep(0.005)


def extract_triples(text, enable_verification=True):
    """从文本中提取三元组，可选择是否启用验证和修正流程
    
    Args:
        text: 待提取三元组的文本
        enable_verification: 是否启用验证和修正流程（默认为True）
    
    Returns:
        提取并验证后的三元组列表
    """
    if not text or text.strip() == "":
        return []

    # 第一阶段：初始提取三元组
    print("  [阶段1/2] 初始提取三元组...")
    prompt = GPT_TRIPLE_EXTRACTION_PROMPT.format(init_text=text)
    init_response = request_api(prompt, model=MODEL, temperature=0.0)
    
    # 输出初始提取的API响应到控制台
    print("\n" + "="*80)
    print("初始提取 API 响应:")
    print("-"*80)
    print(init_response)
    print("="*80 + "\n")
    
    # 如果不启用验证，直接返回初始提取结果
    if not enable_verification:
        triples = process_triples_response(init_response)
        return triples
    
    # 第二阶段：验证和修正三元组
    print("  [阶段2/2] 验证和修正三元组...")
    revise_prompt = GPT_TRIPLE_EXTRACTION_PROMPT_REVISE.format(
        p=GPT_TRIPLE_EXTRACTION_PROMPT.format(init_text=text),
        t=init_response
    )
    revised_response = request_api(revise_prompt, model=MODEL, temperature=0.0)
    
    # 输出验证修正后的API响应到控制台
    print("\n" + "="*80)
    print("验证修正 API 响应:")
    print("-"*80)
    print(revised_response)
    print("="*80 + "\n")
    
    # 处理修正后的三元组
    triples = process_triples_response(revised_response)
    return triples


def process_triples_response(response):
    """处理API返回的三元组响应，转换为字符串数组格式"""
    Triple_start = response.find("Triple")
    if Triple_start != -1:
        response = response[Triple_start:]
    else:
        print("响应中未找到 'Triple'")
        return []

    processed_data = []
    lines = response.split('\n')

    for line in lines:
        if line.strip():
            try:
                # 提取 "Triple: " 后面的内容
                if ': ' in line:
                    triple_str = line.split(': ', 1)[1].strip()
                    processed_data.append(triple_str)
            except Exception as e:
                print(f"处理三元组时出错: {e}, 行内容: {line}")
                continue

    return processed_data


def process_wikibio_file(input_path, output_path, start_index=0, enable_verification=True):
    """处理wikibio.json文件，为每条数据提取三元组
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        start_index: 开始处理的索引（用于断点续传）
        enable_verification: 是否启用三元组验证和修正流程（默认为True）
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

    # 处理每条数据
    total = len(data) if isinstance(data, list) else 1

    # 如果data是单个对象，转换为列表
    if not isinstance(data, list):
        data = [data]

    for idx in range(start_index, len(data)):
        item = data[idx]
        print(f"\n处理第 {idx + 1}/{total} 条数据...")

        try:
            # 提取gpt3_text的三元组
            print("  - 正在提取 gpt3_text 的三元组...")
            gpt3_text = item.get('gpt3_text', '')
            original_triples = extract_triples(gpt3_text, enable_verification=enable_verification)
            item['original'] = original_triples
            print(f"    提取到 {len(original_triples)} 个三元组")

            # 提取wiki_bio_text的三元组
            print("  - 正在提取 wiki_bio_text 的三元组...")
            wiki_bio_text = item.get('wiki_bio_text', '')
            wiki_ref_triples = extract_triples(wiki_bio_text, enable_verification=enable_verification)
            item['wiki_ref'] = wiki_ref_triples
            print(f"    提取到 {len(wiki_ref_triples)} 个三元组")

            # 添加到处理后的数据中
            processed_data.append(item)

            # 每处理10条数据保存一次
            if (idx + 1) % 10 == 0:
                print(f"\n保存进度... (已处理 {idx + 1} 条)")
                save_data(processed_data, output_path)

        except Exception as e:
            print(f"处理第 {idx + 1} 条数据时出错: {e}")
            # 保存当前进度
            save_data(processed_data, output_path)
            raise e

    # 最终保存
    print(f"\n处理完成！正在保存最终结果...")
    save_data(processed_data, output_path)
    print(f"结果已保存到: {output_path}")
    return processed_data


if __name__ == "__main__":
    # 配置输入输出路径
    input_file = "./wikibio.json"
    output_file = "./wikibio_with_triples_new.json"
    
    # 配置是否启用三元组验证和修正流程
    # True: 启用两阶段提取（初始提取 + 验证修正），质量更高但耗时更长
    # False: 仅进行初始提取，速度快但可能包含错误
    ENABLE_VERIFICATION = True

    # 处理文件
    try:
        result = process_wikibio_file(input_file, output_file, enable_verification=ENABLE_VERIFICATION)
        print(f"\n成功处理了 {len(result)} 条数据")
    except KeyboardInterrupt:
        print("\n用户中断，进度已保存")
    except Exception as e:
        print(f"\n处理过程中出现错误: {e}")