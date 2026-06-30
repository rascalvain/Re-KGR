import json
import re
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import time

# 假设使用 OpenAI API
from openai import OpenAI

# ============ Prompt 模板 ============
USER_PROMPT_TEMPLATE = """
You are an expert Wikidata Ontology Mapper and Knowledge Graph Engineer.

Your Task:
Receive a LIST of input triples (Subject, Relation, Object). For each triple, normalize the "Relation" phrase into a standard, specific Wikidata Property Label (e.g., convert "was born in" to "place of birth", "wrote" to "screenwriter" or "author").

CRITICAL RULES:
1. Output Format: You must return a SINGLE valid JSON Array containing objects.
2. Context Awareness: Use the Subject and Object types to determine the correct property (e.g., (Person, from, Country) -> "country of citizenship", but (Product, from, Company) -> "manufacturer").
3. No Fact Checking: DO NOT verify if the triple is true. Your job is only to normalize the schema. Even if the fact is hallucinated (e.g., "Elon Musk, painted, Mona Lisa"), you must map "painted" to "creator" or "painter", do not change the entities.
4. Entities: Keep the Subject and Object strings EXACTLY as provided.

JSON Structure per item:
{{
    "index": <Integer, original list index>,
    "original_relation": "<String, original relation>",
    "wikidata_property": "<String, the normalized Wikidata label>",
    "normalized_triple": ["<Subject>", "<wikidata_property>", "<Object>"]
}}

Input List:
{INPUT_TRIPLE_LIST}

Output (valid JSON array only):
"""

SYSTEM_PROMPT_TEXT = """
You are a Wikidata ontology expert. Your task is to normalize relation phrases into standard Wikidata property labels.

Example Input:
1. (Barack Obama, hails from, Kenya)
2. (Inception, directed by, Christopher Nolan)
3. (Paris, is the capital of, France)

Example Output:
[
  {
    "index": 1,
    "original_relation": "hails from",
    "wikidata_property": "place of birth",
    "normalized_triple": ["Barack Obama", "place of birth", "Kenya"]
  },
  {
    "index": 2,
    "original_relation": "directed by",
    "wikidata_property": "director",
    "normalized_triple": ["Inception", "director", "Christopher Nolan"]
  },
  {
    "index": 3,
    "original_relation": "is the capital of",
    "wikidata_property": "capital",
    "normalized_triple": ["Paris", "capital", "France"]
  }
]
"""


# ============ 辅助函数 ============
def is_valid_triple_component(component: str) -> bool:
    """
    检查三元组成分是否有效
    - 不能为空字符串
    - 不能只包含空格
    - 不能为 None
    """
    if component is None:
        return False
    if not isinstance(component, str):
        return False
    if not component.strip():
        return False
    return True


def parse_triple_string(triple_str: str) -> Optional[Tuple[str, str, str]]:
    """
    解析三元组字符串，例如: "(John Russell Reynolds, was, English lawyer)"
    返回: (subject, relation, object) 或 None（如果解析失败或不完整）
    """
    if not triple_str or not isinstance(triple_str, str):
        return None
    
    triple_str = triple_str.strip()
    if triple_str.startswith('(') and triple_str.endswith(')'):
        triple_str = triple_str[1:-1]

    # 使用逗号分割，但考虑到实体名称中可能包含逗号
    parts = [p.strip() for p in triple_str.split(',')]

    if len(parts) < 3:
        return None
    
    # 前两个逗号分隔的是 subject 和 relation，剩余的是 object
    subject = parts[0]
    relation = parts[1]
    obj = ', '.join(parts[2:])  # 合并剩余部分
    
    # 验证三个成分都有效
    if not (is_valid_triple_component(subject) and 
            is_valid_triple_component(relation) and 
            is_valid_triple_component(obj)):
        return None
    
    return (subject, relation, obj)


def validate_triples_list(triples_str_list: List[str]) -> Tuple[List[Tuple[str, str, str]], List[int]]:
    """
    验证三元组列表，返回有效的三元组和无效的索引
    
    Returns:
        (valid_triples, invalid_indices)
    """
    valid_triples = []
    invalid_indices = []
    
    for idx, triple_str in enumerate(triples_str_list):
        parsed = parse_triple_string(triple_str)
        if parsed is None:
            invalid_indices.append(idx)
        else:
            valid_triples.append(parsed)
    
    return valid_triples, invalid_indices


def format_triple_to_string(subject: str, relation: str, obj: str) -> str:
    """
    将三元组格式化为字符串: "(subject, relation, object)"
    """
    return f"({subject}, {relation}, {obj})"


def batch_normalize_relations(
        triples_list: List[Tuple[str, str, str]],
        llm_client,
        max_retries: int = 3,
        delay: float = 1.0
) -> List[Tuple[str, str, str]]:
    """
    批量调用 LLM 标准化三元组的关系

    Args:
        triples_list: 三元组列表 [(subject, relation, object), ...]
        llm_client: OpenAI 客户端实例
        max_retries: 最大重试次数
        delay: 重试延迟（秒）

    Returns:
        标准化后的三元组列表
    """
    if not triples_list:
        return []

    # 1. 构造输入文本
    formatted_input = ""
    for idx, (h, r, t) in enumerate(triples_list, 1):
        formatted_input += f"{idx}. ({h}, {r}, {t})\n"

    # 2. 填充 Prompt
    final_prompt = USER_PROMPT_TEMPLATE.replace("{INPUT_TRIPLE_LIST}", formatted_input)

    # 3. 调用 LLM（带重试机制）
    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model="gpt-3.5-turbo",  # 或使用 gpt-4 获得更好的效果
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEXT},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.0  # 保持稳定输出
            )

            content = response.choices[0].message.content

            # 4. 清洗和解析 JSON
            # 尝试提取 JSON 数组
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                # 验证数据完整性
                if len(data) != len(triples_list):
                    print(f"Warning: 返回的三元组数量 ({len(data)}) 与输入不匹配 ({len(triples_list)})")

                # 提取标准化后的三元组
                normalized_triples = []
                for item in data:
                    if 'normalized_triple' in item and len(item['normalized_triple']) == 3:
                        normalized_triples.append(tuple(item['normalized_triple']))
                    else:
                        # 如果某个三元组格式不正确，使用原始三元组
                        idx = item.get('index', 1) - 1
                        if 0 <= idx < len(triples_list):
                            normalized_triples.append(triples_list[idx])

                return normalized_triples
            else:
                print(f"Warning: 未在响应中找到 JSON 数组 (尝试 {attempt + 1}/{max_retries})")

        except json.JSONDecodeError as e:
            print(f"JSON 解析错误 (尝试 {attempt + 1}/{max_retries}): {e}")
        except Exception as e:
            print(f"API 调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")

        # 等待后重试
        if attempt < max_retries - 1:
            time.sleep(delay)

    # 所有重试都失败，返回原始列表
    print("Error: 所有重试都失败，返回原始三元组")
    return triples_list


def process_single_item(
    item: Dict[str, Any],
    llm_client,
    stats: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    处理单条数据，验证并标准化三元组
    
    Args:
        item: 单条数据
        llm_client: OpenAI 客户端
        stats: 统计信息字典
    
    Returns:
        处理后的数据，如果数据无效则返回 None
    """
    new_item = item.copy()
    has_valid_data = False
    
    # 处理 original 字段
    if 'original' in item and item['original']:
        valid_triples, invalid_indices = validate_triples_list(item['original'])
        
        if invalid_indices:
            stats['invalid_original_triples'] += len(invalid_indices)
            print(f"\n发现 {len(invalid_indices)} 个无效的 original 三元组 (索引: {invalid_indices})")
        
        if valid_triples:
            try:
                # 批量标准化
                normalized_original = batch_normalize_relations(valid_triples, llm_client)
                
                # 转换回字符串格式
                new_item['original_wikidata'] = [
                    format_triple_to_string(s, r, o) 
                    for s, r, o in normalized_original
                ]
                new_item['original_valid_count'] = len(valid_triples)
                new_item['original_invalid_count'] = len(invalid_indices)
                has_valid_data = True
                stats['valid_original_triples'] += len(valid_triples)
            except Exception as e:
                print(f"\n处理 original 字段时出错: {e}")
                stats['processing_errors'] += 1
        else:
            print(f"\noriginal 字段没有有效的三元组")
            new_item['original_valid_count'] = 0
            new_item['original_invalid_count'] = len(item['original'])
    
    # 处理 wiki_ref 字段
    if 'wiki_ref' in item and item['wiki_ref']:
        valid_triples, invalid_indices = validate_triples_list(item['wiki_ref'])
        
        if invalid_indices:
            stats['invalid_wiki_ref_triples'] += len(invalid_indices)
            print(f"\n发现 {len(invalid_indices)} 个无效的 wiki_ref 三元组 (索引: {invalid_indices})")
        
        if valid_triples:
            try:
                # 批量标准化
                normalized_wiki_ref = batch_normalize_relations(valid_triples, llm_client)
                
                # 转换回字符串格式
                new_item['wiki_ref_wikidata'] = [
                    format_triple_to_string(s, r, o) 
                    for s, r, o in normalized_wiki_ref
                ]
                new_item['wiki_ref_valid_count'] = len(valid_triples)
                new_item['wiki_ref_invalid_count'] = len(invalid_indices)
                has_valid_data = True
                stats['valid_wiki_ref_triples'] += len(valid_triples)
            except Exception as e:
                print(f"\n处理 wiki_ref 字段时出错: {e}")
                stats['processing_errors'] += 1
        else:
            print(f"\nwiki_ref 字段没有有效的三元组")
            new_item['wiki_ref_valid_count'] = 0
            new_item['wiki_ref_invalid_count'] = len(item['wiki_ref'])
    
    # 如果两个字段都没有有效数据，返回 None（将被过滤）
    if not has_valid_data:
        stats['skipped_items'] += 1
        return None
    
    return new_item


def print_stats(stats: Dict[str, Any]):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print("统计信息:")
    print(f"{'='*60}")
    print(f"总数据条数:          {stats['total_items']}")
    print(f"成功处理:            {stats['processed_items']}")
    print(f"跳过（无有效数据）:  {stats['skipped_items']}")
    print(f"-" * 60)
    print(f"有效 original 三元组: {stats['valid_original_triples']}")
    print(f"无效 original 三元组: {stats['invalid_original_triples']}")
    print(f"有效 wiki_ref 三元组: {stats['valid_wiki_ref_triples']}")
    print(f"无效 wiki_ref 三元组: {stats['invalid_wiki_ref_triples']}")
    print(f"-" * 60)
    print(f"处理错误:            {stats['processing_errors']}")
    print(f"{'='*60}\n")


def process_dataset(
        input_file: str,
        output_file: str,
        llm_client,
        batch_size: int = 10,
        process_limit: int = None,
        save_interval: int = 100
):
    """
    处理整个数据集，对 original 和 wiki_ref 字段进行 Wikidata 格式化
    自动跳过不完整的三元组，并在导出时筛除无效数据

    Args:
        input_file: 输入 JSON 文件路径
        output_file: 输出 JSON 文件路径
        llm_client: OpenAI 客户端实例
        batch_size: 每批处理的三元组数量（暂未使用）
        process_limit: 限制处理的数据条数（用于测试）
        save_interval: 保存间隔（每处理多少条数据保存一次）
    """
    # 读取数据
    print(f"正在读取数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if process_limit:
        dataset = dataset[:process_limit]
        print(f"限制处理前 {process_limit} 条数据")

    print(f"共 {len(dataset)} 条数据")

    # 统计信息
    stats = {
        'total_items': len(dataset),
        'processed_items': 0,
        'skipped_items': 0,
        'valid_original_triples': 0,
        'invalid_original_triples': 0,
        'valid_wiki_ref_triples': 0,
        'invalid_wiki_ref_triples': 0,
        'processing_errors': 0
    }

    # 处理每条数据
    processed_data = []

    for idx, item in enumerate(tqdm(dataset, desc="处理数据")):
        processed_item = process_single_item(item, llm_client, stats)
        
        if processed_item is not None:
            processed_data.append(processed_item)
            stats['processed_items'] += 1

        # 定期保存
        if (idx + 1) % save_interval == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            print(f"\n已保存中间结果 ({len(processed_data)} 条有效数据)")
            print_stats(stats)

    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # 保存统计报告
    stats_file = output_file.replace('.json', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("处理完成！")
    print(f"{'='*60}")
    print(f"结果文件: {output_file}")
    print(f"统计文件: {stats_file}")
    print_stats(stats)


# ============ 使用示例 ============
if __name__ == "__main__":
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",
        base_url="https://api.openai-proxy.org/v1"  # 或其他兼容接口
    )

    # 处理数据集
    process_dataset(
        input_file="./dataset/WikiBio_dataset/wikibio_with_triples_new.json",
        output_file="./dataset/WikiBio_dataset/wikibio_with_triples_wikidata.json",
        llm_client=client,
        batch_size=10,  # 每批处理 10 个三元组（暂未使用）
        process_limit=None,  # 处理所有数据，设置为数字可限制处理条数用于测试
        save_interval=50  # 每 50 条保存一次
    )

    # 测试单个样例
    # test_triples = [
    #     ("John Russell Reynolds", "was", "English lawyer"),
    #     ("John Russell Reynolds", "was born in", "London"),
    #     ("John Russell Reynolds", "educated at", "Eton College")
    # ]
    # result = batch_normalize_relations(test_triples, client)
    # print(result)

