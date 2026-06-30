#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查询Wikidata PID对应的真正谓词名称
从 extracted_pids.txt 读取PID，查询对应的属性标签，写入 pred2id.txt
"""

import requests
import json
import time
from typing import Dict, List
import random


def get_headers():
    """
    获取合适的请求头
    """
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }


# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用SPARQL查询Wikidata PID对应的真正谓词名称
"""

import requests
import json
import time
from typing import Dict, List
import random


def query_properties_sparql(pids: List[str]) -> Dict[str, str]:
    """
    使用SPARQL查询属性标签
    """
    result = {}
    batch_size = 100  # SPARQL可以处理更大的批量

    sparql_endpoint = "https://query.wikidata.org/sparql"

    for i in range(0, len(pids), batch_size):
        batch_pids = pids[i:i + batch_size]

        # 构建SPARQL查询
        pid_values = " ".join([f"wd:{pid}" for pid in batch_pids])

        sparql_query = f"""
        SELECT ?property ?propertyLabel WHERE {{
            VALUES ?property {{ {pid_values} }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        """

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; WikidataPropertyQuery/1.0)',
                'Accept': 'application/sparql-results+json'
            }

            response = requests.get(
                sparql_endpoint,
                params={'query': sparql_query, 'format': 'json'},
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            for binding in data['results']['bindings']:
                property_uri = binding['property']['value']
                property_id = property_uri.split('/')[-1]  # 提取PID
                property_label = binding.get('propertyLabel', {}).get('value', property_id)
                result[property_id] = property_label

            print(f"SPARQL查询已处理 {min(i + batch_size, len(pids))}/{len(pids)} 个PID")
            time.sleep(1)  # SPARQL查询间隔

        except Exception as e:
            print(f"SPARQL查询第 {i // batch_size + 1} 批时出错: {e}")
            # 为这批未成功的PID设置默认值
            for pid in batch_pids:
                if pid not in result:
                    result[pid] = pid

    return result


# 其余函数保持不变...


def batch_query_wikidata_properties(pids: List[str], language: str = "en") -> Dict[str, str]:
    """
    批量查询Wikidata属性

    Args:
        pids: PID列表
        language: 查询语言

    Returns:
        PID到标签的映射字典
    """
    result = {}
    batch_size = 20  # 减小批量大小，避免URL过长和请求过大

    for i in range(0, len(pids), batch_size):
        batch_pids = pids[i:i + batch_size]

        try:
            # Wikidata API URL
            url = "https://www.wikidata.org/w/api.php"

            # API 参数
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch_pids),
                "props": "labels",
                "languages": language,
                "format": "json"
            }

            # 发送请求，添加合适的请求头
            response = requests.get(url, params=params, headers=get_headers(), timeout=30)
            response.raise_for_status()

            # 解析响应
            data = response.json()

            if "entities" in data:
                for pid in batch_pids:
                    if pid in data["entities"]:
                        entity = data["entities"][pid]
                        if "labels" in entity and language in entity["labels"]:
                            result[pid] = entity["labels"][language]["value"]
                        else:
                            result[pid] = pid  # 没有找到标签，使用原PID
                    else:
                        result[pid] = pid  # 实体不存在，使用原PID

            # 添加随机延迟避免请求过快
            time.sleep(random.uniform(0.5, 1.5))

            print(f"已处理 {min(i + batch_size, len(pids))}/{len(pids)} 个PID")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"遇到403错误，切换为单个查询模式...")
                # 如果遇到403错误，逐个查询这一批
                for pid in batch_pids:
                    if pid not in result:
                        result[pid] = query_wikidata_property_single(pid, language)
                        time.sleep(random.uniform(1, 2))  # 单个查询时间间隔更长
            else:
                print(f"HTTP错误 {e.response.status_code}: {e}")
                # 其他HTTP错误也回退到单个查询
                for pid in batch_pids:
                    if pid not in result:
                        result[pid] = query_wikidata_property_single(pid, language)
                        time.sleep(random.uniform(0.5, 1))
        except Exception as e:
            print(f"批量查询第 {i // batch_size + 1} 批时出错: {e}")
            # 如果批量查询失败，逐个查询这一批
            for pid in batch_pids:
                if pid not in result:
                    result[pid] = query_wikidata_property_single(pid, language)
                    time.sleep(random.uniform(0.3, 0.8))

    return result


def query_wikidata_property_single(pid: str, language: str = "en") -> str:
    """
    单个查询Wikidata属性（带重试机制）
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbgetentities",
                "ids": pid,
                "props": "labels",
                "languages": language,
                "format": "json"
            }

            response = requests.get(url, params=params, headers=get_headers(), timeout=15)
            response.raise_for_status()

            data = response.json()

            if "entities" in data and pid in data["entities"]:
                entity = data["entities"][pid]
                if "labels" in entity and language in entity["labels"]:
                    return entity["labels"][language]["value"]

            return pid

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                wait_time = 2 ** attempt  # 指数退避
                print(f"请求过于频繁，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            elif e.response.status_code == 403:
                print(f"403错误，尝试更长延迟...")
                time.sleep(random.uniform(2, 5))
                if attempt == max_retries - 1:
                    return pid
                continue
            else:
                print(f"HTTP错误 {e.response.status_code} 查询 {pid}: {e}")
                return pid
        except Exception as e:
            print(f"查询 {pid} 第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))
            else:
                return pid

    return pid


def read_pids_from_file(filename: str) -> List[str]:
    """
    从文件中读取PID列表

    Args:
        filename: 文件名

    Returns:
        PID列表
    """
    pids = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                pid = line.strip()
                if pid and pid.startswith('P'):
                    pids.append(pid)
        print(f"从 {filename} 读取到 {len(pids)} 个PID")
    except FileNotFoundError:
        print(f"错误：找不到文件 {filename}")
    except Exception as e:
        print(f"读取文件 {filename} 时出错: {e}")

    return pids


def write_pred2id_file(pid_to_label: Dict[str, str], filename: str):
    """
    将PID到标签的映射写入文件

    Args:
        pid_to_label: PID到标签的映射字典
        filename: 输出文件名
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write("# PID到谓词标签的映射\n")
            f.write("# 格式: PID\t标签\n")
            f.write("# ==========================================\n\n")

            # 按PID排序写入
            for pid in sorted(pid_to_label.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
                label = pid_to_label[pid]
                f.write(f"{pid}\t{label}\n")

        print(f"结果已写入 {filename}")

    except Exception as e:
        print(f"写入文件 {filename} 时出错: {e}")


def main():
    """主函数"""
    input_file = "./output/extracted_pids.txt"
    output_file = "./output/pred2id.txt"

    print("开始处理PID到谓词标签的映射...")
    print("注意：由于API限制，查询可能需要较长时间...")

    # 1. 读取PID列表
    pids = read_pids_from_file(input_file)
    if not pids:
        print("没有找到有效的PID，程序退出")
        return

    # 2. 批量查询Wikidata
    print("开始查询Wikidata...")
    pid_to_label = batch_query_wikidata_properties(pids)

    # 3. 写入结果文件
    write_pred2id_file(pid_to_label, output_file)

    # 4. 显示统计信息
    successful_queries = sum(1 for pid, label in pid_to_label.items() if label != pid)
    print(f"\n=== 处理完成 ===")
    print(f"总PID数量: {len(pids)}")
    print(f"成功查询到标签: {successful_queries}")
    print(f"查询失败（保持原PID）: {len(pids) - successful_queries}")

    # 显示一些示例
    print(f"\n=== 部分结果示例 ===")
    count = 0
    for pid in sorted(pid_to_label.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        if count >= 10:
            break
        label = pid_to_label[pid]
        if label != pid:  # 只显示成功查询到的
            print(f"{pid}: {label}")
            count += 1


if __name__ == "__main__":
    main()