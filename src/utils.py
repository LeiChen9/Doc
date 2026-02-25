import json
import re

def parse_json_string(json_str: str) -> dict:
    """
    解析 JSON 字符串，支持普通 JSON 字符串和被 ```json``` 包裹的 JSON 字符串
    
    Args:
        json_str: 待解析的 JSON 字符串，可以是普通格式或被 ```json``` 包裹的格式
    
    Returns:
        dict: 解析后的字典对象
    
    Raises:
        ValueError: 当输入不是有效的 JSON 格式时抛出
    """
    # 先去除字符串两端的空白字符
    cleaned_str = json_str.strip()
    
    # 定义匹配 ```json``` 包裹内容的正则表达式
    # 匹配以 ```json 开头，``` 结尾的内容，并提取中间的部分
    pattern = r'^```json\s*\n?(.*?)\n?```$'
    
    # 使用正则表达式查找匹配的内容
    match = re.search(pattern, cleaned_str, re.DOTALL)
    
    if match:
        # 如果匹配到 ```json``` 包裹的内容，提取中间的 JSON 部分
        json_content = match.group(1).strip()
    else:
        # 如果没有匹配到，使用原始清理后的字符串
        json_content = cleaned_str
    
    try:
        # 解析 JSON 字符串为字典
        result = json.loads(json_content)
        return result
    except json.JSONDecodeError as e:
        # 解析失败时抛出有意义的异常信息
        raise ValueError(f"无法解析 JSON 字符串: {e}") from e

# ------------------- 测试用例 -------------------
if __name__ == "__main__":
    # 测试 1: 普通的 JSON 字符串
    test1 = '{"name": "张三", "age": 25, "city": "北京"}'
    print("测试 1 结果:", parse_json_string(test1))
    
    # 测试 2: 被 ```json``` 包裹的 JSON 字符串
    test2 = '''```json
    {
        "name": "李四",
        "age": 30,
        "city": "上海"
    }
    ```'''
    print("测试 2 结果:", parse_json_string(test2))
    
    # 测试 3: 边缘情况（包裹格式但无换行）
    test3 = '```json{"name": "王五", "age": 28}```'
    print("测试 3 结果:", parse_json_string(test3))