import json
from scrapegraphai.graphs import SmartScraperGraph

# 配置 SmartScraperGraph
graph_config = {
    "llm": {
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-xx",
        "model": "deepseek-chat",
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434",  # Ollama 本地服务
    },
    "verbose": True,
}

# 网站链接和分类名
categories = [
    ("https://www.lorealparis.com.cn/makeup/face-makeup", "彩妆-脸部彩妆"),
    ("https://www.lorealparis.com.cn/makeup/lip-makeup", "彩妆-唇部彩妆"),
    ("https://www.lorealparis.com.cn/makeup/eye-makeup", "彩妆-眼部产品"),
    ("https://www.lorealparis.com.cn/makeup/face-makeup", "彩妆彩妆脸部"),
    ("https://www.lorealparis.com.cn/skincare/female-skincare", "女士护肤"),
    ("https://www.lorealparis.com.cn/men/men-skincare", "男士护肤"),
    ("https://www.lorealparis.com.cn/men/men-makeup", "男士彩妆")
]

# 最终结果保存
all_results = []

# 遍历每个分类链接
for url, category in categories:
    print(f"正在提取：{category} ({url})")

    smart_scraper_graph = SmartScraperGraph(
        prompt=f"""
            请提取页面中所有产品的详细页网址，并以 JSON 格式返回。
            每个产品请包含以下字段：
            - 产品名
            - 详细页网址:如 https://www.lorealparis.com.cn/impeccable/loreal-men-expert-impeccablec-foundation-001
            格式必须为 JSON 列表，字段名必须使用英文 key。
        """,
        source=url,
        config=graph_config
    )

    try:
        result = smart_scraper_graph.run()
        print(f"✅ 成功提取：{category}")

        # 包装结构：包含分类与提取内容
        all_results.append({
            "category": category,
            "url": url,
            "products": result  # 如果结果是 JSON 列表，保留原始结构
        })

    except Exception as e:
        print(f"❌ 提取失败：{category}，错误：{e}")

# 保存为 JSON 文件
with open("loreal_products.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("✅ 所有数据已保存到 loreal_products.json")
