import json
import time
from scrapegraphai.graphs import SmartScraperGraph

# SmartScraperGraph配置（同之前）
graph_config = {
    "llm": {
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-xx",
        "model": "deepseek-chat",
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434",
    },
    "verbose": True,
}

# 加载已有产品基本信息
with open("loreal_products.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

# 用于保存最终结构
detailed_data = []

for category_block in all_data:
    category = category_block["category"]
    print(f"\n📦 分类: {category}")

    detailed_products = []

    for product in category_block["products"]['products']:
        product_url = product["product_url"]
        print(f"  🔍 正在抓取产品详情：{product_url}")

        # 配置 SmartScraperGraph 提取详情
        detail_scraper = SmartScraperGraph(
            prompt="""
                请从产品详细页中提取以下信息，并以 JSON 格式返回：
                - name: 产品名称
                - efficacy: 产品功效（简洁扼要）
                - ingredients: 核心成分（如有）
                - usage: 使用方法（简洁描述）
                - url: 当前详细页网址
                如果某项没有内容，用空字符串填充。
            """,
            source=product_url,
            config=graph_config
        )

        try:
            result = detail_scraper.run()

            # 添加详细页链接（确保保留）
            result["url"] = product_url

            detailed_products.append(result)
            print("    ✅ 抓取成功")
        except Exception as e:
            print(f"    ❌ 抓取失败：{e}")
            continue

        time.sleep(1.5)  # 避免请求过快被屏蔽

    detailed_data.append({
        "category": category,
        "url": category_block["url"],
        "products": detailed_products
    })

# 保存为最终结果
with open("loreal_products_detailed.json", "w", encoding="utf-8") as f:
    json.dump(detailed_data, f, ensure_ascii=False, indent=2)

print("\n✅ 所有产品详情已保存到 loreal_products_detailed.json")
