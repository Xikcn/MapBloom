import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import add_messages
from langgraph.func import entrypoint, task
from langchain_core.tools import tool
from langchain_core.messages import convert_to_messages
from typing import Dict, Any
import os
import json
import gradio as gr
from openai import OpenAI
import base64
import re
from datetime import datetime, timedelta

from tools.Community_query import KnowledgeGraphQueryTool


load_dotenv(dotenv_path="./.env")

query_tool = KnowledgeGraphQueryTool()




# 初始化大模型
model = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    model_name=os.getenv("MODEL_NAME"),
    temperature=os.getenv("TEMPERATURE"),
)

AppState = Dict[str, Any]


# ---------- 工具定义 ----------
@tool
def get_latest_products(profile: Dict[str, str]) -> str:
    """
    根据用户肤质、季节、性别，推荐对应类别下的最多5个产品["男士护肤", "男士彩妆","女士护肤", "彩妆-脸部彩妆","彩妆-唇部彩妆","彩妆-眼部产品"]，供大模型决策。
    返回结构化产品列表，允许无结果。
    """
    gender = profile.get("性别", "女")
    if gender == "男":
        prefer_categories = ["男士护肤", "男士彩妆"]
    else:
        prefer_categories = ["女士护肤", "彩妆-脸部彩妆","彩妆-唇部彩妆","彩妆-眼部产品"]
    try:
        with open("products/products.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"读取产品库失败: {e}"
    results = []
    for block in data:
        if any(cat in block.get("category", "") for cat in prefer_categories):
            for prod in block.get("products", []):
                prod_info = prod.copy()
                prod_info["category"] = block.get("category", "")
                results.append(prod_info)
                if len(results) >= 5:
                    break
        if len(results) >= 5:
            break
    # 允许无结果
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def get_community_triples_by_entities(query: str)->str:
    """调用知识图谱提取函数，返回参考链接与知识三元组"""
    return  query_tool.query_community(
            query=query,
            file="美妆知识",
            top_n=10,
            entity_top_k=5
        )


@tool
def transfer_to_product_agent():
    """转接到美妆设计Agent"""
    return "转接到美妆设计Agent"


@tool
def transfer_to_comparison_agent():
    """转接到产品推荐Agent"""
    return "转接到产品推荐Agent"


@tool
def transfer_to_expert_agent():
    """转接到专家评估Agent"""
    return "转接到专家评估Agent"


@tool
def recommend_checkin_spots(profile: Dict[str, str]) -> str:
    """
    根据用户地区和当前季节，推荐适合拍照打卡的文案和地点。
    profile: 包含地区（location）、季节（季节）、性别等信息
    """
    # 简单示例，可根据实际需求扩展
    location = profile.get("地区", "上海")
    season = profile.get("季节", "夏季")
    # 示例地点和文案
    spot_map = {
        "上海": {
            "夏季": [
                ("外滩夜景", "夏夜微风，灯火阑珊，记录属于你的魔都浪漫。"),
                ("世纪公园", "绿意盎然，阳光洒满，快来一张清新夏日大片！")
            ],
            "冬季": [
                ("南京路步行街", "冬日暖阳下的繁华都市，最适合拍一组时尚街拍。")
            ]
        },
        "北京": {
            "夏季": [
                ("颐和园", "荷花盛开，古韵悠长，夏日北京的诗意打卡地。")
            ],
            "冬季": [
                ("故宫雪景", "银装素裹的紫禁城，拍出最美中国风。")
            ]
        }
    }
    spots = spot_map.get(location, {}).get(season, [("本地热门景点", "记录你的美好瞬间！")])
    # 随机推荐一条
    spot, text = random.choice(spots)
    return json.dumps({"推荐地点": spot, "推荐文案": text, "地区": location, "季节": season}, ensure_ascii=False, indent=2)




from fuzzywuzzy import process, fuzz


def get_product_info(query: str = "", category: str = "") -> str:
    """
    根据产品名称模糊匹配，返回相似度最高的产品列表（按相似度降序）
    query: 产品名，可选
    category: 可选，指定产品类别["男士护肤", "男士彩妆","女士护肤", "彩妆-脸部彩妆","彩妆-唇部彩妆","彩妆-眼部产品"]
    """
    try:
        with open("products/products.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"读取产品库失败: {e}"
    print(query, category)
    # 提取所有符合类别的产品信息
    candidates = []
    for block in data:
        if category and block.get("category") != category:
            continue
        for prod in block.get("products", []):
            product_info = {
                "category": block.get("category", ""),
                **prod
            }
            candidates.append(product_info)

    # 若未指定查询词，随机返回最多5个产品
    if not query.strip():
        k = min(5, len(candidates))
        return json.dumps(random.sample(candidates, k), ensure_ascii=False, indent=2)

    # 模糊匹配逻辑
    product_names = [prod["name"] for prod in candidates]
    query_keywords = query.strip().split()  # 支持多关键词空格分割

    # 计算每个产品名称与查询词的相似度
    scored_products = []
    for prod in candidates:
        name = prod["name"]
        # 多关键词取最高分（或综合分）
        scores = [fuzz.partial_ratio(keyword, name) for keyword in query_keywords]
        max_score = max(scores)
        scored_products.append((prod, max_score))

    # 按相似度降序排序
    scored_products.sort(key=lambda x: x[1], reverse=True)

    # 提取前5个高匹配项
    top_results = [item[0] for item in scored_products[:5]]
    print(top_results)
    return json.dumps(top_results, ensure_ascii=False, indent=2) if top_results else "未找到相关产品信息。"




# ---------- 智能体定义 ----------
product_agent_tools = [
    get_community_triples_by_entities,
    get_product_info,
    transfer_to_comparison_agent,
    transfer_to_expert_agent,
]


product_agent = create_react_agent(
    model,
    product_agent_tools,
    prompt="""
你是美妆设计师，专注为每位用户量身定制独特且细致的妆容教程。请充分结合用户画像（如肤质、季节、性别、年龄、气候、特殊场合等）和外部环境因素（如流行趋势、气候变化、节日等），为用户生成个性化、实用且详细的妆容步骤和技巧。
- 妆容教程需包含底妆、眼妆、唇妆、护肤搭配等模块，每一部分都要有详细的操作步骤、技巧和注意事项。
- 美妆教程不要推荐任何美妆产品。
- 如遇到用户需求涉及产品选择、对比或推荐，请直接调用 'transfer_to_comparison_agent'。
- 如遇到科学性问题，危害问题，需要专业知识，请直接调用 'transfer_to_expert_agent'。
- 回答应避免冗长的欢迎语，直接进入妆容教程内容。
- 如用户画像信息不全，可适当追问补充。
- 在生成妆容教程前，务必先调用 'get_community_triples_by_entities' 工具，获取与用户条件最相关的少量美妆知识（三元组），只需获取最有用的几条即可，避免冗余。这些知识仅作为科学依据辅助妆容设计，不必全部展示给用户。
- 用户一旦提及具体产品名，必须优先调用 'get_product_info' 检索本地产品库，若有结果请直接展示详细信息（如成分、功效、用法、适用肤质、链接等）；若无结果请明确告知"本地产品库暂无该产品信息"，不要随意猜测或泛泛描述。
- 禁止重复输出妆容教程或产品介绍，每轮只输出一次核心内容。
- 转接后请自动响应用户原始问题，无需用户重复输入。
- 如果用户表达了对比、推荐等需求，请直接调用对应的转接工具，并自动携带用户原始问题给下一个Agent，由下一个Agent直接生成答案返回用户。禁止输出"如需对比请告诉我"或"可以转接"之类的提示语。
- 禁止输出任何"已为你转接到..."、"稍后将为你..."、"我会帮你转接到..."等提示语，只需直接生成对比/推荐/评估结果。
"""
)


comparison_agent_tools = [
    get_product_info,
    transfer_to_expert_agent,
]

comparison_agent = create_react_agent(
    model,
    comparison_agent_tools,
    prompt="""
你是产品推荐员，专注于根据美妆设计师提出的需求或用户指定的产品，进行多产品横向对比分析。
- 推荐产品前，必须先调用 'get_latest_products' 检查本地产品库是否有符合用户需求的产品。
- 用户一旦提及具体产品名，必须优先调用 'get_product_info' 检索本地产品库，若有结果请直接展示详细信息（如成分、功效、用法、适用肤质、链接等）；若无结果请明确告知"本地产品库暂无该产品信息"，不要随意猜测或泛泛描述。
- 详细分析各产品的成分、功效、适配性、优缺点，结合用户画像（肤质、季节、性别等）给出客观的对比结论。
- 最终推荐最适合用户的产品，并说明推荐理由。
- 只推荐本地产品库中真实存在且符合条件的产品。
- 产品推荐完成后，请调用 'transfer_to_expert_agent' 交由评判专家进行化学成分与适配性评估。
- 不要输出妆容教程或操作步骤，专注于产品对比与推荐。
- 如果没有合适的产品，请直接告知用户"暂无合适产品"，不要强行推荐其他品牌或无关产品。
- 禁止重复输出产品介绍，每轮只输出一次核心内容。
- 转接后请自动响应用户原始问题，无需用户重复输入。
- 如果用户表达了需要妆造教程、方案评估等需求，请直接调用对应的转接工具，并自动携带用户原始问题给下一个Agent，由下一个Agent直接生成答案返回用户。禁止输出"如需对比请告诉我"或"可以转接"之类的提示语。
- 禁止输出任何"已为你转接到..."、"稍后将为你..."、"我会帮你转接到..."等提示语，只需直接生成对比/推荐/评估结果。
"""
)

expert_agent_tools = [
    get_community_triples_by_entities,
    get_product_info,
    transfer_to_product_agent,
]

expert_agent = create_react_agent(
    model,
    expert_agent_tools,
    prompt="""
你是美妆方案评估专家，专注于对产品对比推荐的产品或美妆方案进行专业成分和适配性评估。
- 请结合知识图谱和专业判断，分析推荐产品的成分、功效是否适合用户的肤质、季节和特殊需求，判断其他agent生成的回答是否符合科学性。
- 回答科学性原理性问题前，务必先调用 'get_community_triples_by_entities'，获取美妆与功能化妆品的专业知识，将使用到的专业知识显示给用户，保证推理的透明性。
- 评估时需引用知识三元组作为科学依据，给出是否适合用户的明确结论和详细理由。
- 如发现推荐不适合用户，请说明原因并调用 'transfer_to_product_agent'，重新进入妆容设计流程。
- 不要输出产品对比或妆容教程，专注于化学成分与适配性评估。
- 回答应简明、专业、直接。
- 用户一旦提及具体产品名，必须优先调用 'get_product_info' 检索本地产品库，若有结果请直接展示详细信息（如成分、功效、用法、适用肤质、链接等）；若无结果请明确告知"本地产品库暂无该产品信息"，不要随意猜测或泛泛描述。
- 禁止重复输出产品介绍，每轮只输出一次核心内容。
- 转接后请自动响应用户原始问题，无需用户重复输入。
- 如果用户表达了对比、推荐，需要妆造教程等需求，请直接调用对应的转接工具，并自动携带用户原始问题给下一个Agent，由下一个Agent直接生成答案返回用户。禁止输出"如需对比请告诉我"或"可以转接"之类的提示语。
- 禁止输出任何"已为你转接到..."、"稍后将为你..."、"我会帮你转接到..."等提示语，只需直接生成对比/推荐/评估结果。
"""
)

checkin_agent_tools = [recommend_checkin_spots]

checkin_agent = create_react_agent(
    model,
    checkin_agent_tools,
    prompt="""
你是打卡推荐员，专注于根据用户所在地区和当前季节，为用户推荐适合拍照打卡的文案和地点。
- 请结合用户画像中的地区（如上海、北京等）、季节（如夏季、冬季等）等信息，生成有创意、适合社交平台分享的拍照打卡文案和推荐地点。
- 推荐内容要简洁有趣，适合用户直接复制使用。
- 如用户未提供地区信息，可主动询问或默认推荐本地热门景点。
- 每次推荐三组文案和地点，3到6个拍照姿势脚本。
- 禁止输出与拍照打卡无关的内容。
"""
)



# ---------- 多智能体调度 ----------
@task
def call_product_agent(messages, state: AppState):
    response = product_agent.invoke({"messages": messages, "state": state})
    return response["messages"]


@task
def call_comparison_agent(messages, state: AppState):
    response = comparison_agent.invoke({"messages": messages, "state": state})
    return response["messages"]


@task
def call_expert_agent(messages, state: AppState):
    response = expert_agent.invoke({"messages": messages, "state": state})
    return response["messages"]


@task
def call_checkin_agent(messages, state: AppState):
    response = checkin_agent.invoke({"messages": messages, "state": state})
    return response["messages"]


# ---------- 主流程 ----------
@entrypoint()
def workflow(user_messages, user_profile=None):
    messages = add_messages([], user_messages)
    # 优先用外部传入的 user_profile
    state: AppState = {
        "user_profile": user_profile if user_profile else {
            "肤质": "油性",
            "年龄": "22",
            "季节": "夏季",
            "性别": "女"
        },
        "some_flag": True,
        "last_user_question": "",
        "last_transfer": None,
    }

    # 记录上一轮用户问题
    for m in reversed(user_messages):
        if m["role"] == "user":
            state["last_user_question"] = m["content"]
            break

    call_active_agent = call_product_agent

    while True:
        agent_messages = call_active_agent(messages, state).result()
        messages = add_messages(messages, agent_messages)

        ai_msg = next(m for m in reversed(agent_messages) if isinstance(m, AIMessage))

        if not ai_msg.tool_calls:
            break

        tool_call = ai_msg.tool_calls[-1]
        name = tool_call["name"]

        # 只在转接类型变化时追加原始问题
        if name != state.get("last_transfer"):
            if "last_user_question" in state:
                messages = add_messages(messages, [{"role": "user", "content": state["last_user_question"]}])
            state["last_transfer"] = name

        if name == "transfer_to_product_agent":
            call_active_agent = call_product_agent
        elif name == "transfer_to_comparison_agent":
            call_active_agent = call_comparison_agent
        elif name == "transfer_to_expert_agent":
            state["from_product_or_comparison"] = True
            state["current_recommendation"] = messages[-1]["content"] if messages else ""
            call_active_agent = call_expert_agent
        elif name == "recommend_checkin_spots":
            call_active_agent = call_checkin_agent
        else:
            raise ValueError(f"未知工具调用：{name}")

    return messages


def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"从子图更新 {graph_id}:\n")
    for node_name, node_update in update.items():
        print(f"从节点更新 {node_name}:\n")
        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_skin(image_path):
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key=os.getenv("VL_API_KEY"),  # 替换为你的API Key
        base_url=os.environ["VL_BASE_URL"],
    )
    prompt = '''
你是一位专业的皮肤检测AI，请根据用户上传的人脸图片，识别并输出以下皮肤状态，要求输出标准JSON格式：

{
  "肤质": "油性/干性/混合/中性/敏感/无法识别",
  "色斑": "无/轻微/明显/严重/无法识别",
  "痘痘": "无/轻微/明显/严重/无法识别",
  "黑头": "无/轻微/明显/严重/无法识别",
  "毛孔粗大": "无/轻微/明显/严重/无法识别",
  "敏感度": "低/中/高/无法识别",
  "其他问题": "如有请简要描述，否则为无"
}
只输出JSON，不要输出多余解释。
    '''
    completion = client.chat.completions.create(
        model=os.getenv("VL_MODEL"),
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "你是专业的皮肤检测AI。"}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def get_current_season():
    # 获取北京时间
    now = datetime.utcnow() + timedelta(hours=8)
    month = now.month
    if month in [3, 4, 5]:
        return "春季"
    elif month in [6, 7, 8]:
        return "夏季"
    elif month in [9, 10, 11]:
        return "秋季"
    else:
        return "冬季"

def gradio_workflow_interface():
    history = []
    user_profile = {
        "肤质": "混合",
        "年龄": "22",
        "季节": get_current_season(),
        "性别": "女"
    }
    skin_data = {}
    # 新增：当前agent类型，默认美妆设计师
    agent_type_state = gr.State({"type": "product_agent"})

    def upload_and_analyze(image):
        nonlocal skin_data, user_profile
        if image is None:
            return "未上传图片", None
        # 保存临时图片
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        result_str = analyze_skin(tmp_path)
        # 清洗 markdown 代码块标记
        try:
            pattern = r"```json\s*({.*?})\s*```"
            match = re.search(pattern, result_str, re.DOTALL)
            if match:
                json_content = match.group(1)
                result = json.loads(json_content)
            else:
                print("没找到")
        except Exception:
            result = {"error": f"皮肤分析结果解析失败: {result_str}"}
        if 'error' in result:
            return result['error'], None
        skin_data = result
        for k, v in result.items():
            if v and v != "无法识别":
                user_profile[k] = v
        return f"皮肤分析结果: {json.dumps(result, ensure_ascii=False)}", result

    def chat(user_input, chat_history, skin_info, agent_type):
        # agent_type 是 dict
        agent_type_value = agent_type.get("type", "product_agent")
        # 始终优先用skin_data（全局最新皮肤状态）补充user_profile
        if skin_data:
            for k, v in skin_data.items():
                if v and v != "无法识别":
                    user_profile[k] = v
        elif skin_info:
            for k, v in skin_info.items():
                if v and v != "无法识别":
                    user_profile[k] = v
        # 构造消息历史
        messages = []
        # system prompt
        system_profile = f"用户画像：{json.dumps(user_profile, ensure_ascii=False)}"
        messages.append({"role": "system", "content": system_profile})
        if chat_history is None:
            chat_history = []
        for msg in chat_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(msg)
        messages.append({"role": "user", "content": user_input})
        # 检查用户画像是否有皮肤状态
        if not user_profile.get("肤质","混合"):
            reply = "未检测到皮肤状态，请上传图片或补充肤质信息（如油性、干性等）。"
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": reply})
            return "", chat_history, skin_info, agent_type
        # agent切换逻辑
        if agent_type_value == "product_agent":
            output = product_agent.invoke({"messages": messages, "state": {"user_profile": user_profile}})
            agent_role = "美妆设计师"
        elif agent_type_value == "comparison_agent":
            output = comparison_agent.invoke({"messages": messages, "state": {"user_profile": user_profile}})
            agent_role = "产品推荐员"
        elif agent_type_value == "expert_agent":
            output = expert_agent.invoke({"messages": messages, "state": {"user_profile": user_profile}})
            agent_role = "评估专家"
        elif agent_type_value == "checkin_agent":
            output = checkin_agent.invoke({"messages": messages, "state": {"user_profile": user_profile}})
            agent_role = "打卡推荐员"
        else:
            output = product_agent.invoke({"messages": messages, "state": {"user_profile": user_profile}})
            agent_role = "美妆设计师"
        from langchain_core.messages import AIMessage
        ai_msgs = [m for m in convert_to_messages(output["messages"]) if isinstance(m, AIMessage)]
        if ai_msgs:
            reply = f"{agent_role}：{ai_msgs[-1].content}"
        else:
            reply = "抱歉，未能生成回复。"
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": reply})
        return "", chat_history, skin_info, agent_type

    def set_agent_product(agent_type):
        return agent_type, "美妆设计师已激活！"
    def set_agent_comparison(agent_type):
        return agent_type, "产品推荐员已激活！"
    def set_agent_expert(agent_type):
        return agent_type, "评估专家已激活！"
    def set_agent_checkin(agent_type):
        return agent_type, "打卡推荐员已激活！"

    # 在gr.Blocks()外部或顶部增加自定义CSS
    if hasattr(gr, 'CSS'):
        custom_css = gr.CSS('''
        .gr-button[elem_id^="product-btn-"] {
            width: 120px !important;
            height: 36px !important;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 14px;
            padding: 0 8px;
            margin-right: 8px;
        }
        ''')
    else:
        custom_css = None

    with gr.Blocks(css=custom_css) as demo:
        with gr.Tabs():
            with gr.TabItem("AI美妆多智能体系统"):
                gr.Markdown("# 图妆（AI美妆多智能体系统）\n先在左侧上传图片分析皮肤状况-再进行咨询效果更好哦（采用专业的外部知识图谱加持的专家能够从化学成分帮你进行科学分析）")
                with gr.Row():
                    # 新增一列，放封面图片
                    with gr.Column(scale=1):
                        with gr.Row(scale=1):
                            cover_image = gr.Image(
                                value="生成美妆团队图片.png",  # 你的封面图片路径，放在项目根目录或指定路径
                                interactive=False,
                                show_label=False,
                                height=256,
                                width=256
                            )
                            image_input = gr.Image(type="pil", label="上传面部图片", height=256, width=256)
                        skin_output = gr.Textbox(label="皮肤分析结果", interactive=False)
                        upload_btn = gr.Button("分析皮肤")
                        gr.Markdown("### 选择当前服务Agent：")
                        with gr.Row():
                            btn_product = gr.Button("美妆设计师")
                            btn_comparison = gr.Button("产品推荐员")
                            btn_expert = gr.Button("评估专家")
                            btn_checkin = gr.Button("打卡推荐员")
                        agent_status = gr.Textbox(label="当前Agent", value="美妆设计师已激活！", interactive=False)
                    with gr.Column(scale=2):
                        # 新增：新品推荐区
                        with gr.Row():
                            # 5个产品按钮，设置统一宽度和高度，超出部分用省略号
                            product_btns = [
                                gr.Button(
                                    f"产品{i+1}",
                                    elem_id=f"product-btn-{i}",
                                    scale=1
                                ) for i in range(5)
                            ]
                            refresh_btn = gr.Button("换一批")
                        # 聊天区
                        chatbot = gr.Chatbot(label="AI美妆对话", type="messages")
                        user_input = gr.Textbox(label="输入你的问题", placeholder="请输入美妆相关问题...", lines=2)
                        send_btn = gr.Button("发送")
                # 状态存储
                skin_info_state = gr.State({})
                chat_history_state = gr.State([])
                # agent类型状态
                agent_type_state = gr.State({"type": "product_agent"})
                # 事件绑定
                upload_btn.click(upload_and_analyze, inputs=[image_input], outputs=[skin_output, skin_info_state])
                send_btn.click(chat, inputs=[user_input, chat_history_state, skin_info_state, agent_type_state], outputs=[user_input, chatbot, skin_info_state, agent_type_state])
                btn_product.click(lambda: ({"type": "product_agent"}, "美妆设计师已激活！"), None, [agent_type_state, agent_status])
                btn_comparison.click(lambda: ({"type": "comparison_agent"}, "产品推荐员已激活！"), None, [agent_type_state, agent_status])
                btn_expert.click(lambda: ({"type": "expert_agent"}, "评估专家已激活！"), None, [agent_type_state, agent_status])
                btn_checkin.click(lambda: ({"type": "checkin_agent"}, "打卡推荐员已激活！"), None, [agent_type_state, agent_status])

                # 新品推荐逻辑
                def refresh_products(user_profile):
                    names = get_five_products(user_profile)
                    return names

                # 初始化时加载
                demo.load(refresh_products, inputs=[gr.State(user_profile)], outputs=product_btns)

                # 换一批按钮
                refresh_btn.click(refresh_products, inputs=[gr.State(user_profile)], outputs=product_btns)

                # 产品按钮点击事件
                for i, btn in enumerate(product_btns):
                    btn.click(
                        append_product_to_input,
                        inputs=[btn, user_input],
                        outputs=user_input
                    )
            # with gr.TabItem("评估专家Agent知识库"):
            #     value = f"""
            #                 <div style="width:100%; height:800px;">
            #                     <iframe srcdoc='{open("美妆知识.html", "r", encoding="utf-8").read().replace("'", "&apos;")}'
            #                             style="width:100%; height:100%; border:none;"></iframe>
            #                 </div>
            #     """
            #     gr.HTML(label="评估专家Agent知识库", value=value)

    return demo


def get_five_products(user_profile=None):
    try:
        result = get_random_products()
        # print("get_random_products result:", result)
        products = json.loads(result)
        # print("products:", products)
        # 只要有产品就返回前5个名字
        if isinstance(products, list) and products:
            return [prod.get("name", "无名产品") for prod in products][:5]
        else:
            return ["暂无推荐"] * 5
    except Exception as e:
        print("新品推荐异常：", e)
        return ["暂无推荐"] * 5

def append_product_to_input(product_name, user_input):
    # 末尾追加 #产品名
    if user_input is None:
        user_input = ""
    if not user_input.endswith(" "):
        user_input += " "
    return user_input + f"#{product_name}"


def get_random_products() -> str:
    """
    不考虑用户画像，直接从products/products.json中所有产品里随机抽取5个，返回产品信息列表（json字符串，含产品名、类别等）。
    """
    try:
        with open("products/products.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"读取产品库失败: {e}"
    candidates = []
    for block in data:
        for prod in block.get("products", []):
            product_info = {
                "category": block.get("category", ""),
                **prod
            }
            candidates.append(product_info)
    if not candidates:
        return "未找到产品信息。"
    k = min(5, len(candidates))
    return json.dumps(random.sample(candidates, k), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    gradio_workflow_interface().launch()