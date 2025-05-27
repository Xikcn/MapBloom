import os
import json
import chromadb
from chromadb.utils import embedding_functions
from kg.embedding_tools.embedding_tools import BgeZhEmbeddingFunction
import networkx as nx
from collections import defaultdict
import community as community_louvain


class KnowledgeGraphQueryTool:
    def __init__(self, chroma_path=None, embeddings_path=None):
        # 初始化路径配置
        self._init_paths(chroma_path, embeddings_path)

        # 初始化嵌入模型
        self.embedding_func = BgeZhEmbeddingFunction(model_path=self.embeddings_path)

        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=self.chroma_path)

        # 获取知识图谱集合
        self.kg_states = self.client.get_or_create_collection(name="kg_states")

    def _init_paths(self, chroma_path, embeddings_path):
        """初始化存储路径"""
        if not chroma_path:
            self.chroma_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '../chroma_mapbloom'))
        else:
            self.chroma_path = chroma_path

        if not embeddings_path:
            self.embeddings_path = "BAAI/bge-base-zh"
        else:
            self.embeddings_path = embeddings_path

    def _extract_entities(self, query, entity_list, top_k=5):
        """通过嵌入相似度提取相关实体"""
        if not entity_list:
            return []

        # 计算查询嵌入
        query_emb = self.embedding_func([query])[0]

        # 计算所有实体嵌入
        entity_embs = self.embedding_func(entity_list)

        # 计算余弦相似度
        import numpy as np
        def cos_sim(a, b):
            a, b = np.array(a), np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        # 排序并返回top_k实体
        scored = [(e, cos_sim(query_emb, emb)) for e, emb in zip(entity_list, entity_embs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, s in scored[:top_k]]

    def query_community(self, query, file=None, top_n=20, entity_top_k=10):
        """
        社区发现主查询方法
        :param query: 用户查询文本
        :param file: 指定知识库文件名称
        :param top_n: 返回三元组数量上限
        :param entity_top_k: 实体提取数量
        :return: JSON格式的三元组列表
        """
        # 获取可用知识库文件
        files_info = self.kg_states.get()
        file_ids = files_info.get('ids', [])

        if not file_ids:
            return json.dumps([])

        # 自动选择第一个文件
        if not file:
            file = file_ids[0]
        elif file not in file_ids:
            return json.dumps([])

        # 加载图数据结构
        kg_info = self.kg_states.get(ids=[file])
        if not kg_info["metadatas"]:
            return json.dumps([])

        metadata = kg_info["metadatas"][0]
        if "current_G" not in metadata:
            return json.dumps([])

        try:
            G = nx.node_link_graph(json.loads(metadata["current_G"]))
        except Exception as e:
            print(f"图数据解析失败: {str(e)}")
            return json.dumps([])

        # 实体提取和社区发现
        entity_list = list(G.nodes)
        selected_entities = self._extract_entities(query, entity_list, entity_top_k)
        if not selected_entities:
            return json.dumps([])

        # 执行Louvain社区发现
        partition = community_louvain.best_partition(G.to_undirected())
        community_ids = {partition[e] for e in selected_entities if e in partition}

        # 构建社区子图
        community_nodes = [n for n, cid in partition.items() if cid in community_ids]
        subgraph = G.subgraph(community_nodes)

        # 提取并格式化三元组
        triples = []
        for u, v, data in subgraph.edges(data=True):
            triples.append({
                'source': u,
                'target': v,
                'relation': data.get('label', '相关'),
                'context': data.get('title', ''),
                'weight': data.get('weight', 0.5)
            })

        # 排序和结果处理
        triples.sort(key=lambda x: x['weight'], reverse=True)
        cleaned_data = [{k: v for k, v in item.items() if k != 'weight'}
                        for item in triples[:top_n]]

        return json.dumps(cleaned_data, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # 使用示例
    query_tool = KnowledgeGraphQueryTool()

    while True:
        user_input = input("请输入查询内容（输入q退出）: ")
        if user_input.lower() == 'q':
            break

        result = query_tool.query_community(
            query=user_input,
            file="美妆知识",
            top_n=10,
            entity_top_k=5
        )
        print("查询结果:")
        print(result)