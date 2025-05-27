# 利用langgraph搭建多智能体系统

### 对agent需要的数据知识与向量库chroma进行准备
#### 为推理提供更科学与透明的方式
1. 对知识pdf进行准备：处理过程在 数据处理.ipynb
2. 商品数据爬取：./products/test.py 采用ai爬虫方式解析
3. 知识图谱构建采用项目:https://github.com/Xikcn/KnowledgeMapNotes


## 项目结构
```plaintext
MapBloom/
├── chroma_mapbloom/       # 知识图谱三元组Chroma向量数据库
├── .env                   # 环境变量配置文件
├── app.py                 # 部署魔搭社区的启动入口
├── workflow.py            # 主程序文件（智能体定义、皮肤分析、对话逻辑）
├── products/              # 产品数据目录
│   └── products.json      # 产品信息文件
├── tools/                 # 工具目录
│   └── Community_query.py # 知识图谱查询工具
├── kg/                    # 知识图谱相关工具目录
│   └── embedding_tools.py # 嵌入模型函数类
├── 美妆知识.html          # 评估专家Agent的知识图谱展示页面
└── 生成美妆团队图片.png   # 封面图片
```


### 使用的模型如下：
1. DeepSeek-V3-0324
2. qwen-vl-max-latest
3. BAAI/bge-base-zh

## 许可证
MIT