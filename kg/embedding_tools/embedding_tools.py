from modelscope import AutoModel, AutoTokenizer
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
import logging

# 配置日志记录
logger = logging.getLogger(__name__)


class BgeZhEmbeddingFunction(EmbeddingFunction):
    """
        基于魔搭社区BAAI/bge-base-zh模型的嵌入函数
    """

    _instance = None  # 保持单例模式

    def __new__(cls, model_path: str = 'BAAI/bge-base-zh', **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_model(model_path, **kwargs)
        return cls._instance

    def _init_model(self, model_path: str, **kwargs):
        """通过魔搭API初始化模型"""
        try:
            # 加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path, **kwargs)

            # 设备自动检测
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)

            # 配置默认参数
            self.encode_config = {
                "max_length": 512,  # 与魔搭模型参数对齐[5](@ref)
                "batch_size": 32,
                "normalize_embeddings": True
            }

            logger.info(f"魔搭模型加载成功 | 设备:{self.device} | 模型:{model_path}")

        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise RuntimeError("魔搭模型初始化失败，请检查模型ID或网络连接") from e

    def _preprocess(self, texts: Documents) -> dict:
        """符合中文场景的文本预处理"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.encode_config["max_length"],
            return_tensors="pt"
        ).to(self.device)

    def __call__(self, texts: Documents, **kwargs) -> Embeddings:
        """生成嵌入的核心逻辑"""
        try:
            # 参数合并（允许动态修改）
            config = {**self.encode_config, **kwargs}

            # 输入验证
            if not texts or not all(isinstance(t, str) for t in texts):
                raise ValueError("输入必须是非空字符串列表")

            # 文本预处理
            inputs = self._preprocess(texts)

            # 生成嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 获取CLS位置的嵌入向量
            embeddings = outputs.last_hidden_state[:, 0, :]

            # 归一化处理
            if config["normalize_embeddings"]:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # 转换为列表格式
            return embeddings.cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"编码失败: {str(e)}")
            raise RuntimeError(f"嵌入生成异常: {str(e)}") from e

