from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from utils.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, GLM_BASE_URL, GLM_API_KEY

# =====================================================================
# 1. 创建共享模型 —— 供项目内普通示例统一复用
# =====================================================================
deepseek_llm: BaseChatModel = init_chat_model(
    model="DeepSeek-V4-Flash",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    # api_base 是 ChatDeepSeek 的原生服务地址字段。
    api_base=DEEPSEEK_BASE_URL,
    # 关闭思考模式，使基础示例的响应更直接，并保持与原公共模型配置一致。
    extra_body={"thinking": {"type": "disabled"}},
)


# =====================================================================
# 2. 创建 GLM 模型 —— 使用 OpenAI 兼容接口接入智谱模型
# =====================================================================
glm_llm: BaseChatModel = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL,
)
