# =====================================================================
# 1. 准备文件工作区 —— 作为 Agent 读写文件的沙盒目录
# =====================================================================
# root_dir 定义 Agent 允许访问的文件工作区；这里不要指向项目根目录，避免暴露 .env 等敏感文件。
# virtual_mode=True 会启用路径沙盒校验，阻止 Agent 通过 ../ 或 ~ 访问 root_dir 之外的位置。
import os
import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

from agent.my_tools import web_search
from model.init_chat_model import glm_llm

temp_workspace = Path(__file__).resolve().parent / "agent_workspace"
temp_workspace.mkdir(exist_ok=True)


# =====================================================================
# 2. 创建文件系统 Agent —— 给 Deep Agent 注入 FilesystemBackend
# =====================================================================
agent = create_deep_agent(
    model=glm_llm,
    tools=[web_search],
    backend=LocalShellBackend(
        # 固定 Agent 的文件工作区；这里对应 src/agent/backend/agent_workspace。
        root_dir=str(temp_workspace),
        # 始终开启虚拟模式：一方面适合示例、草稿和临时文件操作，
        # 另一方面可以启用路径沙盒，防止 ../、~ 等路径逃逸。
        # 生产环境如果安全要求更高，应使用更强隔离的沙盒后端。
        virtual_mode=True,
        max_output_bytes=1024 * 1024,
        timeout=30,
        # 设置环境变量，包含编码相关的配置
        env={
            # 获取当前Python解释器的完整路径
            "PATH": f"{os.path.dirname(sys.executable)};{os.environ.get('PATH', '')}",
        },

    ),
    system_prompt='你是一个助手，请根据用户输入的指令，进行相应的操作。'
)
