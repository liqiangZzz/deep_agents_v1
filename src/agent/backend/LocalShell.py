import os
import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

from agent.my_tools import web_search
from model.init_chat_model import glm_llm

# =====================================================================
# 1. 准备文件工作区 —— 作为 Agent 读写文件的沙盒目录
# =====================================================================
# root_dir 定义 Agent 允许访问的文件工作区；这里不要指向项目根目录，避免暴露 .env 等敏感文件。
# virtual_mode=True 会启用路径沙盒校验，阻止 Agent 通过 ../ 或 ~ 访问 root_dir 之外的位置。
# LocalShellBackend 会在这个工作区内执行命令，命令产生的文件也应限制在这里。
temp_workspace = Path(__file__).resolve().parent / "agent_workspace"
temp_workspace.mkdir(exist_ok=True)

# =====================================================================
# 2. 创建本地 Shell Agent —— 给 Deep Agent 注入 LocalShellBackend
# =====================================================================
# LocalShellBackend 会给 Agent 提供本地命令执行能力，例如 execute。
# 当 Agent 判断需要运行 Python、ls、cat 等命令时，会通过这个后端访问本机环境执行代码。
# 这比普通 FilesystemBackend 权限更高，生产环境需要谨慎限制 root_dir、环境变量、超时和输出大小。
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
        # 限制单次命令输出，避免命令打印大量内容导致内存或日志膨胀。
        max_output_bytes=1024 * 1024,
        # 限制命令最长执行时间，避免 Agent 触发长时间运行或卡死的本地进程。
        timeout=30,
        # 设置命令执行时的环境变量。这里只补 PATH，让 execute 能找到当前 Python 环境里的命令。
        # 不要把密钥、Token、数据库密码等敏感变量主动注入给本地 Shell 后端。
        env={
            # 把当前 Python 解释器所在目录放到 PATH 前面，优先使用当前虚拟环境的 python/pip 等命令。
            "PATH": f"{os.path.dirname(sys.executable)};{os.environ.get('PATH', '')}",
        },

    ),
    system_prompt='你是一个助手，请根据用户输入的指令，进行相应的操作。'
)
