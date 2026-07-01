"""
多 Agent 入口模块
=================

本模块负责构建一个「主 Agent + 子 Agent」的多层协作架构：

1. 通过 MCP（Model Context Protocol）客户端连接远端工具服务（数据分析、购票等），
   将远端工具动态注入到子 Agent 中；
2. 读取 subagents.yaml 配置文件，按声明式配置组装各子 Agent；
3. 最终调用 create_deep_agent() 创建主 Agent，携带本地工具和子 Agent 列表，
   由框架自动完成 Agent 间调度与委派。

关键依赖：
- mcp_tool_config.mcp_client : MCP 多服务客户端实例
- subagents.yaml              : 子 Agent 声明式配置（名称、提示词、工具组）
- my_tools.web_search         : 主 Agent 可用的本地联网搜索工具
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from agent.multi_agent.mcp_tool_config import mcp_client
from agent.my_tools import web_search
from model.init_chat_model import glm_llm
from utils.env_utils import PRINT_MCP_TOOL_SCHEMA  # ★ 统一从 env_utils 读取环境变量

# =====================================================================
# 1. 路径常量与本地工具注册 —— 项目基准路径 & 本地工具映射表
# =====================================================================

# 项目根目录（src/ 的父目录），作为文件读写、子 Agent 报告保存的基准路径
EXAMPLE_DIR = Path(__file__).resolve().parents[2]
print(f'当前代码执行的工作目录为：{EXAMPLE_DIR}')

# ★ 将所有本地自定义工具统一注册到此处，供 load_subagents 使用。
#   subagents.yaml 中 tools 列表引用的工具名会优先查此映射；
#   找不到的才走 MCP 远端加载。
#   新增本地工具时只需在此处添加一条映射即可。
LOCAL_TOOLS: dict[str, Any] = {
    "web_search": web_search,
}


# =====================================================================
# 2. 工具展示辅助函数 —— 在控制台打印工具清单，方便调试确认
# =====================================================================
def _print_tools(label: str, tools: list[Any]) -> None:
    """在控制台打印工具清单，方便确认 MCP 暴露了哪些能力。

    Args:
        label: 显示标签，例如 "MCP[data-analyzer] / 工具组[data_analyzer]"。
        tools: 工具对象列表，每个对象需支持 ``name``、``description``、
               ``args_schema`` 属性（LangChain BaseTool 兼容接口）。

    环境变量:
        PRINT_MCP_TOOL_SCHEMA: 设为 "1" 时，额外输出每个工具的完整
            JSON Schema；默认不输出，仅显示名称、描述和参数名列表。
            ★ 该变量统一通过 env_utils.PRINT_MCP_TOOL_SCHEMA 管理，
            可在 .env 或系统环境变量中设置。
    """
    # ★ 改为从 env_utils 统一读取，不再直接 os.getenv
    print_schema = PRINT_MCP_TOOL_SCHEMA == "1"
    print(f"\n========== {label} 可用工具: {len(tools)} 个 ==========")

    # 无工具时直接返回
    if not tools:
        print("无可用工具")
        print("=" * 54)
        return

    # 遍历工具列表，逐个打印名称、描述和参数信息
    for index, tool in enumerate(tools, start=1):
        name = getattr(tool, "name", "<unknown>")
        description = (getattr(tool, "description", "") or "").strip()

        # args_schema 可能是 Pydantic model，需转换为 JSON dict
        args_schema = getattr(tool, "args_schema", None)
        if hasattr(args_schema, "model_json_schema"):
            args_schema = args_schema.model_json_schema()

        # 提取参数名列表
        arg_names = []
        if isinstance(args_schema, dict):
            arg_names = list(args_schema.get("properties", {}).keys())

        # 打印工具基本信息
        print(f"{index}. {name}")
        if description:
            print(f"   描述: {description.splitlines()[0]}")
        if arg_names:
            print(f"   参数: {', '.join(arg_names)}")

        # 环境变量开启时，额外输出完整 JSON Schema
        if print_schema and args_schema:
            print(
                "   参数 Schema: "
                + json.dumps(args_schema, ensure_ascii=False, default=str)
            )
    print("=" * 54)


# =====================================================================
# 3. MCP 工具组加载 —— 按工具组名从远端服务获取工具列表
# =====================================================================
async def _load_tool_group(group_name: str) -> list[Any]:
    """加载一个 MCP 工具组；远端不可用时返回空列表，避免图启动失败。

    根据工具组名称映射到 MCP 服务端名称，然后通过 mcp_client 获取
    该服务端暴露的所有工具。

    Args:
        group_name: 工具组名称，对应 subagents.yaml 中 tools 列表里的值。
            目前支持 "data_analyzer" 和 "ticket_booking"。

    Returns:
        工具对象列表；加载失败时返回空列表。
    """
    # ★ 工具组名 → MCP 服务端名的映射
    #   需与 mcp_tool_config.py 中的注册名保持一致；
    #   本地工具不在此映射，而是在 my_tools.py 的 LOCAL_TOOLS 中注册。
    server_names = {
        "data_analyzer": "data-analyzer",
        "ticket_booking": "ticket-booking",
    }

    # 未知的工具组名 → 跳过
    if group_name not in server_names:
        print(f"未知 MCP 工具组 {group_name}，已跳过。")
        return []

    # ★ mcp_client 可能为 None（mcp_tool_config.py 初始化失败时降级）
    if mcp_client is None:
        print(f"MCP 客户端不可用，跳过工具组 {group_name}。")
        return []

    # 通过 MCP 客户端拉取远端工具列表
    server_name = server_names[group_name]
    try:
        tools = await mcp_client.get_tools(server_name=server_name)
        _print_tools(f"MCP[{server_name}] / 工具组[{group_name}]", tools)
        return tools
    except Exception as exc:
        # 远端服务异常时打印警告并返回空列表，不阻断整个流程
        print(f"加载 MCP 工具组 {group_name} 失败，已跳过对应子 Agent: {exc}")
        return []


# =====================================================================
# 4. 子 Agent 配置加载 —— 读取 YAML 配置，组装子 Agent 列表
# =====================================================================
async def load_subagents(config_path: str) -> list[dict]:
    """通过读取 YAML 配置文件，加载并组装子 Agent 列表。

    配置文件格式示例（subagents.yaml）::

        chart-agent:
          description: 专门负责数据分析...
          system_prompt: |
            你是图表生成专家...
          tools:
            - data_analyzer          # MCP 工具组名，会通过 _load_tool_group 加载

        researcher:
          description: 专门负责网络信息研究...
          system_prompt: |
            你是研究助手...
          tools:
            - web_search             # 本地工具名，会通过 LOCAL_TOOLS 映射解析

    流程：
    1. 读取 YAML 配置，遍历每个子 Agent 定义；
    2. 按 tools 列表解析工具：优先查 LOCAL_TOOLS 本地映射，找不到再走 MCP 加载；
    3. 若子 Agent 声明了工具但全部加载失败，则跳过该子 Agent；
    4. 组装为 create_deep_agent 所需的 dict 列表。

    Args:
        config_path: subagents.yaml 配置文件的绝对路径。

    Returns:
        子 Agent 字典列表，每个字典包含 name / description / system_prompt，
        可选包含 model 和 tools。
    """
    # 读取 YAML 配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 工具组缓存：同名 MCP 工具组只向服务端请求一次
    tool_groups: dict[str, list[Any]] = {}
    subagents = []

    # 遍历每个子 Agent 定义，逐个解析工具列表
    for name, spec in config.items():
        tools = []
        for group_name in spec.get("tools", []):
            # 优先从本地工具映射中查找
            if group_name in LOCAL_TOOLS:
                tools.append(LOCAL_TOOLS[group_name])
                continue

            # 本地找不到，走 MCP 远端加载（懒加载 + 缓存）
            if group_name not in tool_groups:
                tool_groups[group_name] = await _load_tool_group(group_name)

            # 将列表中的每个元素逐个添加
            # 示例：tools.extend([a, b]) → 结果 [a, b]（扁平列表）
            tools.extend(tool_groups[group_name])

        # 声明了工具却全部加载失败 → 该子 Agent 无法工作，跳过
        if spec.get("tools") and not tools:
            print(f"子 Agent {name} 没有可用工具，已跳过。")
            continue

        # 组装子 Agent 字典
        subagent = {
            "name": name,
            "description": spec["description"],
            "system_prompt": spec["system_prompt"],
        }

        # 可选字段：自定义模型（不指定则继承主 Agent 的模型）
        if "model" in spec:
            subagent["model"] = spec["model"]
        if tools:
            subagent["tools"] = tools

        subagents.append(subagent)

    return subagents


# =====================================================================
# 5. 主 Agent 创建 —— 绑定本地工具和子 Agent 列表，启动多层协作
# =====================================================================
async def create_agent():
    """创建主 Agent 实例，绑定本地工具和子 Agent 列表。

    架构概览::

        主 Agent (glm_llm)
        ├── 本地工具: web_search
        ├── 子 Agent: chart-agent   ← MCP data-analyzer
        ├── 子 Agent: ticket-agent  ← MCP ticket-booking + web_search  ★
        └── 子 Agent: researcher    ← web_search

    后端使用 FilesystemBackend（虚拟模式），支持子 Agent 通过
    write_file 保存研究报告到本地文件系统。

    Returns:
        DeepAgents 框架的 Agent 实例，可直接调用 .invoke() 等方法。
    """

    # 加载 subagents.yaml 中定义的所有子 Agent
    sub_agent = await load_subagents(str(EXAMPLE_DIR / 'subagents.yaml'))

    # 打印主 Agent 可用的本地工具清单
    _print_tools("主 Agent 本地工具", [web_search])

    # 创建主 Agent：注入共享模型、本地工具、记忆文件、文件后端和子 Agent 列表
    return create_deep_agent(
        model=glm_llm,
        tools=[web_search],
        memory=['/AGENTS.md'],
        backend=FilesystemBackend(root_dir=EXAMPLE_DIR, virtual_mode=True),
        subagents=sub_agent
    )


# =====================================================================
# 6. 模块级初始化 —— import 时即完成 Agent 创建
# =====================================================================
# 模块被 import 时即完成 Agent 创建，方便其他模块直接引用 agent 变量
agent = asyncio.run(create_agent())

"""
测试，不要删除：
1.帮我分析下中国近期七日进口牛肉占比，以国家去分组，从小到大排序，帮我生成一个鱼骨图
2.我要在七日后从上海到北京，先帮我看下高铁票或者火车票，我需要在当天下午三点前到达北京，我本人在上海松江，请帮我规划一条最佳合理的出行路线图。
3.帮我对比分析下 Python 语言和 Java 语言对于构建企业级Agent 的优劣势
"""