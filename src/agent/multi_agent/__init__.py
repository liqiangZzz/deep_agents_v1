"""
multi_agent 包 —— 主 Agent + 子 Agent 多层协作架构
====================================================

架构概览::

    主 Agent (glm_llm)
    ├── 本地工具: web_search
    ├── 子 Agent: chart-agent   ← MCP data-analyzer
    ├── 子 Agent: ticket-agent  ← MCP ticket-booking + web_search
    └── 子 Agent: researcher    ← web_search

模块说明
--------
mcp_tool_config
    MCP 客户端配置与初始化。定义 data-analyzer、ticket-booking 两个远端服务连接，
    创建 MultiServerMCPClient 实例。初始化失败时降级为 None，不影响本地工具。
    导出: ``mcp_client``

main_agent
    主 Agent 创建入口。读取 subagents.yaml 组装子 Agent 列表，
    绑定本地工具、文件后端和子 Agent，import 时自动完成创建。
    导出: ``agent``, ``create_agent``, ``load_subagents``
"""