import asyncio
from typing import AsyncIterator

from agent.backend.CustomizeBackend import agent


# from agent.backend.Composite import agent  # CompositeBackend 组合后端
# from agent.backend.Store import agent  # StoreBackend 持久化存储后端
# from agent.backend.LocalShell import agent  # LocalShellBackend 本地Shell后端
# from agent.backend.Filesystem import agent  # FilesystemBackend 本地磁盘后端


# =====================================================================
# 1. 启动流式交互 —— 循环接收用户输入并输出 Agent 回复
# =====================================================================

async def stream_agent_interaction_corrected(thread_id: str) -> AsyncIterator[str]:
    """
    使用官方推荐的 `agent.stream()` 方法进行流式交互。
    根据调试信息，chunk的结构是 (AIMessageChunk, metadata_dict)

    thread_id 用于标识同一轮会话；当前主要作为运行配置传入。
    如果后续再加入 checkpointer，同一个 thread_id 才可以帮助恢复对应的对话状态。
    """
    # config 会随每次调用传给 LangGraph / Deep Agent，用来标识当前会话线程。
    config = {"configurable": {"thread_id": thread_id}}

    while True:  # 多轮交换
        try:
            user_input = input("\n\n[用户] >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n对话结束。")
            break

        if user_input.lower() in ('quit', 'exit', '退出', 'q'):
            print("再见！")
            break
        if not user_input:
            continue

        print("\n[Agent] >>>>", end="", flush=True)
        # Deep Agent 接收的是 LangChain 消息格式，这里把命令行输入包装成 user message。
        inputs = {"messages": [{"role": "user", "content": user_input}]}

        try:
            # agent.stream() 返回同步生成器；外层函数是异步生成器，所以这里用普通 for 消费流式片段。
            # stream_mode="messages" 表示按消息片段返回：模型每生成一段内容，这里就会拿到一个 chunk。
            for chunk in agent.stream(inputs, config=config, stream_mode="messages", subgraphs=False):
                # 当前 deepagents 返回的 chunk 通常是 (token, metadata)：
                # token 是模型输出片段，metadata 是附带的运行元信息；本示例只使用 token。
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    token, metadata = chunk

                    # 1. 流式输出 AI 生成的文本内容
                    if hasattr(token, 'content') and token.content is not None:
                        content_str = str(token.content)
                        if content_str:
                            # yield 会把当前这一小段文本先交给外层 async for 打印，
                            # 然后函数暂停在这里，下一次有新 chunk 时再继续往下执行。
                            # 这样终端能实时看到流式输出，而不是等完整回答结束后再一次性显示。
                            yield content_str

                    # 2. 捕获并显示工具调用开始
                    if hasattr(token, 'tool_call_chunks') and token.tool_call_chunks:
                        for tool_chunk in token.tool_call_chunks:
                            if tool_chunk and hasattr(tool_chunk, 'get') and tool_chunk.get('name'):
                                # 工具调用提示也用 yield 输出，这样它会和模型文本一起按发生顺序显示。
                                yield f"\n[调用工具: {tool_chunk['name']}]\n"
                else:
                    # 如果 chunk 不是预期的元组结构，打印调试信息
                    print(f"\n[调试] 意外的 chunk 结构: {type(chunk)}")
        except Exception as e:
            # 出错时也通过 yield 交给外层统一打印，避免交互循环直接中断。
            yield f"\nAgent 执行出错: {e}\n"


# =====================================================================
# 2. 运行命令行入口 —— 直接运行该文件即可进入循环对话
# =====================================================================
async def main() -> None:
    # 测试运行Agent，并且进行交互
    thread_id = "demo_thread_01"
    async for response in stream_agent_interaction_corrected(thread_id):
        print(response, end="", flush=True)


if __name__ == '__main__':
    asyncio.run(main())
