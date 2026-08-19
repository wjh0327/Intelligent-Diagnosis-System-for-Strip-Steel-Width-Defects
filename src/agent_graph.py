# src/agent_graph.py
"""
LangGraph Agent 图 —— ReAct 风格多步推理诊断
支持历史对话、CSV批量诊断持久化摘要（保留全部详情）。
"""

import json
import os
import glob
from typing import TypedDict, List, Annotated, Dict
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    query: str
    features: str
    csv_path: str
    conversation_history: List[Dict[str, str]]
    intermediate_steps: Annotated[List[tuple], operator.add]
    final_answer: str
    step_count: Annotated[int, operator.add]
    batch_done: bool
    batch_summary: str             # 新增：批量诊断的完整结果，跨轮次持久化


def build_agent_graph(
    tools: List,
    model_name: str = "deepseek-v4-flash",
    api_key: str = None,
    temperature: float = 0.1,
    max_steps: int = 6,
    skills_dir: str = None,
) -> StateGraph:
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("未设置 DEEPSEEK_API_KEY")

    # 加载 Skill Markdown 规则
    skill_prompts = ""
    if skills_dir and os.path.isdir(skills_dir):
        for md_file in glob.glob(os.path.join(skills_dir, "*.md")):
            with open(md_file, "r", encoding="utf-8") as f:
                skill_prompts += f.read() + "\n"

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com",  # DeepSeek 官方 API
        temperature=temperature,
    )
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}

    base_prompt = (
        "你是带钢热轧宽度控制专家，可以根据用户输入和历史对话自主选择工具完成任务。\n"
        "你会收到一段历史对话记录（JSON数组），请先快速总结与当前问题相关的信息。\n"
        "如果历史中已有诊断结果（例如批量CSV诊断的故障分布），可直接基于这些信息回答，"
        "不要重复调用 batch_csv_diagnosis_tool 或 classify_fault_tool。\n"
        "可用工具：\n"
        "1. classify_fault_tool：输入12维特征（逗号分隔）预测故障类型（仅允许调用一次！）\n"
        "2. retrieve_knowledge_tool：根据文本查询搜索知识库\n"
        "3. search_similar_cases_tool：输入12维特征查找相似历史案例\n"
        "4. get_feature_importance_tool：查询某故障类型的关键特征\n"
        "5. batch_csv_diagnosis_tool：对上传的CSV文件进行批量诊断\n\n"
        "决策规则：\n"
        "- **CSV批量诊断优先**：如果用户提供了CSV文件路径（csv_path非空），且历史中尚未执行过批量诊断（batch_done为False），"
        "  你必须首先调用 batch_csv_diagnosis_tool，禁止在此时调用 classify_fault_tool 逐条处理。\n"
        "- 完成批量诊断后，如果用户询问整体故障分布或统计信息，直接基于批量诊断的返回结果回答，不要再次调用工具。\n"
        "- 如果用户针对CSV中某条具体数据提问，可结合已有的批量诊断详情或调用 retrieve_knowledge_tool / get_feature_importance_tool 来解释，"
        "  但严禁再次调用 classify_fault_tool。\n"
        "- 如果用户提供了12个单独的数字特征（非CSV场景），且历史中未对该特征进行分类，则调用 classify_fault_tool。\n"
        "- **空结果处理**：如果 retrieve_knowledge_tool 返回空字符串或无有效内容，"
        "  禁止用相同 query 重试；应改用不同的查询角度（如改用故障中文名、故障机理关键词、或故障调整方法）最多重试一次，"
        "  若仍为空则直接基于已有知识给出尽可能好的回答，不要继续调用工具。\n"
        "- **批量诊断完成后的行为**：如果 batch_summary 非空，"
        "  若用户询问与本次诊断相关（如某卷的具体问题、整体分布、某故障类型占比），优先基于 batch_summary 直接回答，无需调用工具；"
        "  若用户询问某故障类型的通用知识（如「XX是怎么样的」「XX应如何处理」），可以调用 retrieve_knowledge_tool 补充说明；"
        "  严禁再次调用 batch_csv_diagnosis_tool 或 classify_fault_tool。\n"
        "- 总工具调用次数不超过{max_steps}次，之后必须给出最终答案。\n"
        "- classify_fault_tool 一旦调用，绝对禁止再次调用。\n"
    ).format(max_steps=max_steps)

    system_prompt = base_prompt + "\n" + skill_prompts

    def agent_node(state: AgentState) -> dict:
        # 步数已满 → 强制生成最终答案
        if state.get("step_count", 0) >= max_steps:
            history = "\n".join(
                [f"{name}: {output}" for name, output in state["intermediate_steps"]]
            )
            msgs = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"基于以下工具调用结果，请直接给出最终诊断结论：\n{history}"),
            ]
            resp = llm.invoke(msgs)
            return {"final_answer": resp.content}

        # 构建消息
        messages = [SystemMessage(content=system_prompt)]

        # 注入历史对话，让 LLM 自行总结
        conv_hist = state.get("conversation_history", [])
        if conv_hist:
            hist_truncated = conv_hist[-8:]  # 最多 8 条（4 轮）
            history_json = json.dumps(hist_truncated, ensure_ascii=False)
            history_prompt = (
                "以下是历史对话记录（JSON数组，每项包含 role 和 content）：\n"
                f"{history_json}\n"
                "请自行总结与当前问题相关的关键信息，并在后续决策中加以利用。"
            )
            messages.append(HumanMessage(content=history_prompt))

        # 注入持久化的批量诊断完整结果（如果存在）
        if state.get("batch_summary"):
            messages.append(
                HumanMessage(content=f"以下为此前批量CSV诊断的完整结果，请直接基于这些信息回答用户问题，无需再次调用批量诊断工具：\n{state['batch_summary']}")
            )

        # 用户当前问题
        user_content = state["query"]
        if state.get("features"):
            user_content = f"待诊断特征值：{state['features']}\n问题：{user_content}"
        if state.get("csv_path"):
            user_content = f"CSV文件路径：{state['csv_path']}\n用户需求：{user_content}"
        messages.append(HumanMessage(content=user_content))

        # 附加上一步工具调用结果
        for tool_name, tool_output in state.get("intermediate_steps", []):
            messages.append(
                HumanMessage(content=f"已执行工具 {tool_name}，返回结果如下：\n{tool_output}")
            )

        response = llm_with_tools.invoke(messages)

        # 无工具调用 → 最终答案
        if not response.tool_calls:
            return {"final_answer": response.content}

        # 执行第一个工具调用
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                tool_args = {}
        if isinstance(tool_args, dict):
            arg_value = list(tool_args.values())[0] if tool_args else ""
        else:
            arg_value = str(tool_args)

        tool_func = tool_map.get(tool_name)
        if tool_func is None:
            observation = f"工具 {tool_name} 不存在。"
        else:
            try:
                observation = tool_func.run(arg_value)
            except Exception as e:
                observation = f"工具执行失败: {str(e)}"

        # 禁止重复调用分类工具
        if tool_name == "classify_fault_tool":
            observation += "\n[系统强制规则] 你已经获得了故障分类结果，绝对禁止再次调用 classify_fault_tool。"

        # 批量诊断工具特殊处理：保存完整结果到 batch_summary
        if tool_name == "batch_csv_diagnosis_tool":
            return {
                "step_count": 1,
                "intermediate_steps": [(tool_name, observation)],
                "batch_done": True,
                "batch_summary": observation,   # 保存全部详情，供后续轮次使用
            }
        else:
            return {
                "step_count": 1,
                "intermediate_steps": [(tool_name, observation)],
            }

    def should_continue(state: AgentState) -> str:
        return "end" if state.get("final_answer") else "continue"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "agent", "end": END})

    return workflow.compile()