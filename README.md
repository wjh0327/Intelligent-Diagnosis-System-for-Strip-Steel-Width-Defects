# 带钢宽度缺陷智能诊断系统

基于大小模型协同的工业缺陷诊断 Agent，支持多轮对话、批量 CSV 诊断、知识问答和 RAGAS 量化评估。  
系统采用 LangGraph 构建 ReAct Agent，将故障分类模型、知识检索、相似案例匹配、特征重要性分析等封装为工具，由 LLM 动态调度，实现从故障识别到根因分析的端到端诊断。

注意：本仓库仅包含代码与示例数据，真实生产数据、模型权重及完整知识库未公开。
主要功能
单条/批量诊断：输入 12 维特征值或上传 CSV 文件，自动分类并生成诊断报告。
专家 Agent：LLM 自主规划调用分类、检索、相似案例、特征重要性等工具，多步推理。
多轮对话记忆：支持连续追问，Agent 自动总结历史对话并复用已有结果。
知识库管理：在线更新技术文档（双栏 PDF 友好）与故障四元组，实时生效。
RAGAS 评估：内置评估脚本，可量化忠实性、上下文精度/召回等指标。

准备模型文件
将以下文件放入 models/ 目录：
model_config.pkl # 分类模型配置（含 scaler）
best_traceability_model.pth # 分类模型权重
bge-large-zh-v1.5/ # 嵌入模型文件夹
bge-reranker-v2-m3/ # 重排序模型文件夹

构建知识库（需要文档与四元组）
python src/build_kb.py
启动诊断系统
streamlit run src/app.py
浏览器访问 http://localhost:8501 即可使用。

<img width="1736" height="7014" alt="QQ_1786253632752" src="https://github.com/user-attachments/assets/ebf7a0d0-6589-495a-9cfd-ada0d80a90ed" />


