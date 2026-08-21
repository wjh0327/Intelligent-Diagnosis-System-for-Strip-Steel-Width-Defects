# 带钢宽度缺陷智能诊断系统

基于大小模型协同的工业缺陷诊断 Agent，支持多轮对话、批量 CSV 诊断、知识问答和 RAGAS 量化评估。  
系统采用 LangGraph 构建 ReAct Agent，将故障分类模型、知识检索、相似案例匹配、特征重要性分析等封装为工具，由 LLM 动态调度，实现从故障识别到根因分析的端到端诊断。

注意：本仓库仅包含代码与示例数据，真实生产数据、模型权重及完整知识库未公开。

主要功能
- 单条/批量诊断：输入 12 维特征值或上传 CSV 文件，自动分类并生成诊断报告；批量结果可一键导出 CSV。
- 专家 Agent：LLM 自主规划调用分类、检索、相似案例、特征重要性、历史诊断等工具，多步推理；LLM 调用带超时与自动重试。
- 多轮对话记忆：支持连续追问，Agent 自动总结历史对话并复用已有结果。
- 长期记忆（Redis 可选）：会话消息、批量诊断、单卷诊断与检索缓存跨重启持久化，Redis → 文件 → 内存三级降级。
- 历史诊断查询：输入 12 维特征即可反查该卷历史诊断记录（跨会话追溯）。
- 知识库管理：在线更新技术文档（双栏 PDF 友好）与故障四元组，实时生效。
- REST API：FastAPI 服务（故障分类 / 批量诊断 / 知识问答），支持 session_id 会话持久化，供 MES/ERP 集成。
- RAGAS 评估：内置评估脚本，可量化忠实性、上下文精度/召回等指标。

目录结构
src/
  app.py              # Streamlit 前端（复用 service 核心层）
  api.py              # FastAPI REST 服务（供 MES/ERP 集成）
  service.py          # 核心服务层：模型/检索/预测/Agent，前端与 API 共用
  kb_utils.py         # 知识库解析公共工具（双栏 PDF、特征重排、四元组）
  logger.py           # 统一日志（控制台 + logs/app.log）
  tools.py            # Agent 工具集（分类/检索/相似案例/特征重要性/批量/历史查询）
  agent_graph.py      # LangGraph ReAct Agent 图（LLM 超时 + 自动重试）
  memory_store.py     # 长期记忆存储（Redis → 文件 → 内存三级降级）
app_file_uploader.py  # 知识库在线更新界面

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

运行日志输出到 `logs/app.log`（自动创建）；上传的 CSV 临时文件存放于 `data/tmp_uploads/`，超过 24 小时自动清理。

---

## REST API（供 MES/ERP 集成）

除 Streamlit 界面外，系统提供 FastAPI 服务，外部系统可通过 HTTP 调用诊断能力（故障分类、批量诊断、知识问答）。

启动 API 服务：

```powershell
$env:DEEPSEEK_API_KEY = "sk-xxx"   # 使用 /api/v1/query 时必需；诊断接口不需要
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

接口一览（完整文档见 http://localhost:8000/docs）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查（模型/Redis/LLM 配置状态） |
| GET | `/api/v1/meta/features` | 特征字段顺序与故障类型清单 |
| POST | `/api/v1/diagnose` | 单条诊断：12 维特征向量 → 故障类型/置信度/关键特征 |
| POST | `/api/v1/diagnose/batch` | 批量诊断：JSON 数组 → 逐条结果 + 分布统计 |
| POST | `/api/v1/diagnose/batch/file` | 批量诊断：上传 CSV 文件（multipart） |
| POST | `/api/v1/query` | LLM 知识问答 / Agent 多步诊断（需 API Key，支持 session_id 会话持久化） |

单条诊断示例（curl）：

```bash
curl -X POST http://localhost:8000/api/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{"features": [1,1,1,0,0,1,1247.3,1257.3,10,1239.8,1257.6,1234.0]}'
```

响应示例：

```json
{
  "request_id": "3f9a2c1e8b4d",
  "fault_cn": "整体窄",
  "fault_desc": "精轧自然宽展偏差或给定PDI不合适导致精轧整体窄",
  "confidence": 0.7069,
  "probabilities": {"整体窄": 0.7069, "...": 0.0123},
  "top_features": [{"feature": "is_FM", "importance": 0.31}, "..."]
}
```

批量诊断（CSV 上传）：

```bash
curl -X POST http://localhost:8000/api/v1/diagnose/batch/file \
  -F "file=@test_samples.csv"
```

CSV 列名须与 `/api/v1/meta/features` 返回的特征顺序一致（或直接使用特征列名）。

知识问答（带会话持久化）：

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "整体窄故障如何处理？", "session_id": "3f9a2c1e8b4d6a7f8e9d0c1b2a3f4e5d"}'
```

`session_id` 为 32 位十六进制字符串。提供后，API 会自动：

1. 从 MEMORY（Redis/文件）读取该会话的对话历史与批量诊断状态（`batch_done`/`batch_summary`）；
2. 问答结束后把新增对话与最新批量状态写回 MEMORY，外部系统无需每次传入完整历史。

请求体还支持可选的 `conversation_history`、`batch_done`、`batch_summary` 字段；未提供时优先使用 MEMORY 中保存的状态。不带 `session_id` 时为无状态模式，与旧行为一致。

### LLM 调用参数（环境变量，均可选）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_TIMEOUT` | `60`（秒） | LLM 单次调用超时 |
| `LLM_MAX_RETRIES` | `3` | LLM 调用失败自动重试次数 |
| `LLM_RETRY_BACKOFF` | `2.0`（秒） | 重试退避基数（第 n 次等待 n × backoff 秒） |

---

## Redis 长期记忆（可选，推荐启用）

系统内置三层记忆降级策略，启动时自动选择可用后端，无需改代码：

1. **Redis**：首选。连接成功即启用，支持跨进程、跨重启持久化（默认 TTL 30 天）。
2. **本地文件**：Redis 不可用时自动落到 `data/memory/` 下的 JSON 文件，同样跨重启持久化。
3. **内存**：以上均不可用时退化为进程内 dict（仅当前进程有效）。

Redis 未安装或未启动时，应用照常运行，页面顶部会显示“文件长期记忆”或“仅内存记忆”，不会报错。

### 安装 Redis（WSL2 Ubuntu 方案）

Windows 需先启用“适用于 Linux 的 Windows 子系统”与“虚拟机平台”两个可选功能并重启：

```powershell
dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

安装 Ubuntu 发行版：

```powershell
wsl --install -d Ubuntu
```

> 已知坑：WSL 2.7.x 存在安装时未部署 `system.vhd` / `modules.vhd` 的缺陷，导入或启动发行版会报
> `Wsl/Service/RegisterDistro/CreateVm/HCS/ERROR_FILE_NOT_FOUND`。
> 解决方案：降级安装官方 WSL 2.6.3 MSI
> （`github.com/microsoft/WSL/releases/download/2.6.3/wsl.2.6.3.0.x64.msi`），再重新导入发行版。

Ubuntu 内安装并启动 Redis：

```bash
apt update && apt install -y redis-server
sed -i 's/^# *appendonly no/appendonly yes/' /etc/redis/redis.conf   # 开启 AOF 持久化
service redis-server start
redis-cli ping   # 返回 PONG 即成功
```

让 Redis 随 WSL 启动自动拉起（任选其一）：

```bash
# 方式一：systemd（WSL 1.0+ 支持）
systemctl enable redis-server

# 方式二：wsl.conf 启动命令
printf '\n[boot]\ncommand = /usr/sbin/service redis-server start\n' >> /etc/wsl.conf
```

Windows 登录时自动启动 WSL 中的 Redis（计划任务）：

```powershell
schtasks /create /tn "WSL Redis Autostart" /tr "C:\Windows\System32\wsl.exe -d Ubuntu -u root -e /usr/sbin/service redis-server start" /sc onlogon /rl limited /delay 0000:30 /f
```

> 注意：命令必须使用绝对路径 `/usr/sbin/service`。`wsl -e` 与 wsl.conf 的 boot 命令使用精简 PATH，
> 不带路径的 `service` 会报 `execvpe(service) failed: No such file or directory` 导致自启失败。

### 依赖

`requirements.txt` 已包含 `redis>=5.0`（redis-py 客户端），安装依赖后即可使用。

### 配置（环境变量，均可选）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis 连接串 |
| `RAG_MEMORY_TTL_SECONDS` | `2592000`（30 天） | 会话消息与诊断记录存活时间 |
| `RAG_MAX_STORED_MESSAGES` | `100` | 每个会话最多保留的消息条数 |
| `RAG_MEMORY_DIR` | `data/memory` | 文件降级目录（仅 Redis 不可用时使用） |

示例（Windows PowerShell）：

```powershell
$env:RAG_REDIS_URL = "redis://127.0.0.1:6379/0"
streamlit run src/app.py
```

### 存储结构

| Redis Key | 内容 | TTL |
| --- | --- | --- |
| `rag:session:{sid}:messages` | 会话消息（JSON，最近 100 条） | 30 天 |
| `rag:session:{sid}:batch` | 批量诊断状态与汇总 | 30 天 |
| `rag:diag:{feat_hash}` | 单卷诊断记录（按 12 维特征哈希索引） | 30 天 |
| `rag:cache:retrieve:{sha1}` | 检索结果缓存（知识库更新后自动失效） | 24 小时 |

会话 ID（`sid`）由 URL 参数 `?sid=` 指定：首次访问自动生成，之后用**同一个带 sid 的网址**打开即可跨重启恢复完整对话历史。

单卷诊断记录支持按特征反查：在 Web 界面「历史诊断查询」输入 12 维特征（或让 Agent 调用
`query_hist_diag_tool`），即可查询该卷此前是否诊断过及诊断结论。

### 查看记忆与缓存

```bash
wsl -d Ubuntu -u root
redis-cli --scan --pattern 'rag:*'                  # 列出全部项目键
redis-cli TTL rag:session:{sid}:messages            # 剩余存活时间
redis-cli GET  rag:session:{sid}:messages           # 查看会话消息原文
redis-cli DEL  <key>                                # 删除单个键
redis-cli FLUSHDB                                   # 清空整个库（会清掉全部记忆，慎用）
```

---


<img width="1738" height="12628" alt="QQ_1787191717918" src="https://github.com/user-attachments/assets/1c8c9b13-bd1f-4b7a-b8cb-795b649d99b5" />


