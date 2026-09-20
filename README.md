# 带钢宽度缺陷智能诊断系统

基于大小模型协同的工业缺陷诊断 Agent，支持多轮对话、批量 CSV 诊断、知识问答和 RAGAS 量化评估。  
系统采用 LangGraph 构建 ReAct Agent，将诊断模型（双分支混合卷积神经网络）、知识检索、相似案例匹配、
特征重要性分析等封装为工具，由 LLM 动态调度，实现从缺陷识别到根因分析的端到端诊断。

注意：本仓库仅包含代码与示例数据，真实生产数据、模型权重及完整知识库未公开。

主要功能
- 单条/批量诊断：上传与 `data/width.csv` 同格式的带钢数据（STRIPNO + 6 工艺参数 + 3 段全长宽度曲线），
  服务自动计算 6 个机理标志位并诊断；批量结果可一键导出 CSV。
- 置信度门限与人工复核：最高概率 < 0.6 或前两类概率差 < 0.2 的样本自动标记"需人工复核"，
  不强行给出确定性结论。
- 梯度归因证据链：每条带钢输出 Top 证据特征（梯度×输入）与曲线段关注度，并匹配根因知识模板
  生成诊断建议，结论可审计。
- 专家 Agent：LLM 自主规划调用诊断、检索、相似案例、特征重要性、历史诊断等工具，多步推理；
  LLM 调用带超时与自动重试。
- 多轮对话记忆：支持连续追问，Agent 自动总结历史对话并复用已有结果。
- 长期记忆（Redis/MySQL 可选）：会话消息与检索缓存走 Redis；单卷诊断记录可落 MySQL，
  支持按故障类型/时间 SQL 查询；Redis → 文件 → 内存三级降级。
- 历史诊断查询：按追溯键（STRIPNO + 12 维机理特征哈希）反查该卷历史诊断记录（跨会话）。
- 知识库管理：在线更新技术文档（双栏 PDF 友好）与故障四元组，实时生效。
- REST API：FastAPI 服务（单条/批量诊断 / 知识问答），支持 session_id 会话持久化，供 MES/ERP 集成。
- RAGAS 评估：内置评估脚本，可量化忠实性、上下文精度/召回等指标（评估 Agent 真实链路，用法见下文 RAGAS 评估）。

目录结构
src/
  api.py              # FastAPI REST 服务（供 MES/ERP 集成）
  service.py          # 核心服务层：模型/检索/预测/Agent，前端与 API 共用
  hybrid_model.py     # 诊断模型推理服务（加载权重、flag 现算、置信度门限、梯度归因）
  cnn_model.py        # 双分支混合CNN结构（曲线卷积分支 + 12维特征分支）与预测/归因函数
  features.py         # 机理特征提取：6个规则标志位按原版 te.py 逻辑现算（+71维实验特征）
  tools.py            # Agent 工具集（诊断/检索/相似案例/特征重要性/批量/历史查询）
  agent_graph.py      # LangGraph ReAct Agent 图（LLM 超时 + 自动重试）
  memory_store.py     # 长期记忆存储（Redis → 文件 → 内存三级降级）
  kb_utils.py         # 知识库解析公共工具（双栏 PDF、特征重排、四元组）
  build_kb.py         # 知识库全量构建（文档分块 + 四元组入 Milvus）
  rebuild_feature_vectors.py  # 仅重建故障特征向量集合
  kb_audit.py         # 知识库体检（只读：文档质量指标 + 库内垃圾块溯源）
  eval_generate.py    # RAGAS 评估数据生成（走 Agent 真实推理链路）
  eval_ragas.py       # RAGAS 指标评估（适配 ragas 0.4.x）
  logger.py           # 统一日志（控制台 + logs/app.log）
  app.py              # Streamlit 旧入口（保留，新功能以 API + 前端为主）
app_file_uploader.py  # 知识库在线更新界面

准备模型文件
将以下文件放入 models/ 目录：
cnn_hybrid_diag.pt            # 诊断模型权重（双分支混合CNN，单种子）
cnn_hybrid_importance.json    # 按类梯度归因（特征重要性工具的数据源）
model_config.pkl              # 仅含 12 维 scaler，供 build_kb / rebuild_feature_vectors 使用
bge-large-zh-v1.5/            # 嵌入模型文件夹
bge-reranker-v2-m3/           # 重排序模型文件夹

构建知识库（需要文档与四元组）
python src/build_kb.py
启动诊断系统
方式一（推荐）：FastAPI 后端 + Web 前端
uvicorn src.api:app --host 0.0.0.0 --port 8000

方式二：Streamlit 界面
streamlit run src/app.py
浏览器访问 http://localhost:8501 即可使用。上传与 width.csv 同格式的 CSV 后，
可在对话中让 Agent 完成批量诊断与逐条追问（页面「历史诊断查询」仅支持旧格式记录）。

运行日志输出到 `logs/app.log`（自动创建）；上传的 CSV 临时文件存放于 `data/tmp_uploads/`，超过 24 小时自动清理。

---

## REST API（供 MES/ERP 集成）

除 Web 界面外，系统提供 FastAPI 服务，外部系统可通过 HTTP 调用诊断能力（单条/批量诊断、知识问答）。

启动 API 服务：

```powershell
$env:DEEPSEEK_API_KEY = "sk-xxx"   # 使用 /api/v1/query 时必需；诊断接口不需要
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

接口一览（完整文档见 http://localhost:8000/docs）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查（模型/Redis/LLM 配置状态） |
| GET | `/api/v1/meta/features` | 诊断输入必需列与故障类型清单 |
| POST | `/api/v1/diagnose` | 单条诊断：带钢记录 JSON → 故障类型/置信度/需人工复核/归因证据 |
| POST | `/api/v1/diagnose/batch` | 批量诊断：带钢记录数组 → 逐条结果 + 分布统计 |
| POST | `/api/v1/diagnose/batch/file` | 批量诊断：上传 CSV 文件（multipart，推荐） |
| POST | `/api/v1/query` | LLM 知识问答 / Agent 多步诊断（需 API Key，支持 session_id 会话持久化） |
| GET | `/api/v1/diag/history` | 单卷诊断历史查询：`?fault_cn=&days=&limit=`（需启用 MySQL） |

单条诊断示例（curl）：

```bash
curl -X POST http://localhost:8000/api/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{"strip": {"STRIPNO": "220416900800",
                 "FMWTARGETHOT": 1197.53, "RDWTARGETTOTAL": 1202.53, "PDIWIDTHTOL": 8.55,
                 "FMWIDTHACTHOT": 1196.74, "RMWIDTHACTHOT": 1200.15, "FMWTARGETCOL": 1183.55,
                 "RMDATASEQ": "1201.3,1200.8,1199.5,...",
                 "FMDATASEQ": "1198.6,1197.9,...",
                 "DCDATASEQ": "1195.2,1194.8,..."}}'
```

响应示例：

```json
{
  "request_id": "3f9a2c1e8b4d",
  "fault_cn": "整体窄",
  "fault_desc": "精轧自然宽展偏差或给定PDI不合适导致精轧整体窄",
  "confidence": 0.9982,
  "needs_review": false,
  "evidence": "整体窄(+9.99)；精轧目标宽(冷)(+2.57)｜曲线关注:精轧段",
  "probabilities": {"整体窄": 0.9982, "...": 0.0009},
  "top_features": [{"feature": "整体窄", "importance": 3.49}, "..."]
}
```

批量诊断（CSV 上传）：

```bash
curl -X POST http://localhost:8000/api/v1/diagnose/batch/file \
  -F "file=@test_samples.csv"
```

CSV 列要求：STRIPNO(可选) + 6 个工艺参数 + 3 段全长宽度曲线；6 个 `is_*` 规则标志位
由服务按机理逻辑自动计算，无需提供。响应中 `needs_review` 为 `true` 的样本表示
置信度不足，建议人工确认。

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

### Agent 工具返回契约

所有 Agent 工具统一返回 JSON：成功时含 `success: true` 与业务字段；失败时含
`success: false`、`error_code`、`message`、`recoverable` 与 `recommended_action`
（建议的恢复动作：追问用户 / 换角度重试 / 停止说明）。

- 诊断类工具（`diagnose_strip_tool` / `batch_csv_diagnosis_tool`）的输入为带钢原始记录；
  6 个 `is_*` 机理标志位由服务按原版逻辑自动计算，无需外部提供；
- 批量诊断的 `details` 每条含 `features_named`（12 维机理特征带名字典：六个宽度参数与
  六个自动计算的 `is_*` 标志位）与 `needs_review`、`归因证据` 字段，解读特征值以
  `features_named` 为准；
- details 样本数超过 `RAG_BATCH_DETAIL_LIMIT`（默认 50）时截断并标记
  `details_truncated`，完整逐卷结果自动写入长期记忆，可用 `query_hist_diag_tool`
  按追溯键反查。

相关环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_BATCH_DETAIL_LIMIT` | `50` | 批量诊断返回给 LLM 的逐条详情上限，完整结果不受影响 |

---

## RAGAS 评估（可选）

评估的是当前系统的真实行为：评估问题集（`data/eval_questions.json`）→ Agent
完整推理生成回答与检索上下文 → ragas 打分，而非单独的"检索 + 生成"旧链路。

```powershell
$env:DEEPSEEK_API_KEY = "sk-xxx"
python src/eval_generate.py   # 生成 data/eval_dataset.json（逐题调用 Agent）
python src/eval_ragas.py      # 输出 data/ragas_eval.csv 与指标摘要
```

- 指标：faithfulness / answer_relevancy / context_precision / context_recall
- 数据集字段为 ragas 0.4 schema：`user_input` / `reference` / `retrieved_contexts` / `response`
- 评估 LLM 的超时与重试复用 `LLM_TIMEOUT` / `LLM_MAX_RETRIES` 环境变量
- 依赖 ragas==0.4.3（requirements.txt 已锁定）

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
| `RAG_MYSQL_HOST` | 空（不启用） | MySQL 主机，如 `127.0.0.1`；设置后启用诊断记录结构化存储 |
| `RAG_MYSQL_PORT` | `3306` | MySQL 端口 |
| `RAG_MYSQL_USER` | `rag` | MySQL 用户 |
| `RAG_MYSQL_PASSWORD` | `rag123456` | MySQL 密码 |
| `RAG_MYSQL_DB` | `rag_agent` | MySQL 数据库（表 `rag_diag` 自动创建） |

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
| `rag:diag:{feat_hash}` | 单卷诊断记录（按 STRIPNO + 12 维机理特征哈希索引） | 30 天 |
| `rag:cache:retrieve:{sha1}` | 检索结果缓存（知识库更新后自动失效） | 24 小时 |

会话 ID（`sid`）由 URL 参数 `?sid=` 指定：首次访问自动生成，之后用**同一个带 sid 的网址**打开即可跨重启恢复完整对话历史。

单卷诊断记录支持按追溯键反查：在对话中让 Agent 调用 `query_hist_diag_tool`（输入带钢记录，
服务自动计算 STRIPNO + 12 维追溯键），即可查询该卷此前是否诊断过及诊断结论。

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

## MySQL 诊断记录库（可选，推荐启用）

单卷诊断记录默认只按追溯键存取（仅支持精确反查）。启用 MySQL 后，诊断记录落
`rag_diag` 表，可按**故障类型、时间范围、置信度**做 SQL 查询，便于 MES 追溯与统计分析。

### 1. 初始化数据库与账号（一次性，需 root 密码）

```powershell
mysql -u root -p < sql/init_mysql.sql
```

脚本会创建数据库 `rag_agent` 与专用账号 `rag/rag123456`；数据表由应用首次启动时自动创建。

### 2. 配置环境变量并启动

```powershell
$env:RAG_MYSQL_HOST = "127.0.0.1"
$env:RAG_MYSQL_USER = "rag"
$env:RAG_MYSQL_PASSWORD = "rag123456"
$env:RAG_MYSQL_DB = "rag_agent"
streamlit run src/app.py
```

启用后，Web 界面「历史诊断查询」会额外提供“最近诊断记录”查询（按故障类型/天数），
REST API 提供 `GET /api/v1/diag/history`。MySQL 不可用时自动回落到 Redis/文件，不影响运行。

---

### 知识库体检（只读）

```powershell
python src/kb_audit.py
```

1. 逐 PDF 统计碎行率、平均行长、全角混用率等质量指标，输出问题文档清单
   （扫描件 / 低质量 OCR 文本层会在入库前暴露）；
2. 审计 Milvus `rag_knowledge` 已存文本块，垃圾块按来源文档聚合排名。

报告输出至 `data/kb_audit_report.csv`，判定阈值仅供筛查参考，脚本不修改任何数据。
建议在每次批量更新文档或重建知识库后运行一次，及时发现问题来源文档。


<img width="1746" height="7175" alt="QQ_1789884626675" src="https://github.com/user-attachments/assets/a809acab-7daa-47a9-83c3-acdc74796da3" />
