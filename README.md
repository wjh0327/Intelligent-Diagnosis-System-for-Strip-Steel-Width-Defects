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
printf '\n[boot]\ncommand = service redis-server start\n' >> /etc/wsl.conf
```

Windows 登录时自动启动 WSL 中的 Redis（计划任务）：

```powershell
schtasks /create /tn "WSL Redis Autostart" /tr "C:\Windows\System32\wsl.exe -d Ubuntu -u root -e service redis-server start" /sc onlogon /rl limited /f
```

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

### 查看记忆与缓存

```bash
wsl -d Ubuntu -u root
redis-cli --scan --pattern 'rag:*'                  # 列出全部项目键
redis-cli TTL rag:session:{sid}:messages            # 剩余存活时间
redis-cli GET  rag:session:{sid}:messages           # 查看会话消息原文
redis-cli DEL  <key>                                # 删除单个键
redis-cli FLUSHDB                                   # 清空整个库（会清掉全部记忆，慎用）
```


<img width="1738" height="12628" alt="QQ_1787191717918" src="https://github.com/user-attachments/assets/1c8c9b13-bd1f-4b7a-b8cb-795b649d99b5" />


