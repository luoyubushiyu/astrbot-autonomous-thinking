# QQ Bot 自主思考系统 · 第六阶段（Contact Intent / Behavior Guard / 长期运行）

在 第一阶段 的可运行骨架上，把「自主思考」接成了完整闭环：消息进来会沉淀记忆、
产生念头；没有消息时 Agent 自己挑一条念头去「想」，想完由程序决定它该进抽屉、
写日记、继续追问，还是转成一次受严格限制的主动联系。

## 核心原则

- **LLM 只负责推理与建议。** 它输出结构化 JSON：回复、等待、静默，或对某条 Thought
  的结论（`THINK`）。它拿不到 SQLite、文件系统、QQ API、调度器。
- **程序负责一切落地。** 流水线固定为：
  事件 → Agent → Context → LLM → JSON → Schema Validator →（Thought 校验）→
  Behavior Guard → Executor。任何一环失败都回退到 `DO_NOTHING`。
- **主动联系有独立意图与护栏。** LLM 只能提「联系意图」（`CONTACT_INTENT`），
  频率、夜间、冷却、重复内容、同一话题反复、连续未回复、异常拦截都会挡；
  长期没回复最终进入 `SILENT`——只停联系，不停思考。
- **想法有生命周期。** 新建、排队、思考、得到结论、等待、抽屉、遗忘、归档；
  抽屉里的念头可以被新信息重新唤醒。
- **黑盒不可见。** 被判定为无价值/危险的想法进入 `blackbox/`，不进上下文、不展示。

## 目录结构

```
autonomous_agent/
├── main.py                 # CLI 入口（不堆业务逻辑）
├── config.py               # 配置加载与校验
├── core/
│   ├── agent.py            # 事件循环、状态机、启动/恢复
│   ├── context.py          # 一阶段聊天上下文
│   ├── emotion.py          # 情绪状态（只影响倾向）
│   ├── event.py            # 事件模型
│   ├── executor.py         # 真正执行动作
│   ├── guard.py            # 行为护栏
│   ├── state.py            # Agent 状态机
│   ├── timewindow.py       # 夜间/静默时间窗口
│   └── thought_engine.py   # 自主思考引擎
├── thought/                # Thought 模型/生命周期/评分/遗忘/唤醒/文件仓库
├── memory/                 # 记忆与话题
├── diary/                  # 日记
├── contact/                # 主动联系意图与限制
├── llm/                    # 适配器、JSON 契约、校验、提示词
├── database/               # SQLite 连接、迁移、仓储
├── qq/                     # QQ 适配器（含 Mock）
├── monitoring/             # 日志
└── tests/                  # 275 个测试
```

## 三步跑起来

1. 下载：仓库页面点绿色 **Code** → **Download ZIP**
   （或者直接点这个：[下载 main 分支 ZIP](https://github.com/luoyubushiyu/astrbot-autonomous-thinking/archive/refs/heads/main.zip)）
2. 解压，Windows 双击 `启动.bat`（macOS / Linux 在终端里 `bash start.sh`）
3. 日志开始滚就是活了：`logs/agent.log` 有输出，`data/agent.db` 自己建好

不用装依赖（只用 Python 标准库），不用 API key —— 默认 `mock` 模式先把整条环路跑通，
离线、免费。唯一要求：Python 3.11 或更新（安装时勾上 *Add python.exe to PATH*）。

想换真模型：改 `config.json` 的 `llm` 段，或者设环境变量
`AGENT_LLM_PROVIDER=deepseek`、`AGENT_LLM_API_KEY=sk-...` 再启动。

只想自检、不常驻：`py -3 -m autonomous_agent --once`。

不用脚本也能跑：

```bash
cp config.example.json config.json
python -m autonomous_agent
```

也可以指定配置文件：`python -m autonomous_agent.main --config path/to/config.json`。
环境变量会覆盖配置，常用项见下面的配置说明。

> 注意：`qq.adapter` 目前只有 `mock`，所以它不会真的收发 QQ 消息，跑的是完整闭环的本地版本。

## 配置

配置分十二段，全部有默认值，环境变量可覆盖常用项：

| 段 | 作用 |
| --- | --- |
| `database` | SQLite 路径、busy timeout、是否记录事件 |
| `llm` | provider（`mock` / `deepseek` / `openai` / `openai_compatible`）、模型、超时、JSON 模式 |
| `qq` | 适配器（当前仅 `mock`）与轮询间隔 |
| `scheduler` | 自主 tick（空闲/活跃两档）、最大等待、SAFE_IDLE 告警 |
| `behavior` | 动作白名单、消息长度、静默时段、限流 |
| `thought` | 工作区目录、Thought 上限、评分权重、连续思考间隔、遗忘/唤醒参数 |
| `memory` | 检索条数、去重阈值、记忆上限、价值权重、遗忘/整理/重新激活参数 |
| `topic` | 话题去重阈值、创建门槛、活跃度权重、降档天数、上下文条数 |
| `curiosity` | 每 tick 候选上限、评分权重、最低分、扩展上限 |
| `contact` | 主动联系的频率/夜间/冷却/重复/同话题/异常限制、连续未回复转 SILENT、等待回复超时 |
| `emotion` | 情绪基线、衰减速度、单次事件最大变化 |
| `diary` | 单条日记长度 |
| `logging` | 级别、文件、JSON 行、控制台 |

环境变量示例：`AGENT_CONFIG`、`AGENT_DB_PATH`、`AGENT_LLM_API_KEY`、
`AGENT_LLM_PROVIDER`、`AGENT_LLM_BASE_URL`、`AGENT_LLM_MODEL`、
`AGENT_LLM_JSON_MODE`、`AGENT_THOUGHT_ENABLED`、`AGENT_WORKSPACE_DIR`、
`AGENT_ACTIVE_TICK_SECONDS`、`AGENT_CONTINUE_THINK_MINUTES`、
`AGENT_CONTACT_ENABLED`、`AGENT_CONTACT_HOURLY_LIMIT`、
`AGENT_CONTACT_TIMEOUT_SECONDS`、`AGENT_CONTACT_TOPIC_REPEAT_LIMIT`、
`AGENT_CONTACT_SILENT_AFTER`、`AGENT_CONTACT_SILENT_MINUTES`、
`AGENT_ALLOW_AUTONOMOUS_CONTACT`、`AGENT_MEMORY_ENABLED`、
`AGENT_TOPIC_ENABLED`、`AGENT_CURIOSITY_ENABLED`、`AGENT_CURIOSITY_MIN_SCORE`、
`AGENT_EMOTION_ENABLED`、`AGENT_EMOTION_MAX_DELTA`、`AGENT_DIARY_ENABLED`、`AGENT_LOG_LEVEL`。

相对路径按配置文件所在目录解析。`thought` / `memory` / `topic` / `curiosity` /
`emotion` / `diary` 的 `enabled=false` 会真的跳过对应子系统（想法与主动联系不再产生、
记忆不写、话题不更新、好奇不再扫、情绪不变、日记不落盘），聊天回复链路不受影响。

## 自主思考怎么跑

`AUTONOMOUS_TICK` 到来时：

1. 程序先做维护（按配置每 N 个 tick 扫一次遗忘、按闲置天数给话题降档、整理记忆）；
2. 从数据库挑候选 Thought（`NEW` / `QUEUED` / `FOLLOW_UP`，或到期的 `WAITING`），
   用评分公式排序；没有候选时让 Curiosity 扫一遍（互相矛盾的记忆、低置信 `INFERENCE`、
   久未推进的重要话题、只有一个话题时的温和探索），把值得想的问题落成 `CURIOSITY` 念头，
   和存量想法进同一个候选队列；**还是没有就直接 `DO_NOTHING`，不调用 LLM**；
3. 被选中的念头进入 `THINKING`，上下文里只放它本身、相关记忆、话题、最近对话和情绪；
4. LLM 返回 `THINK`：`insight`、`uncertainty`、`disposition`、新问题、下次检查时间；
5. 校验通过后由程序落地：

| disposition | 结果 |
| --- | --- |
| `KEEP` | 结论存为记忆，念头 → `INSIGHT_FOUND` |
| `DRAWER` | 念头进抽屉，等待被唤醒 |
| `DIARY` | 写当天日记，念头 → `RESOLVED` |
| `SPEAK` | 生成联系意图 `CONTACT_INTENT`，再过护栏与联系限制；允许才发，否则记一条 DENIED |
| `BLACKBOX` / `DISCARD` | 进黑盒 |
| `WAIT` | 念头 → `WAITING`，设 `next_check_at` |
| `FOLLOW_UP` | 念头 → `FOLLOW_UP`，下次继续 |

LLM 出错、超时、JSON 坏、Schema 坏、`thought_id` 不存在，都会把念头放回 `QUEUED`，
记下 `last_error`，5 分钟后再试。

`USER_MESSAGE` 到来时：写消息、更新记忆和话题、检查是否值得产生新念头；同时走原来的
聊天回复链路。被主动联系后用户回复了，会重置「连续未回复」计数。

## LLM 适配层

Agent 只依赖 `LLMAdapter.complete(request) -> LLMResponse`，不关心厂商；两个实现：

- `MockLLMAdapter`：测试与离线运行，按队列返回文本、异常、延迟。
- `OpenAICompatibleAdapter`：标准库 urllib 调 `POST {base_url}/chat/completions`。

`llm.provider` 可选 `mock`、`deepseek`、`openai`、`openai_compatible`。写
`deepseek` / `openai` 时 `base_url` 与默认模型会自动填好（仍可显式覆盖）；接别的
兼容服务用 `openai_compatible` 并写 `base_url`。`llm.json_mode=true`（默认）时会带
`response_format={"type":"json_object"}`，兼容层不支持就关掉。

模型必须返回一段 JSON。任何一步出问题都收敛成 `DO_NOTHING`，不会把半成品动作执行掉：

| 情况 | 处理 |
| --- | --- |
| 超时 / 网络错误 | 记 `LLM_TIMEOUT` / `LLM_ERROR`，状态回 `SAFE_IDLE` |
| 空内容 | `EMPTY_RESPONSE` → `DO_NOTHING` |
| 不是 JSON | `JSON_PARSE_ERROR` → `DO_NOTHING` |
| 结构不对 | `SCHEMA_ERROR` → `DO_NOTHING` |
| 动作不在白名单 | `UNKNOWN_ACTION` → `DO_NOTHING` |
| 参数类型/范围错误 | `INVALID_PARAMETER` → `DO_NOTHING` |
| 会话或 `thought_id` 不存在 | `UNKNOWN_ID` / `INVALID_THOUGHT_ID` → `DO_NOTHING` |

## Thought Workspace

```
autonomous_thinking/
├── thinking/active/     NEW QUEUED THINKING INSIGHT_FOUND FOLLOW_UP RESOLVED
├── thinking/pending/    WAITING
├── drawer/              DRAWER
├── diary/               YYYY-MM-DD.md
├── spoken/              被说出去的念头副本
├── blackbox/discarded/  DISCARDED
├── blackbox/expired/    EXPIRED
└── archive/             ARCHIVED
```

数据库是事实来源，文件是镜像。所有读写都经过 `ThoughtRepository`，业务代码不直接碰文件：

```python
create / read / update / move / archive / delete / restore / reactivate
parent_of / children_of / related_of / supports_of / contradicts_of / tree
sync / repair
```

- **写入**：先落库再写文件；文件写失败只告警，启动时 `repair()` 按数据库补回。
- **一致性**：`repair()` 会补齐缺失文件；对「文件在、数据库里没有」的 Thought 尝试
  导入，解析不了的文件隔离到 `blackbox/discarded/*.orphan.json`，不直接删。
- **删除**：只有软删除（`DISCARDED`），数据和文件都保留。
- **Thought Tree**：`parent_thought_id` 加 `child_thought_ids` 双向记录，关系表另有
  `RELATED` / `SUPPORTS` / `CONTRADICTS` / `CHILD` / `MERGED_INPUT`。
- **评分**：重要度、好奇度、相关度、新鲜度、未解决价值、关系价值、未来价值、
  时间合适度，减去重复、陈旧和成本惩罚，权重在 `thought` 段配置。
- **遗忘**：软删除到 `EXPIRED`，不物理删除；进入黑盒后不再参与上下文。
- **唤醒**：抽屉/已遗忘/已归档的念头，如果和新信息足够相似，会被拉回 `QUEUED`；
  有次数上限和冷却。

### 思考文本（连续思考）

每一条被选中的 Thought 都有一个同名的 `.txt`，和 `.json` 放在同一个状态目录下。
下一次思考先读回这份文本，接着上次往下想，而不是每次从零开始：

```
核心问题 / 已知信息 / 当前思考 / 关键发现 / 不确定 /
新问题 / 相关问题 / 下一步 / 继续思考条件 / 停止原因 / 历史摘要
```

- 一次 Tick 只推进一步；`CONTINUE` 表示没想完，程序把 `next_action` 记成
  `THINK_AGAIN`，状态回到 `QUEUED`，隔 `continue_think_minutes`（默认 3 分钟）
  再接着想同一份文本，不会立刻插队重来。
- 调度节奏跟着念头池走：池子里有能立刻想的念头时，tick 缩到
  `active_tick_seconds`（默认 60 秒）；池子空了就回到 `autonomous_tick_seconds`
  （默认 300 秒）。有念头在等下次思考时，不会额外造好奇问题。
- 文本只保留结论、疑点、新问题和一段简短历史摘要，**不保存完整思维链**；
  超过 `thinking_text_max_chars` 时压缩历史部分，当前状态不会被截断。
- 写回顺序是「先落库、再写文件」；文件写失败只告警，下次从数据库和已有文本恢复。
- 子问题受 `max_followups_per_thought`、`max_thought_depth` 约束；和父问题/祖先
  太像的问题会被判为循环/重复，直接跳过。

## 动作与校验

允许的动作：`DO_NOTHING`、`SEND_MESSAGE`、`WAIT`、`ENTER_SILENT`、`THINK`。

- `SEND_MESSAGE` 必须带 `conversation_id` 和 `content`，目标必须是已知会话。
- 自主 tick 默认不允许发消息；只有 `SPEAK` 通过联系限制、以「已批准的联系意图」
  进入 Guard 时才放行。
- 所有 JSON 都先过 Schema 校验；校验失败按 `EMPTY_RESPONSE`、`JSON_PARSE_ERROR`、
  `SCHEMA_ERROR`、`UNKNOWN_ACTION`、`INVALID_PARAMETER`、`UNKNOWN_ID` 分类记录。

## 主动联系限制

`contact` 段里：每小时上限、每天上限、最小间隔、连续未回复上限、夜间时段、
冷却、重复内容检测（相似度 + 窗口）、同一条念头最多联系次数、等待回复超时
（`timeout_seconds`，默认 1800 秒）。任何一条不满足就记 `DENIED` 并说明原因；
连续多次没回复后频率下降，最终进入 `SILENT`。

主动联系发出去之后，每个自主 tick 会检查有没有等太久的：等够 `timeout_seconds`
就产生一个 `CONTACT_TIMEOUT` 事件（同一条联系只报一次，用户回复后不再报），
如果 Agent 还在 `WAITING`，就把状态拉回 `IDLE`。

### 联系意图（CONTACT_INTENT）

Thought 的 `SPEAK` 不再直接发消息，而是先落成一个意图对象：

```
ContactIntent: intent_id / source_thought_id / conversation_id /
               topic / reason / message_draft / priority / urgency / created_at
```

- 意图只是「想说什么」，**没有任何执行权**；`contact_attempts` 里同时记下
  `intent_id`、`topic`、`message_hash`、`blocked_reason`，发出去以后再补
  `reply_received` 与 `response_time_seconds`。
- 拦截顺序：`CONTACT_DISABLED` → `SILENT_MODE` → `NIGHT_TIME` →
  `ANOMALY_PROTECTION` → `CONSECUTIVE_NO_REPLY` → 每小时/每天上限 → 最小间隔 →
  冷却 → 重复内容 → 同一话题反复 → 同一条念头次数。任何一条不过都只记 DENIED，
  念头不会被丢掉（退回 `WAITING`）。
- 连续 `silent_after_consecutive` 条主动联系没回复，Agent 自动进入 `SILENT`
  （写 `silent_until`、挂闹钟）；`SILENT` 期间**照常选念头、照常调 LLM**，
  只是所有联系意图都被拦成 `DENIED(SILENT_MODE)`。用户一回复就退出静默。
- 短时间内被挡太多次（`anomaly_block_threshold` / `anomaly_window_minutes`）
  触发 `ANOMALY_PROTECTION`，先停一轮，避免异常刷屏。

### 长期运行

- 每 `20` 个自主 tick 跑一次维护：按 `database.event_retention_days` /
  `metrics_retention_days` 清旧行、`PRAGMA optimize`、`ThoughtRepository.repair()`；
  维护失败只记日志，不影响主循环。
- 日志文件按 `logging.max_bytes` / `backup_count` 轮转。
- 重启时恢复 Agent 状态、`silent_until` 闹钟和联系计数；恢复后继续按硬限制判断。
- 状态机异常走 `SAFE_IDLE`，不会因为一次坏事件卡死。

## 记忆、话题、好奇、情绪、日记

- 记忆类型：`FACT`、`PREFERENCE`、`EVENT`、`RELATIONSHIP`、`TOPIC`、`INFERENCE`、
  `THOUGHT`、`EPISODIC_MEMORY`、`SUMMARY_MEMORY`；推测不会自动升级成事实——
  从思考里来的 `FACT` 一律降级成 `INFERENCE`，存库时也会拒绝把 `INFERENCE` 改写成 `FACT`。
- 来源分开记：`source_type`（`USER_MESSAGE`/`THOUGHT`/`DIARY`/…）加 `source_id`，
  Thought 与 Memory 之间还写一条 `MEMORY` 关系，双向都能查。
- LLM 不能直接写记忆：它只能在 `THINK` 里提 `memory_candidates`，程序按
  「太短/太弱就丢 → 冲突记 `CONTRADICTS` → 高度重复合并进旧条目 → 否则新建」走一遍，
  同一件事反复想不会越存越多。
- 价值评分是程序算的：重要度、置信度、相关度、未来价值、使用频率、最近使用，减掉
  陈旧、重复、低置信惩罚；低于阈值就按
  `ACTIVE → LOW_VALUE → SOFT_DELETED → EXPIRED → BLACKBOX → ARCHIVED` 逐级降档，
  不物理删除，`restore()` 能拉回 `ACTIVE`（记一次重新激活）。
- 长期记忆会定期 `sweep()` 降档、`consolidate()` 合并高度相似的重复条目并把旧的归档；
  宁可保留，也不在拿不准时改事实。
- 新消息命中旧记忆（相似度 + 价值 + 冷却 + 次数都过关）会把它恢复成 `ACTIVE`，
  并顺手唤醒引用它的抽屉/过期 Thought。
- 话题（Topic）表示「长期关注什么」，不是想法本身：生命周期
  `NEW → ACTIVE → INTERESTING → LOW_ACTIVITY → DORMANT → ARCHIVED`，活跃度由
  最近活动、Thought 数、Memory 数、重要度算出来；长期没动静的话题降档后不再占上下文，
  但可以因为一条新消息或一个念头被重新激活。新话题要先过名字相似度和
  `min_importance_to_create` 门槛，不会因为出现一个关键词就建。
- 好奇（Curiosity）负责「为什么想知道」：程序先扫矛盾记忆、低置信推测、久未推进的
  话题，算出 novelty/uncertainty/importance/knowledge_gap/… 的综合分，过线的才落成
  `CURIOSITY` 念头；同一类问题反复生成会被相似度和
  「还没有想过的同类念头」两条守卫挡住，不会无限循环。它不直接发消息。
- 情绪（孤独、好奇、兴趣、能量、心情、依恋）按时间衰减，单次事件的变化被
  `max_emotion_delta_per_event` 夹住；它只影响候选评分和联系倾向，不直接决定行为，
  也改不了护栏。
- 日记按天追加写入 `diary/YYYY-MM-DD.md`，同时落库；只有明确的 `DIARY` disposition
  才写。写进去的推测会带「尚未确认」之类的字样，不因为写进日记就变成事实。

## 状态与事件

Agent 状态：`IDLE`、`OBSERVING`、`THINKING`、`WAITING`、`CONTACTING`、`SILENT`、
`NIGHT`、`SLEEPING`、`SAFE_IDLE`。Thought 状态独立维护——每条想法自己的
`NEW`/`QUEUED`/`THINKING`/… 存在 `thoughts` 表里，跟 Agent 状态机没有任何耦合。

主要流转：

- 收到用户消息：`IDLE`（或 `NIGHT`/`SLEEPING`）→ `OBSERVING` → `THINKING` → 回复/`IDLE`；
- 自主 tick：`IDLE` → 选念头 → 想完按 disposition 落到 `IDLE`/`WAITING`/`SILENT`；
- 主动联系：`IDLE` → `CONTACTING` → 发送/拒绝 → `IDLE`；
- 夜间（`contact.night_hours`）：`IDLE` → `NIGHT` → 下一个 tick 进 `SLEEPING`，
  天亮后第一个 tick 自动醒回 `IDLE`；夜间 tick 不调 LLM；
- 用户消息在夜间仍会把人叫醒并正常回复。

`stats()` 里的 `recent_states` 会给出最近 50 次转移中的最后 10 条
（`from`/`to`/`reason`/`at`），排查状态问题用得上。

事件：`USER_MESSAGE`、`AUTONOMOUS_TICK`、`BOT_START`、`BOT_STOP`、
`ALARM_TRIGGERED`、`CONTACT_TIMEOUT`。事件只负责分发，不做推理；每种事件的
payload 都会在构造时校验，坏数据直接拒绝。

重启时瞬态状态（`OBSERVING`/`THINKING`/`CONTACTING`）回到 `SAFE_IDLE`；`SILENT`
会按剩余时间续上定时器；`WAITING` 回到 `IDLE`（等待中的念头本身有 `next_check_at`，
不依赖 Agent 状态）；`NIGHT`/`SLEEPING` 回到 `IDLE`，如果仍是夜间，下一次 tick
会重新进入 `NIGHT`。

## 安全边界

LLM 永远拿不到：SQLite 连接、任意文件写、任意 QQ API、任意调度、直接发消息、
删除数据。它只能输出动作 JSON；校验器管结构，护栏管权限，执行器管落地。

## 数据库

`agent_state`、`messages`、`events`、`metrics`、`thoughts`、`thought_relations`、
`memories`、`topics`、`contact_attempts`、`diary`。迁移用 `PRAGMA user_version`
记录版本，每条迁移在自己的 `BEGIN IMMEDIATE` 事务里执行：失败就整条回滚，版本号也不会涨；
数据库版本比程序新时拒绝启动。

连接是单连接 + 串行锁，文件库开 WAL。事务里如果从别的任务偷偷访问同一个连接，
直接抛错误而不是和当前事务交错；嵌套 `transaction()` 复用外层，外层回滚时一起回滚。
连接可重复关闭，关闭后再查询会得到明确的“尚未连接”错误。

## 测试

```bash
python -m unittest discover -s autonomous_agent/tests -t .
```

275 个测试，覆盖：启动/重启/状态恢复、夜间/睡眠/联系中状态流转、聊天与自主 tick、
LLM 适配器（真实 HTTP、json_mode、预设、超时/坏响应）、LLM 超时/坏 JSON/坏 Schema、
护栏、SQLite 连接管理/迁移原子性/事务回滚/跨任务拒绝/重启恢复、Thought 生命周期/评分/遗忘/唤醒/文件镜像/Repository 接口/Thought Tree/一致性修复、记忆去重/冲突/价值衰减/软删除与恢复/整理/重新激活与冷却/Thought-Memory 双向关系、话题生命周期/去重/降档/重新激活、好奇检测（矛盾、低置信推测、闲置话题、重复与循环守卫）/落库上限、
日记不确定性、情绪与限幅、主动联系各条限制、六类事件与 payload 校验、联系超时事件、并发事件串行、
过期闹钟、未来 schema 版本、thought/memory/topic/curiosity/emotion/diary 开关、示例配置可加载、
思考文本（渲染/解析/压缩、多轮连续、文本丢失重建、坏 JSON 保留旧文本、深度与重复守卫）、
QQ 适配器接口/核心解耦/Mock 收发/自定义适配器可插拔、
第六阶段（ContactIntent 序列化、contact_attempts 新列、意图字段落库、回复时长、
连续未回复、同话题反复、异常保护、SILENT 中仍思考但拦截联系、维护清理旧行）。
`tests/test_integration.py` 还串起了 1～6 阶段的完整链路：用户消息 → 回复 → 自主 tick 发起联系 → 用户回复写回反馈 → 重启恢复，以及连续未回复自动进 SILENT 后仍继续思考。

## QQ 适配器

核心只依赖 `autonomous_agent/qq/adapter.py` 里的抽象接口，不 import 任何具体 QQ 框架：

```python
async def start() -> None
async def stop() -> None
async def receive_message() -> IncomingMessage | None   # 轮询式取一条，没有就 None
async def send_message(message: OutgoingMessage) -> SendResult
async def get_user_info(user_id: str) -> UserInfo | None
async def get_group_info(group_id: str) -> GroupInfo | None
```

`Agent` 通过构造参数 `Agent(config, qq=...)` 注入，或者用 `config.qq.adapter` 走
`create_qq_adapter()` 工厂。轮询、异常退避、发送失败处理都在 Agent/Executor 里，适配器只管收和发。

`qq.adapter=mock` 时用 `MockQQAdapter`：`push_text(...)` / `push_incoming(...)` 注入消息，
`outbox` 里看机器人发出的消息，`users` / `groups` 放资料，`send_failure` 可以模拟发送通道异常，
方便测试与本地演示。

接真实框架时写一个 `QQAdapter` 子类：把框架回调放进队列给 `receive_message()` 取，
`send_message()` 转成框架的发送 API 并返回 `SendResult`，再 `Agent(config, qq=MyAdapter(...))` 即可，核心不用改。

## 已完成 / 未完成

已完成：Thought Workspace、Thought Lifecycle、ThoughtRepository（create/read/update/move/archive/delete/restore/reactivate + 一致性修复）、Thought Tree（父/子/相关/支持/矛盾）、Thought Scoring 基础版（含时间合适度与成本项）、自主思考循环（选念头 → THINK → disposition 落地）、
思考文本（每条念头一份可续读的 `.txt`，一次 Tick 推进一步，CONTINUE 下次接着想）、
记忆生命周期（结构化来源、Save-Memory 校验、去重/冲突、价值评分、分级遗忘与恢复、整理、重新激活与冷却、Thought↔Memory 双向关系）、话题生命周期（去重、活跃度、降档、重新激活）、好奇引擎（矛盾记忆/低置信推测/闲置话题 → CURIOSITY 念头，带重复与循环守卫）、情绪（限幅更新、只影响倾向）、日记（按天追加、保留推测的不确定性）、遗忘与唤醒、主动联系硬限制、聊天回复链路、
九种 Agent 状态（含夜间/睡眠/联系中）与重启恢复、六类事件与联系超时、
LLM 适配层（Mock / DeepSeek / OpenAI / 兼容接口，JSON 模式与安全 fallback）、
QQ 适配器（接口解耦、Mock、可插拔）、
主动联系意图（CONTACT_INTENT 与 `contact_attempts` 全字段）、联系护栏与硬限制（含同话题反复、异常保护、连续未回复转 SILENT）、
SILENT 中继续思考但拦截联系、长期运行维护（事件/指标保留、DB optimize、Workspace repair、日志轮转）、275 个测试。

未完成（按现有范围约定）：完整管理员命令、`alarms` 表、LLM 语义校验器、
记忆整合里「用 LLM 改写/总结记忆」的部分（当前的 `consolidate()` 是程序规则）、
日终汇总、真实 QQ 接入。

## 现在能「真正自主思考」了吗

还不能——现在是一个**程序驱动的思考循环**，不是自主意识：

- 能做的：用户消息会变成问题型 Thought；没有存量念头时，Curiosity 会从矛盾记忆、
  低置信推测、久未推进的话题里自己找问题，落成 `CURIOSITY` 念头再继续想；自主 tick 会
  挑一条念头、调 LLM 产出一段结论、疑点、下一步，并按 disposition 落到文件/数据库；
  一条念头能生出子问题，子问题进 Workspace 等着被想；旧念头会被遗忘、被新信息唤醒。
- 还能的：会自己判断「这条念头想完了，要不要说给用户听」——`SPEAK` 先变成
  联系意图，再过护栏和硬限制；长期没人回就自己进 `SILENT`，但思考不停。
  长期运行有定期维护，数据库、文件、日志不会无限长。
- 还不能的：没有长期规划，一条念头想完就结束，不会自己排「下一步该干嘛」的日程；
  情绪只影响候选评分，不构成独立动机；没有语义校验，LLM 的结论只做结构校验；
  话题和好奇的阈值都是启发式的，需要按场景调；主动联系的措辞仍完全来自 LLM。

所以更准确的说法是：**它会按规则给自己找一些问题，把一件事想下去，也会在限制内
决定要不要主动找你；但还不会长期计划、也不会真正理解自己在说什么。**

## 已知限制

- 事件队列不落盘：进程被强杀时，队列里未处理的事件会丢（events 表停在 `PENDING`）。
- 数据库连接只由 Agent 内部持有，不做多进程共享。
- `WAITING` 重启后回 `IDLE`，不重建剩余等待时间。
- Mock QQ 只在进程内，不连真实协议。
- Thought 文件镜像写失败只告警，靠启动时 `repair()` 修复。
- `thought.max_active_thoughts` 只做配置预留，当前限制靠 `max_thoughts` 总数。
- 长期不回复只会挡住主动联系（`consecutive_no_reply_limit`），不会自动把状态降到 `SILENT`。
- 记忆价值/遗忘阈值是启发式公式，参数在 `memory` 配置段里，换场景要自己调。
- 涉及旧记忆的「冲突」只标出来并留新条目，不做自动裁决。
- 记忆重新激活按相似度 + 价值判断，不做语义推理（换说法可能命中不了）。
- Curiosity 的候选来源只有矛盾记忆、低置信 `INFERENCE`、久未推进的话题和一个温和探索角度，
  不是自由的「随便想点什么」；没有线索时它会安静下来。
- 话题降档后可能被 Curiosity 当成候选、`touch` 时又回到 `ACTIVE`——这是设计行为，
  不是漏降级。
- 情绪目前只影响新念头的好奇值和评分，没有反过来影响话题偏好或主动联系倾向。

## 下一阶段可以接

- 管理员命令（`/agent status`、`/agent thoughts`、`/agent diary` 等）。
- 语义校验器（在不破坏 Schema 校验的前提下加一层「这条结论是否自相矛盾」）。
- 记忆整合与话题引擎的深化：从对话里抽取稳定偏好、合并同类话题。
- 日终汇总：当天日记 → 一段回顾 → 可能产生新的 `REFLECTION` 念头。