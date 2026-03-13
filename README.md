# AI-PET 后端说明
## 后端实现
- 基于 `FastAPI`，入口在 `main.py`。
- 路由由两个模块组成：
  - `modules/llm/controller.py`：NPC 对话与提示词更新。
  - `data_preparation/controller.py`：知识数据入库任务提交。
- 数据库通过 `SQLAlchemy` 管理会话（见 `common/dependencies.py`）。
- 检索与知识库相关能力在 `retrieval/*` 与 `data_preparation/*` 中实现（加载、分块、向量写入）。
- 统一业务异常通过 `AppError` 返回结构化 JSON：`{code, message}`。

## 对外接口

### 1) NPC 对话
- **POST** `/llm/npc`
- 请求体：
```json
{
  "npc_id": "string",
  "query": "string",
  "rewrite_query": []
}
```
- 响应体：
```json
{
  "npc_id": "string",
  "response": "string",
  "node1_output": null
}
```

### 2) 更新 NPC 提示词
- **POST** `/llm/prompt`
- 请求体：
```json
{
  "npc_id": "string",
  "prompt": "string"
}
```
- 响应体：
```json
{
  "message": "NPC的提示词已更新"
}
```

### 3) 提交数据准备任务
- **POST** `/dataUpload`（返回 `202 Accepted`）
- 请求体：
```json
{
  "npc_id": "string",
  "file_paths": ["path/a.pdf", "path/b.pdf"]
}
```
- 响应体：
```json
{
  "message": "数据准备任务已提交"
}
```

## 备注
- 若 `npc_id` 不存在，会返回 404（统一错误结构：`code` + `message`）。
- `/dataUpload` 为后台异步任务接口，成功返回仅表示任务已提交。


## 优化方向
- 使用mem0在同一对话中，管理上下文感知能力（当前相当于将npc_prompt手动作为了长期记忆，将chat_history手动作为了短期记忆，上下文会造成严重的污染问题）
- 加入打断会话的功能？
- 将之前完成的飞书MCP更改成OUTLOOK的MCP工具，迁移到该项目当中

