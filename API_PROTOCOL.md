# API Protocol

入口：
- `POST /api/process-image` — 生成图片并上传 OSS，返回 OSS URL
- `POST /api/generate-image` — 生成图片，直接返回图片二进制文件
- `POST /api/understand-image` — 图片理解，接收图片返回文本描述
- `GET /health/providers` — 查询服务商熔断状态，需要访问令牌

当前约束：
- backend 支持两条链路：Gemini 和 GPT Image
- 当 `model` 以 `gpt-image` 开头时走 GPT Image 链路，否则走 Gemini 链路
- 默认模型、正式模型、preview 兼容别名和服务商模型映射由 `config/providers.json` 定义
- EasyRouter 和 OpenRouter 地址由 `config/providers.json` 定义
- 服务商 API Key 分别从服务端环境变量 `EASYROUTER_API_KEY` 和 `OPENROUTER_API_KEY` 读取，前端无需传入
- 请求来源校验依赖两个请求头：`X-Base-Signature` 和 `X-Pack-Id`
- 服务端会先对 `X-Base-Signature` 验签，再校验签名中的 `packID` 与 `X-Pack-Id` 一致
- GPT Image 链路不支持参考图（fileUrl/fileUrls/files），传入会返回 400
- `/api/process-image` 支持通过 `imageCount` 并发生成多张图片
- `/api/generate-image` 返回单个图片文件，仅支持 `imageCount=1`

## 支持模型

| 公共模型 ID | 兼容别名 | EasyRouter 模型 ID | OpenRouter 模型 ID | 能力 |
|---|---|---|---|---|
| `gemini-3.1-flash-image` | `gemini-3.1-flash-image-preview` | `gemini-3.1-flash-image` | `google/gemini-3.1-flash-image` | 图片生成、图片理解、参考图 |
| `gemini-3-pro-image` | `gemini-3-pro-image-preview` | `gemini-3-pro-image` | `google/gemini-3-pro-image` | 图片生成、图片理解、参考图 |
| `gemini-2.5-flash-image`（Nano Banana） | `gemini-2.5-flash-image-preview` | `gemini-2.5-flash-image` | `google/gemini-2.5-flash-image` | 图片生成、图片理解、参考图 |
| `gpt-image-2` | - | `gpt-image-2` | `openai/gpt-image-2` | 图片生成 |

默认模型为 `gemini-3.1-flash-image`。preview 兼容别名会解析为对应的公共模型 ID，再映射到实际服务商模型 ID。

## 前端可传参数约定

### 通用参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | string | 否 | `""` | 文本提示词 |
| `model` | string | 否 | `providers.json` 默认模型 | 以 `gpt-image` 开头走 GPT Image，否则走 Gemini |
| `aspectRatio` | string | 否 | 不填 | 输出图片比例 |
| `requestId` | string | 否 | `""` | 请求追踪 ID |
| `imageCount` | integer | 否 | `1` | `/api/process-image` 的图片数量，默认范围 `1～5` |

### Gemini 专属参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `imageSize` | string | 否 | `1K` | 输出分辨率档位（`1K` / `2K` / `4K`） |
| `fileUrl` | string | 否 | - | 单个参考图 URL |
| `fileUrls` | string[] | 否 | - | 多个参考图 URL |
| `file` / `files` | file | 否 | - | 单个或多个参考图文件 |

### GPT Image 2 参数说明

| 字段 | 请求体参数 | 值 | 说明 |
|---|---|---|---|
| `aspectRatio` | `size` | 由 aspectRatio 映射为像素尺寸，默认 `auto` | 不传时模型自动选择 |
| - | `quality` | `auto`（硬编码） | 支持 `low` / `medium` / `high` / `auto` |
| - | `moderation` | `low`（硬编码） | 内容审核级别，`auto` 或 `low` |
| - | `n` | `1`（硬编码） | 生成图片数量 |

GPT Image 2 不支持参考图（fileUrl/fileUrls/files），传入会返回 400。

### `aspectRatio` 可选值

- `1:1`
- `2:3`
- `3:2`
- `3:4`
- `4:3`
- `4:5`
- `5:4`
- `9:16`
- `16:9`
- `21:9`

GPT Image 2 的 aspectRatio 到 size 映射：

| aspectRatio | size |
|---|---|
| `1:1` | `1024x1024` |
| `3:2` / `4:3` / `5:4` | `1536x1024` |
| `16:9` / `21:9` | `2048x1152` |
| `2:3` / `3:4` / `4:5` | `1024x1536` |
| `9:16` | `1152x2048` |
| 不传 | `auto` |

### `imageSize` 可选值（仅 Gemini）

- `1K`
- `2K`
- `4K`

### 多参考图约定（仅 Gemini）

- 支持多张参考图
- URL 和上传文件可以混用
- 总参考图数量上限为 `14`
- 服务端会把参考图暂存为私有 OSS 对象；EasyRouter 接收短期签名 URL，OpenRouter 接收后端读取临时对象后生成的 Base64 Data URL
- 同一批并发生成和服务商回退复用相同的临时参考图对象
- EasyRouter 临时无法连接参考图地址时自动进入 OpenRouter 兜底
- 全部模型调用结束后服务端主动删除临时对象

## 请求格式

### 方式 1：JSON + 文件 URL

请求头：
- `Content-Type: application/json`
- `X-Base-Signature: <context.baseSignature>`
- `X-Pack-Id: <context.packID>`

请求体：

Gemini 示例：
```json
{
  "requestId": "req-001",
  "prompt": "生成一张极简风格的海报",
  "model": "gemini-3.1-flash-image",
  "aspectRatio": "16:9",
  "imageSize": "2K",
  "imageCount": 2,
  "fileUrl": "https://example.com/reference-1.png",
  "fileUrls": [
    "https://example.com/reference-2.png"
  ]
}
```

GPT Image 2 示例：
```json
{
  "requestId": "req-001",
  "prompt": "生成一张极简风格的海报",
  "model": "gpt-image-2",
  "aspectRatio": "16:9"
}
```

说明：
- `fileUrl` 和 `fileUrls` 会合并成同一个 URL 列表（仅 Gemini）
- `model` 可不传，不传时使用 `providers.json` 中的默认模型
- `model` 以 `gpt-image` 开头时走 GPT Image 链路
- `aspectRatio` 不传或传空时，Gemini 保持模型默认行为，GPT Image 使用 `auto`
- `imageSize` 不传时默认 `1K`（仅 Gemini 使用）
- `imageCount` 不传时默认 `1`；多图请求仅支持 `/api/process-image`
- `imageCount > 1` 时，服务端为每张图片并发发起一次单图生成请求

### 方式 2：multipart/form-data + 文件流

请求头：
- `X-Base-Signature: <context.baseSignature>`
- `X-Pack-Id: <context.packID>`

表单字段：
- `requestId`
- `prompt`
- `model`
- `aspectRatio`
- `imageSize`
- `imageCount`
- `file`
- `files`

说明：
- `file` 适合单文件
- `files` 适合多文件
- `model` 可不传，不传时使用 `providers.json` 中的默认模型
- `model` 以 `gpt-image` 开头时走 GPT Image 链路
- `aspectRatio` 不传或传空时，Gemini 保持模型默认行为，GPT Image 使用 `auto`
- `imageSize` 不传时默认 `1K`（仅 Gemini 使用）
- `imageCount` 不传时默认 `1`；多图请求仅支持 `/api/process-image`

## 返回格式

两个接口共用基础图片参数。`imageCount > 1` 仅适用于
`/api/process-image`，`/api/generate-image` 始终返回单个图片文件。

### `POST /api/process-image`

服务端根据 `imageCount` 并发执行单图生成，按请求序号上传到 OSS 并返回
JSON。`ossUrls` 包含全部图片地址，`ossUrl` 指向第一张图片：

```json
{
  "success": true,
  "message": "Image generated and uploaded successfully.",
  "timestamp": "2026-04-23T00:00:00+00:00",
  "data": {
    "requestId": "req-001",
    "model": "gemini-model-id-from-env",
    "requestedCount": 2,
    "generatedCount": 2,
    "ossUrl": "https://your-bucket.oss-cn-hangzhou.aliyuncs.com/path/to/file.png",
    "ossUrls": [
      "https://your-bucket.oss-cn-hangzhou.aliyuncs.com/path/to/file-1.png",
      "https://your-bucket.oss-cn-hangzhou.aliyuncs.com/path/to/file-2.png"
    ],
    "provider": "easyrouter",
    "fallbackUsed": false
  }
}
```

多图生成规则：

- `imageCount` 必须为正整数，默认最大值为 `5`
- 最大数量由 `IMAGE_GENERATION_MAX_COUNT` 配置
- 单个批次的并发数由 `IMAGE_GENERATION_MAX_CONCURRENCY` 配置
- 超过生成并发上限的任务进入内存队列，默认最多等待 `420` 秒
- 排队时间不占用服务商调用时限
- 每个服务商拥有独立的 `PROVIDER_REQUEST_TIMEOUT_SECONDS` 调用窗口，默认 `300` 秒；切换兜底服务商时重新计时
- 每个生成任务独立执行主服务商重试与兜底
- 多图子任务会补充当前序号，用于识别提示词中的分图要求并禁止拼图
- 任意生成任务失败时，整个请求返回失败
- 返回的 `ossUrls` 顺序与生成任务序号一致

### `POST /api/generate-image`

生成图片后直接返回图片二进制文件（不经过 OSS），需要在
`Authorization` 请求头中携带访问令牌。

- 响应 `Content-Type`：图片的 MIME 类型（如 `image/png`）
- 响应 `Content-Disposition`：附带文件名
- 响应 `X-Model-Provider`：实际完成请求的服务商
- 响应 `X-Fallback-Used`：是否使用兜底服务商
- 响应体：图片二进制数据

成功时直接返回图片文件流；失败时返回 JSON 错误信息：

```json
{
  "success": false,
  "message": "错误描述",
  "timestamp": "2026-04-23T00:00:00+00:00",
  "data": {}
}
```

### `POST /api/understand-image`

图片理解：接收图片 URL，由后端下载并转换为 Base64 后调用 Gemini 返回文本描述。EasyRouter 和 OpenRouter 均不会收到原始图片 URL。

请求体（仅 JSON）：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `requestId` | string | 否 | `""` | 请求追踪 ID |
| `prompt` | string | 否 | `""` | 理解提示词，如"描述图片内容" |
| `model` | string | 否 | `gemini-3.1-flash-image` | 模型名称 |
| `fileUrl` | string | 否 | - | 单个图片 URL |
| `fileUrls` | string[] | 否 | - | 多个图片 URL |

请求示例：
```json
{
  "requestId": "req-001",
  "prompt": "描述这张图片的内容",
  "model": "gemini-3.1-flash-image",
  "fileUrls": [
    "https://example.com/photo.png"
  ]
}
```

返回示例：
```json
{
  "success": true,
  "message": "Image understanding completed successfully.",
  "timestamp": "2026-05-30T00:00:00+00:00",
  "data": {
    "requestId": "req-001",
    "model": "gemini-3.1-flash-image",
    "text": "这是一张风景照片，画面中...",
    "provider": "easyrouter",
    "fallbackUsed": false
  }
}
```

说明：
- 仅支持 JSON 请求（不支持 multipart/form-data）
- 不需要 `aspectRatio` / `imageSize` 参数
- 参考图数量上限为 14
- 默认走 Gemini 链路，不支持 GPT Image
- EasyRouter 出现可恢复故障且 `FALLBACK_ENABLED=true` 时使用 OpenRouter
