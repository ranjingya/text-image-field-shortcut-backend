# text-image-field-shortcut-backend

一个 Flask backend，用于承接字段捷径后的处理链路：

1. 接收字段捷径请求
2. 根据 model 自动路由至 Gemini 或 GPT Image 生成图片
3. 按指定数量并发生成图片并上传 OSS，返回 OSS URL 列表（`/api/process-image`）
4. 或直接返回图片文件（`/api/generate-image`）
5. 图片理解：接收图片，调用 Gemini 返回文本描述（`/api/understand-image`）

当前已接入：
- HTTP 接口骨架
- JSON / multipart 两种输入解析
- Gemini 生图（支持参考图与并发多图生成）
- 参考图单次下载、临时 OSS URL 复用和格式校验
- 进程级图片生成并发闸门与等待队列
- GPT Image 2 生图（size/quality/moderation）
- Gemini 图片理解（图片→文本）
- 真实 OSS 上传
- 直接返回图片文件（无需 OSS）
- EasyRouter 主服务商与 OpenRouter 顺序兜底
- 可配置的同服务商重试、请求总时限和路由结果标识

## 设计文档

- [同场景多图生成与飞书回填方案](docs/shared-scene-generation.md)

## 配置

非敏感的服务商地址、主备顺序、默认模型、模型别名、能力和服务商模型映射统一存放在 `config/providers.json`。环境变量只保存密钥、运行开关和部署参数。

### 支持模型

| 公共模型 ID | 兼容别名 | EasyRouter 模型 ID | OpenRouter 模型 ID | 能力 |
|---|---|---|---|---|
| `gemini-3.1-flash-image` | `gemini-3.1-flash-image-preview` | `gemini-3.1-flash-image` | `google/gemini-3.1-flash-image` | 图片生成、图片理解、参考图 |
| `gemini-3-pro-image` | `gemini-3-pro-image-preview` | `gemini-3-pro-image` | `google/gemini-3-pro-image` | 图片生成、图片理解、参考图 |
| `gemini-2.5-flash-image`（Nano Banana） | `gemini-2.5-flash-image-preview` | `gemini-2.5-flash-image` | `google/gemini-2.5-flash-image` | 图片生成、图片理解、参考图 |
| `gpt-image-2` | - | `gpt-image-2` | `openai/gpt-image-2` | 图片生成 |

默认模型为 `gemini-3.1-flash-image`。客户端传入 preview 兼容别名时，会解析为对应的公共模型 ID，再映射到实际服务商模型 ID。

推荐生产环境配置：

```dotenv
APP_ENV=production
LOG_LEVEL=INFO
AUTH_SERVICE_URL=http://kocotree-skills-auth:5050

EASYROUTER_API_KEY=
OPENROUTER_API_KEY=
FALLBACK_ENABLED=true
IMAGE_GENERATION_MAX_COUNT=5
IMAGE_GENERATION_MAX_CONCURRENCY=5
IMAGE_GENERATION_QUEUE_TIMEOUT_SECONDS=420
MEMORY_TRIM_AFTER_IMAGE_REQUEST=false
MEMORY_TRIM_RSS_THRESHOLD_MB=512
MEMORY_TRIM_COOLDOWN_SECONDS=60

OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com
OSS_ACCESS_KEY_ID=
OSS_ACCESS_KEY_SECRET=
OSS_BUCKET_NAME=
OSS_BUCKET_FOLDER_PREFIX=images
OSS_TEMP_FOLDER_PREFIX=temp-references
OSS_TEMP_URL_TTL_SECONDS=3600
OSS_TEMP_CUSTOM_DOMAIN=https://temp-img.kktree.cn

FEISHU_ALERT_ENABLED=true
FEISHU_ALERT_WEBHOOK_URL=
FEISHU_ALERT_KEYWORD=
```

使用 STS 临时凭证访问 OSS 时增加 `OSS_SESSION_TOKEN`。`FEISHU_ALERT_ENABLED=true` 时必须同时配置 Webhook 和机器人关键词。服务会把 `FEISHU_ALERT_KEYWORD` 放在每条告警消息开头。

熔断、兜底告警计数和通知冷却使用应用进程内的线程安全内存。Gunicorn 以单 worker、多线程方式运行，不需要额外的状态中间件；应用进程重启后状态会清空。

`IMAGE_GENERATION_MAX_CONCURRENCY` 表示当前进程同时执行的单张图片生成任务上限，所有 HTTP 请求共享同一个并发闸门。超过上限的任务在内存中等待，最长等待时间由 `IMAGE_GENERATION_QUEUE_TIMEOUT_SECONDS` 控制。任务获得生成名额后，才开始计算 `MODEL_REQUEST_DEADLINE_SECONDS` 定义的模型生成、重试和兜底预算。当前部署使用单 Gunicorn worker，因此该限制就是整个服务实例的生成并发上限。

`MEMORY_TRIM_AFTER_IMAGE_REQUEST` 是内存回收开关，默认关闭。启用后，服务会在图片响应发送完成、生成任务与等待队列均为空且进程 RSS 达到 `MEMORY_TRIM_RSS_THRESHOLD_MB` 时，在 Linux 容器内调用 `malloc_trim(0)` 归还 glibc 保留的空闲堆页。进程通过非阻塞互斥锁避免重复回收，并按照 `MEMORY_TRIM_COOLDOWN_SECONDS` 限制回收频率。裁剪期间新的生成任务会等待状态锁释放后再调用服务商。日志 `memory.image_request.release.completed` 包含触发阈值、冷却时间与回收前后 RSS；该操作可能短暂阻塞进程。

生成参考图统一上传为 `OSS_TEMP_FOLDER_PREFIX` 下的私有临时对象，通过 `OSS_TEMP_CUSTOM_DOMAIN` 生成有效期为 `OSS_TEMP_URL_TTL_SECONDS` 的 HTTPS 签名 URL。EasyRouter 直接使用签名 URL；OpenRouter 调用时由后端读取临时对象并转换为 Base64 Data URL，不向 OpenRouter 发送远程图片 URL。自定义域名需要绑定至当前 Bucket、配置 CNAME 和有效 SSL 证书。`OSS_ENDPOINT` 继续用于对象上传和删除。同一批并发生成和服务商回退复用临时对象，全部模型调用结束后服务会主动删除临时对象。OSS Bucket 需要配置一条仅匹配 `temp-references/` 的生命周期规则，在对象最后修改时间超过 1 天后删除，负责清理进程异常退出产生的残留对象。正式结果目录 `images/` 不得与临时目录重叠。

## 日志

生产环境使用 `LOG_LEVEL=INFO` 时，每个成功业务请求记录接收和完成两条汇总日志。完成日志包含模型、服务商、兜底状态、图片数量和总耗时。图片生成接口还会记录 `queued`、`queuedImageCount` 和 `maxQueueWaitMs`；任务无法立即获得并发名额时即视为排队。单张生成、响应解析和 OSS 上传等逐项明细使用 `DEBUG` 级别，仅在排查问题时通过 `LOG_LEVEL=DEBUG` 开启。服务商失败、熔断状态变化和最终请求失败继续使用 `WARNING` 或 `ERROR`。

服务商 HTTP 失败日志包含 `statusCode`、`providerErrorType`、`providerRequestId`、`responseBytes` 和脱敏后的 `message`。错误正文中的 HTTP(S) URL 会替换为 `<url>`，避免记录临时参考图签名参数。

带参考图的生成请求额外记录 `image.reference.oss.upload.completed` 和 `image.reference.oss.cleanup.completed` 两条批次汇总 INFO 日志，分别包含上传数量、总字节数、耗时以及主动删除结果。日志不会包含签名 URL、URL 查询参数或图片内容。

## Docker
构建
```bash
cd <path-to-workspace>/text-image-field-shortcut-backend

docker build -t text-image-field-shortcut .
```

启动
```bash
docker run --name text-image-field-shortcut --env-file .env -p 5000:5000 text-image-field-shortcut
```

compose
```bash
docker compose up --build
```

部署后确认临时参考图配置：

```bash
docker exec text-image-field-shortcut-backend /app/.venv/bin/python -c '
from services.settings import get_app_settings
s = get_app_settings()
print(
    s.oss.temporary_reference_prefix,
    s.oss.temporary_url_ttl_seconds,
    s.oss.temporary_custom_domain,
)
'
```

带参考图完成一次生成后，日志中应依次出现 `image.reference.oss.upload.completed` 和 `image.reference.oss.cleanup.completed`，且后者的 `failedCount` 为 `0`。回滚时把 `docker-compose.yaml` 的镜像标签切换到上一稳定版本，再执行：

```bash
docker compose pull text-image-field-shortcut-backend
docker compose up -d --force-recreate text-image-field-shortcut-backend
```

## 接口

### 健康检查

```powershell
Invoke-WebRequest http://127.0.0.1:5000/health
```

### 图片处理接口（返回 OSS URL 列表）

接口根据 `imageCount` 并发执行单图生成。多图子任务会补充当前序号，用于识别
提示词中的分图要求并禁止拼图。生成结果按任务序号上传，响应中的 `ossUrls`
包含全部图片地址，`ossUrl` 指向第一张图片。

请求中的参考图会在拆分多图任务之前下载一次并上传为私有 OSS 临时对象。EasyRouter 使用 Gemini `fileData.fileUri` 签名 URL；OpenRouter 使用 `input_references` Base64 Data URL，并仅在实际调用 OpenRouter 时读取临时对象。同一批次的生成任务和服务商回退共享临时对象。上传完成后服务立即释放本地参考图数据，全部模型调用结束后主动删除临时对象。

```powershell
$body = @{
  requestId = "req-001"
  prompt = "生成一张极简风格的海报"
  model = "gemini-3.1-flash-image"
  aspectRatio = "16:9"
  imageSize = "2K"
  imageCount = 2
  fileUrls = @(
    "https://example.com/reference-1.png"
  )
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:5000/api/process-image `
  -ContentType "application/json" `
  -Body $body
```

服务商失败响应包含稳定的 `errorCode`。输入参数或参考图无效返回 HTTP 400；主服务商和兜底服务商均不可用、生成队列超时等可重试故障返回 HTTP 503。例如：

```json
{
  "success": false,
  "message": "模型服务暂时不可用，请稍后重试。",
  "timestamp": "2026-08-06T08:00:00+00:00",
  "data": {},
  "errorCode": "provider_unavailable"
}
```

### 图片生成接口（直接返回图片文件）

Gemini：
```powershell
$body = @{
  prompt = "a cute cat"
  model = "gemini-3.1-flash-image"
  aspectRatio = "1:1"
  imageSize = "1K"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:5000/api/generate-image `
  -ContentType "application/json" `
  -Body $body `
  -OutFile "output.png"
```

GPT Image 2：
```powershell
$body = @{
  prompt = "a cute cat"
  model = "gpt-image-2"
  aspectRatio = "1:1"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:5000/api/generate-image `
  -ContentType "application/json" `
  -Body $body `
  -OutFile "output.png"
```

### 图片理解接口（返回文本）

```powershell
$body = @{
  requestId = "req-001"
  prompt = "描述这张图片的内容"
  model = "gemini-3.1-flash-image"
  fileUrls = @(
    "https://example.com/photo.png"
  )
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:5000/api/understand-image `
  -ContentType "application/json" `
  -Body $body
```
