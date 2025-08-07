# HiggsAudio API 服务器

这是一个基于FastAPI的HiggsAudio音频生成API服务器，提供了RESTful接口来访问HiggsAudio模型的音频生成功能。

## 功能特性

- 🎵 **音频生成**: 将文本转换为高质量语音
- 🎭 **语音克隆**: 基于参考音频进行语音克隆
- 🎪 **多说话人**: 支持多说话人对话生成
- 🌍 **多语言支持**: 支持中英文等多种语言
- ⚡ **流式生成**: 支持实时流式音频生成
- 🎨 **场景描述**: 支持场景描述来影响音频风格
- 📊 **参数调优**: 支持温度、top-k、top-p等生成参数

## 安装依赖

```bash
pip install fastapi uvicorn requests loguru soundfile
```

## 启动服务器

### 基本启动

```bash
python api_server.py
```

### 自定义参数启动

```bash
python api_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path bosonai/higgs-audio-v2-generation-3B-base \
  --audio-tokenizer-path bosonai/higgs-audio-v2-tokenizer \
  --device auto \
  --max-new-tokens 2048
```

### 开发模式启动

```bash
python api_server.py --reload
```

## API 端点

### 1. 服务器信息

**GET** `/info`

获取服务器和模型信息。

**响应示例:**
```json
{
  "model_name": "bosonai/higgs-audio-v2-generation-3B-base",
  "device": "cuda",
  "is_loaded": true,
  "max_new_tokens": 2048,
  "supported_features": [
    "单说话人音频生成",
    "多说话人音频生成",
    "语音克隆",
    "智能语音",
    "场景描述",
    "流式生成"
  ]
}
```

### 2. 获取可用语音

**GET** `/voices`

获取可用的语音样本列表。

**响应示例:**
```json
{
  "voices": [
    {
      "name": "belinda",
      "file": "examples/voice_prompts/belinda.wav",
      "type": "wav"
    }
  ],
  "count": 1
}
```

### 3. 获取可用场景

**GET** `/scenes`

获取可用的场景描述列表。

**响应示例:**
```json
{
  "scenes": [
    {
      "name": "quiet_indoor",
      "content": "A quiet indoor environment...",
      "file": "examples/scene_prompts/quiet_indoor.txt"
    }
  ],
  "count": 1
}
```

### 4. 生成音频

**POST** `/generate`

生成音频文件。

**请求体:**
```json
{
  "text": "Hello, this is a test.",
  "ref_audio": "belinda",
  "scene_prompt": "quiet_indoor",
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.95,
  "max_new_tokens": 2048,
  "seed": 12345,
  "ras_win_len": 7,
  "ras_win_max_num_repeat": 2
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "音频生成成功",
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
  "sampling_rate": 24000,
  "duration": 3.5,
  "generated_text": "Hello, this is a test."
}
```

### 5. 流式生成音频

**POST** `/generate-stream`

流式生成音频，支持实时返回生成进度。

**请求体:** 与 `/generate` 相同

**响应:** Server-Sent Events (SSE) 格式

### 6. 上传音频文件生成

**POST** `/generate-upload`

通过上传参考音频文件来生成音频。

**请求:** multipart/form-data 格式

- `text`: 要转换的文本
- `ref_audio_file`: 参考音频文件 (可选)
- `temperature`: 生成温度
- `top_k`: Top-K参数
- `top_p`: Top-P参数
- `max_new_tokens`: 最大生成token数
- `seed`: 随机种子

## 参数说明

### 生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | string | 必需 | 要转换为语音的文本 |
| `ref_audio` | string | null | 参考音频文件名（不包含扩展名） |
| `scene_prompt` | string | "quiet_indoor" | 场景描述 |
| `temperature` | float | 0.7 | 生成温度 (0.0-2.0) |
| `top_k` | int | 50 | Top-K采样参数 |
| `top_p` | float | 0.95 | Top-P采样参数 |
| `max_new_tokens` | int | 2048 | 最大生成token数 |
| `seed` | int | null | 随机种子 |
| `ras_win_len` | int | 7 | RAS窗口长度 |
| `ras_win_max_num_repeat` | int | 2 | RAS最大重复次数 |

### 分块参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_method` | string | null | 分块方法：speaker/word/None |
| `chunk_max_word_num` | int | 200 | 单词分块最大词数 |
| `chunk_max_num_turns` | int | 1 | 说话人分块最大轮数 |
| `generation_chunk_buffer_size` | int | null | 生成分块缓冲区大小 |

## 使用示例

### Python 客户端

```python
import requests
import base64

# 创建客户端
client = requests.Session()
base_url = "http://localhost:8000"

# 生成音频
response = client.post(f"{base_url}/generate", json={
    "text": "Hello, this is a test of the HiggsAudio API.",
    "ref_audio": "belinda",
    "temperature": 0.7,
    "max_new_tokens": 512,
    "seed": 12345
})

if response.status_code == 200:
    result = response.json()
    if result['success']:
        # 保存音频文件
        audio_bytes = base64.b64decode(result['audio_base64'])
        with open('output.wav', 'wb') as f:
            f.write(audio_bytes)
        print(f"音频已保存，时长: {result['duration']:.2f}秒")
```

### cURL 示例

```bash
# 生成音频
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "ref_audio": "belinda",
    "temperature": 0.7,
    "max_new_tokens": 512
  }'

# 获取可用语音
curl "http://localhost:8000/voices"

# 获取服务器信息
curl "http://localhost:8000/info"
```

### JavaScript 客户端

```javascript
// 生成音频
async function generateAudio(text, refAudio = null) {
    const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            ref_audio: refAudio,
            temperature: 0.7,
            max_new_tokens: 512
        })
    });
    
    const result = await response.json();
    if (result.success) {
        // 播放音频
        const audioBytes = atob(result.audio_base64);
        const audioBlob = new Blob([audioBytes], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
    }
}

// 使用示例
generateAudio("Hello, this is a test.", "belinda");
```

## 测试

使用提供的测试客户端：

```bash
# 运行所有测试
python test_client.py

# 运行特定测试
python test_client.py --test basic
python test_client.py --test voice-clone
python test_client.py --test streaming

# 指定服务器URL
python test_client.py --url http://localhost:8000
```

## 配置选项

### 服务器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | 服务器主机地址 |
| `--port` | 8000 | 服务器端口 |
| `--model-path` | bosonai/higgs-audio-v2-generation-3B-base | 模型路径 |
| `--audio-tokenizer-path` | bosonai/higgs-audio-v2-tokenizer | 音频tokenizer路径 |
| `--device` | auto | 设备类型 (auto/cuda/mps/cpu) |
| `--max-new-tokens` | 2048 | 最大生成token数 |
| `--voice-prompts-dir` | examples/voice_prompts | 语音样本目录 |
| `--scene-prompts-dir` | examples/scene_prompts | 场景描述目录 |
| `--reload` | false | 开发模式重载 |

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确保有足够的GPU内存
   - 尝试使用CPU设备: `--device cpu`

2. **音频生成失败**
   - 检查输入文本格式
   - 确保参考音频文件存在
   - 调整生成参数

3. **服务器启动失败**
   - 检查端口是否被占用
   - 确保依赖包已正确安装
   - 查看日志文件 `api_server.log`

### 日志

服务器日志保存在 `api_server.log` 文件中，包含详细的运行信息和错误信息。

## 许可证

本项目遵循与原HiggsAudio项目相同的许可证。 