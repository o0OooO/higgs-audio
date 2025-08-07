# HiggsAudio API 快速开始指南

## 🚀 快速启动

### 方法1: 直接运行

```bash
# 1. 安装依赖
pip install -r api_requirements.txt

# 2. 启动服务器
python api_server.py

# 3. 测试API
python test_client.py
```

### 方法2: 使用启动脚本

```bash
# 使用默认配置启动
./start_server.sh

# 或自定义参数
HOST=127.0.0.1 PORT=8080 ./start_server.sh
```

### 方法3: 使用Docker

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 📋 API 快速测试

### 1. 检查服务器状态

```bash
curl http://localhost:8000/info
```

### 2. 生成音频

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the HiggsAudio API.",
    "temperature": 0.7,
    "max_new_tokens": 512
  }'
```

### 3. 使用Python客户端

```python
import requests
import base64

# 生成音频
response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello, this is a test.",
    "temperature": 0.7,
    "max_new_tokens": 512
})

if response.status_code == 200:
    result = response.json()
    if result['success']:
        # 保存音频
        audio_bytes = base64.b64decode(result['audio_base64'])
        with open('output.wav', 'wb') as f:
            f.write(audio_bytes)
        print(f"音频已保存，时长: {result['duration']:.2f}秒")
```

## 🎯 常用功能

### 语音克隆

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a voice cloning test.",
    "ref_audio": "belinda",
    "temperature": 0.7
  }'
```

### 多语言支持

```bash
# 中文
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是中文测试。",
    "temperature": 0.7
  }'
```

### 流式生成

```bash
curl -X POST "http://localhost:8000/generate-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a streaming test.",
    "temperature": 0.7
  }'
```

## 🔧 配置选项

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | 0.0.0.0 | 服务器主机 |
| `PORT` | 8000 | 服务器端口 |
| `DEVICE` | auto | 设备类型 |
| `MODEL_PATH` | bosonai/higgs-audio-v2-generation-3B-base | 模型路径 |
| `MAX_NEW_TOKENS` | 2048 | 最大生成token数 |

### 命令行参数

```bash
python api_server.py \
  --host 127.0.0.1 \
  --port 8080 \
  --device cpu \
  --model-path your-model-path \
  --max-new-tokens 1024
```

## 📊 监控和日志

### 查看日志

```bash
# 实时日志
tail -f api_server.log

# 查看错误
grep ERROR api_server.log
```

### 健康检查

```bash
# 检查服务器状态
curl http://localhost:8000/

# 检查模型信息
curl http://localhost:8000/info
```

## 🐛 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 查找占用端口的进程
   lsof -i :8000
   
   # 杀死进程
   kill -9 <PID>
   ```

2. **模型加载失败**
   ```bash
   # 使用CPU设备
   python api_server.py --device cpu
   
   # 检查GPU
   nvidia-smi
   ```

3. **内存不足**
   ```bash
   # 减少最大token数
   python api_server.py --max-new-tokens 1024
   ```

### 调试模式

```bash
# 启用详细日志
python api_server.py --reload

# 查看详细错误信息
python -u api_server.py 2>&1 | tee debug.log
```

## 📚 更多信息

- 完整API文档: [API_README.md](API_README.md)
- 项目主页: [README.md](../README.md)
- 示例代码: [examples/](../examples/)

## 🆘 获取帮助

如果遇到问题，请：

1. 查看日志文件 `api_server.log`
2. 检查网络连接和端口
3. 确认依赖包已正确安装
4. 查看完整的API文档 