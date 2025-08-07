#!/bin/bash

# HiggsAudio API 服务器启动脚本

echo "🚀 启动 HiggsAudio API 服务器..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import fastapi, uvicorn, requests, loguru, soundfile" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  警告: 缺少依赖包，正在安装..."
    pip install fastapi uvicorn requests loguru soundfile
fi

# 设置默认参数
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
MODEL_PATH=${MODEL_PATH:-"bosonai/higgs-audio-v2-generation-3B-base"}
AUDIO_TOKENIZER_PATH=${AUDIO_TOKENIZER_PATH:-"bosonai/higgs-audio-v2-tokenizer"}
DEVICE=${DEVICE:-"auto"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-"2048"}

echo "🔧 配置信息:"
echo "  主机: $HOST"
echo "  端口: $PORT"
echo "  模型: $MODEL_PATH"
echo "  设备: $DEVICE"

# 启动服务器
echo "🎵 启动服务器..."
python3 api_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --audio-tokenizer-path "$AUDIO_TOKENIZER_PATH" \
    --device "$DEVICE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "$@"

echo "✅ 服务器已停止" 