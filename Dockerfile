# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY api_requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r api_requirements.txt

# 复制应用文件
COPY api_server.py .
COPY test_client.py .
COPY start_server.sh .

# 设置权限
RUN chmod +x start_server.sh

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEVICE=auto

# 启动命令
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"] 