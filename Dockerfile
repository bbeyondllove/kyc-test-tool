# KYC 测试工具 - GPU 版本
# 基于 NVIDIA CUDA 12.1 + cuDNN 8 + Ubuntu 22.04 (与本地venv保持一致)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 使用官方源（阿里云源DNS解析失败）
# 清理可能存在的阿里云源配置
RUN sed -i 's|https://mirrors.aliyun.com|http://archive.ubuntu.com|g' /etc/apt/sources.list 2>/dev/null || true

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    python3-tk \
    build-essential \
    gcc \
    g++ \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 并配置清华镜像源
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip setuptools wheel

# 复制依赖文件
COPY LivePortrait/requirements_base.txt /app/LivePortrait/requirements_base.txt
COPY LivePortrait/requirements.txt /app/LivePortrait/requirements.txt
COPY setup.py /app/setup.py

# 安装 Python 依赖（使用清华镜像源）
# 1. 先安装 PyTorch GPU 版本（CUDA 12.1，与本地venv保持一致）
# 注：PyTorch需从官方源下载，清华源无CUDA版本
RUN pip3 install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --timeout 300

# 2. 安装与本地环境一致的核心依赖
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy==2.2.6 \
    onnxruntime-gpu==1.23.2 \
    opencv-python==4.12.0.88

# 3. 安装 LivePortrait 依赖
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r /app/LivePortrait/requirements_base.txt && \
    pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers==4.57.3 \
    albumentations==2.0.8 \
    einops==0.8.1 \
    kornia==0.8.2 \
    facexlib==0.3.0 \
    insightface==0.7.3 && \
    pip3 install --no-cache-dir \
    gfpgan==1.3.8

# 安装项目依赖（与本地venv一致）
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    fastapi==0.128.0 \
    uvicorn==0.40.0 \
    requests==2.32.5 \
    deepface==0.0.97

# 复制项目代码
COPY . /app/

# 安装 idcard_generator 包
RUN cd /app && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

# 创建必要的目录
RUN mkdir -p /app/logs \
    /app/kyc_test \
    /app/avatars \
    /app/LivePortrait/pretrained_weights

# 暴露端口
EXPOSE 9000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:9000/docs || exit 1

# 启动命令
CMD ["python3", "-m", "uvicorn", "kyc_api_standalone:app", "--host", "0.0.0.0", "--port", "9000"]
