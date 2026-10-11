# ============================================================
# 运营智脑 · 一键部署
# 面试考点：为什么 Agent 服务要容器化？
#   ①依赖地狱：LangGraph / CUDA / 各种 SDK 版本冲突，容器是唯一可靠的隔离
#   ②环境一致性：本地能跑、服务器不能跑，99% 是环境差异
#   ③弹性扩缩：K8s 里按 CPU/队列长度自动扩 worker
# ============================================================
FROM python:3.11-slim

# 时区与基础工具
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖再拷代码 —— 利用 Docker 层缓存，改代码不用重装依赖（构建从 3min 降到 10s）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 非 root 运行（安全基线，生产必须）
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/docs && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查：K8s 探针会读这个
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
