# 构建阶段依赖官方 uv 镜像（自带 uv + python3.11）
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# 依赖层缓存优化：先只拷贝依赖清单，改动源码不会重新下载依赖
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 源码层
COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# 默认起 Web Demo；CLI 用法：docker compose run --rm autoreport autoreport --help
CMD ["streamlit", "run", "src/autoreport/ui/streamlit_app.py", "--server.address", "0.0.0.0"]
