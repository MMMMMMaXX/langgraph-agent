FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY README.md ./

RUN mkdir -p data outputs

# 版本元数据：由构建侧注入，应用通过 /health 暴露给监控 / 排障
ARG APP_GIT_SHA=unknown
ARG APP_BUILD_TIME=unknown
ARG APP_VERSION=dev
ENV APP_GIT_SHA=${APP_GIT_SHA} \
    APP_BUILD_TIME=${APP_BUILD_TIME} \
    APP_VERSION=${APP_VERSION}

LABEL org.opencontainers.image.title="langgraph-agent" \
      org.opencontainers.image.revision="${APP_GIT_SHA}" \
      org.opencontainers.image.created="${APP_BUILD_TIME}" \
      org.opencontainers.image.version="${APP_VERSION}"

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
