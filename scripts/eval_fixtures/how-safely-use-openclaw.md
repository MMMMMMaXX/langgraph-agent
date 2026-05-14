如何安全的养一只小龙虾（OpenClaw）？

> **分享人**: 王贵龙 (wangguilong)
> **日期**: 2026-03-18
> **目标受众**: 开发者、运维工程师、安全工程师
> **预计时长**: 45 分钟分享 + 15 分钟 Q&A

---

# 1 OpenClaw 简介

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=41aae6cb9b5e4fb7945871a6f1ab11e4&docGuid=OZhaWn3rdPxhy3)
2026 年火爆全网的 OpenClaw，定位为开源可自部署的 AI 个人助理（曾用名 Clawdbot、Moltbot）。

官网地址：[https://docs.openclaw.ai/](https://docs.openclaw.ai/)

github 地址：[https://github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)

最近使用几天后的个人感受：

1. 更聪明，记忆能力超强
2. 慢，可能是很多时候 skill 新安装，他有很多测试验证逻辑，另外可能会带来 token 消耗大的问题
3. 配套设施完善，支持定时任务、心跳、skills、Tools

## 1.1 OpenClaw 架构简介

OpenClaw 是一个开源的 AI 助手框架，其核心架构如下：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=a00916b09ef34cb9b8b88d9283f0a787&docGuid=OZhaWn3rdPxhy3)
**核心功能**：

- **Multi-channel gateway**: 多渠道网关。只需一个网关进程，即可支持 WhatsApp、Telegram、Discord 和 iMessage。
- **Plugin channels**: 支持插件化渠道，可接入自定义消息
- **Multi-agent routing**: 为每个智能体、工作区或发送者提供隔离的会话。
- **Media support**: 支持发送和接收图像、音频及各类文档。
- **Web Control UI：**提供浏览器端仪表盘，用于管理聊天、配置、会话和节点。
- **Mobile nodes：**配对 iOS 和 Android 节点，支持 Canvas、摄像头及语音驱动的工作流。

## 1.2 OpenClaw 安全策略简介

OpenClaw 内置了多层安全机制：

#### 🔐 **鉴权机制**

- **API Key/Token 认证**: Gateway 级别的访问控制
- **会话隔离**: 不同会话间的资源隔离
- **插件权限**: 细粒度的工具调用权限

#### 🏝️ **执行隔离**

- **沙盒环境**: 代码执行在受限环境中
- **资源限制**: CPU/内存/磁盘使用限制
- **网络隔离**: 限制外部网络访问

#### 🛠️ **工具调用控制**

- **白名单机制**: 只允许预定义的工具调用
- **权限分级**: 不同工具的不同权限级别
- **审核机制**: 危险操作需要人工确认

#### 🌐 **网络与文件访问控制**

- **文件沙盒**: 限制文件系统访问范围
- **网络策略**: 控制出站/入站连接
- **环境隔离**: 不同环境的不同安全策略

---

# 2 主流的 OpenClaw 部署方式及安全风险

## 2.1 本机直接运行

部署可参考官网，一键安装，参考链接：[https://docs.openclaw.ai/install](https://docs.openclaw.ai/install)

**安全风险分析**:

- ✅ **优点**: 简单快速，适合开发调试
- ❌ **风险**:
  - **全权限访问**: 以当前用户权限运行，可访问所有文件
  - **无网络隔离**: 可以访问本地所有服务
  - **持久化风险**: 会话数据可能泄露敏感信息
  - **依赖污染**: 使用系统 Python 环境，可能被恶意包污染

## 2.2 开发机 / 云主机裸部署

**安全风险分析**:

1. **端口扫描**: 18789 端口暴露在公网
2. **未授权访问**: 默认无认证或弱认证
3. **API 滥用**: 无速率限制，可被用于攻击跳板
4. **信息泄露**: 版本信息、配置信息可能泄露
5. **供应链攻击**: 依赖包可能被植入恶意代码

---

# 3 基于 Docker 容器的 OpenClaw 安全部署方案

## 3.1 Docker 简介

**容器隔离机制**:

- **Namespaces**: PID, Network, Mount, IPC, UTS, User
- **Cgroups**: CPU, Memory, Disk I/O, Network
- **UnionFS**: 分层文件系统

**与虚拟机的区别**:

| 维度     | Docker 容器 | 虚拟机 |
| -------- | ----------- | ------ |
| 启动速度 | 秒级        | 分钟级 |
| 资源占用 | 低          | 高     |
| 隔离性   | 进程级      | 系统级 |
| 镜像大小 | MB 级       | GB 级  |
| 安全性   | 中等        | 高     |

**为什么 Docker 能降低风险**:

1. **资源隔离**: 限制容器资源使用
2. **文件隔离**: 容器有独立文件系统
3. **网络隔离**: 默认有独立网络栈
4. **进程隔离**: 看不到主机进程

**Docker 引入的新问题**:

- **性能开销**: 约 5-10%的性能损失
- **权限问题**: 容器内 root != 主机 root
- **逃逸风险**: 内核漏洞可能导致逃逸
- **镜像安全**: 镜像可能包含恶意代码

## 3.2 本地安全部署方案

OpenClaw 官方基于 docker 的部署指导文档：[https://docs.openclaw.ai/install/docker#docker](https://docs.openclaw.ai/install/docker#docker)

### 3.2.1 安装 Docker

macOS 下载 Docker Desktop App 安装即可，无需额外配置！下载地址：[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=0587e36441744866aa8d642f70d38bce&docGuid=OZhaWn3rdPxhy3)
安装完启动 app，在终端使用命令验证 docker 是否安装成功：`docker --version`

### 3.2.2 准备 Docker 配置文件

1、新建一个 docker 配置地址存放路径，存放启动 docker 容器所需的 `.env` 和 `docker-compose.yml` 文件。

2、准备 Openclaw 网关 token，使用 openssl 生成随机令牌：`openssl rand -hex 32`

3、准备 `.env` 环境变量文件，基础配置如下：

```json
DOCKER_GID=
OPENCLAW_HOME_NODE_DIR=
OPENCLAW_IMAGE=ghcr.io/openclaw/openclaw:latest
OPENCLAW_GATEWAY_TOKEN=
OPENCLAW_ALLOW_INSECURE_PRIVATE_WS=0
OPENCLAW_GATEWAY_BIND=lan
OPENCLAW_GATEWAY_PORT=18789
OPENCLAW_BRIDGE_PORT=18790
```

**⚠️ 注意：**

- DOCKER_GID，非必需，容器内操作 docker 命令使用，取值通过命令：`stat -f %g /var/run/docker.sock` 获取。
- OPENCLAW_GATEWAY_TOKEN：网关 token，使用第一步生成的随机令牌字符串
- OPENCLAW_IMAGE，镜像地址，**官方镜像地址：ghcr.io/openclaw/openclaw:latest。请勿使用任何非官方的镜像地址，谨防木马、后门等安全风险。**
- OPENCLAW_GATEWAY_PORT，网关端口号，默认为 18789，可以修改；
- OPENCLAW_HOME_NODE_DIR，OpenClaw home/node 工作区路径，路径地址生成参考章节：3.2.3（如：`/Users/name/Documents/OpenClaw`）；

4、准备 `docker-compose.yml`镜像编排文件

```json
networks:
  openclaw-network:
    name: openclaw-network
    driver: bridge

services:
  openclaw-gateway:
    image: ${OPENCLAW_IMAGE:-openclaw:local}
    container_name: openclaw-gateway
    environment:
      PUID: 1000
      PGID: 1000
      TZ: Asia/Shanghai
      HOME: /home/node
      TERM: xterm-256color
      OPENCLAW_GATEWAY_TOKEN: ${OPENCLAW_GATEWAY_TOKEN:-}
      OPENCLAW_ALLOW_INSECURE_PRIVATE_WS: ${OPENCLAW_ALLOW_INSECURE_PRIVATE_WS:-}
    volumes:
      - ${OPENCLAW_GATEWAY_TOKEN}:/home/node
      - /var/run/docker.sock:/var/run/docker.sock
    group_add:
      - "${DOCKER_GID:-999}"
    ports:
      - "127.0.0.1:${OPENCLAW_GATEWAY_PORT:-18789}:18789"
      - "127.0.0.1:${OPENCLAW_BRIDGE_PORT:-18790}:18789"
    init: true
    restart: unless-stopped
    command:
      [
        "node",
        "dist/index.js",
        "gateway",
        "--bind",
        "${OPENCLAW_GATEWAY_BIND:-lan}",
        "--port",
        "${OPENCLAW_GATEWAY_PORT:-18789}",
      ]
    healthcheck:
      test:
        [
          "CMD",
          "node",
          "-e",
          "fetch('http://127.0.0.1:18789/healthz').then((r)=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))",
        ]
      interval: 30s
      timeout: 5s
      retries: 5
      start_period: 20s
    networks:
      - openclaw-network
```

**⚠️ 注意：**

- **services.openclaw-gateway.ports 必须设置为：**`"127.0.0.1:${OPENCLAW_GATEWAY_PORT:-18789}:18789"`**，它代表仅支持本地访问，局域网内的其它机器无法访问。**
- 其它配置无特殊情况无需修改

### 3.2.3 准备 OpenClaw 配置文件

1、准备一个 home/node 路径，相当于整个容器的工作区，比如：/Users/name/Documents/OpenClaw。

**⚠️ 注意：**

- 这个路径是 docker 可读写路径，最好新建一个路径，**禁止使用系统根目录或者存放重要文件的目录，谨防数据丢失**。
- 后续的依赖库、skills、plugin 的安装都会放到这里，OpenClaw 用到的记忆，生成和修改的文件都会在这里。

2、在 home/node 以下路径 `.openclaw/openclaw.json`（比如：/Users/name/Documents/OpenClaw/.openclaw/openclaw.json）创建文件，基础配置如下：

```json
{
  "auth": {
    "profiles": {
      "qianfan:default": {
        "provider": "qianfan",
        "mode": "api_key"
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "qianfan": {
        "baseUrl": "https://qianfan.baidubce.com/v2",
        "api": "openai-completions",
        "models": [
          {
            "id": "deepseek-v3.2",
            "name": "DEEPSEEK V3.2",
            "reasoning": true,
            "input": ["text"],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 98304,
            "maxTokens": 32768
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "qianfan/deepseek-v3.2"
      },
      "models": {
        "qianfan/deepseek-v3.2": {
          "alias": "QIANFAN"
        }
      },
      "workspace": "/home/node/.openclaw/workspace"
    }
  },
  "tools": {
    "profile": "coding"
  },
  "commands": {
    "native": "auto",
    "nativeSkills": "auto",
    "restart": true,
    "ownerDisplay": "raw"
  },
  "session": {
    "dmScope": "per-channel-peer"
  },
  "gateway": {
    "port": 18789,
    "mode": "local",
    "bind": "lan",
    "auth": {
      "mode": "token"
    },
    "controlUi": {
      "enabled": true,
      "allowedOrigins": ["http://127.0.0.1:18789", "http://localhost:18789"]
    },
    "tailscale": {
      "mode": "off",
      "resetOnExit": false
    },
    "nodes": {
      "denyCommands": [
        "camera.snap",
        "camera.clip",
        "screen.record",
        "contacts.add",
        "calendar.add",
        "reminders.add",
        "sms.send"
      ]
    }
  }
}
```

**⚠️ 注意：**

- 模型相关配置换成自己的，示例中使用的是千帆（qianfan），自己部署可以使用智谱 GLM，便宜且目前送免费额度。
- `gateway.controlUi` 为 webUI 配置项，**其中 allowedOrigins 代表允许你访问的源地址，在本地使用时参考设置即可，修改请谨慎**
- `gateway.auth` 配置为 token 访问模式，官方示例会直接把 token 暴露配置文件中，亲测可以去掉，防止 token 泄露
- 更多配置可以在后续通过对话进行修改，也可以参考官方文档：

3、在 home/node 以下路径 `.openclaw/agents/main/agent/auth-profiles.json`（比如：/Users/name/Documents/OpenClaw/.openclaw/agents/main/agent/auth-profiles.json）创建认证相关配置文件，基础配置如下：

```json
{
  "version": 1,
  "profiles": {
    "qianfan:default": {
      "type": "api_key",
      "provider": "qianfan",
      "key": ""
    }
  }
}
```

**⚠️ 注意：**

- 不存在的文件夹自行创建即可。
- 更换为自己模型对应的配置。

### 3.2.4 启动容器并进行初始配置

在配置路径启动 docker 容器，命令：`docker compose up -d`

在浏览器输入地址：[http://127.0.0.1:18789/](http://127.0.0.1:18789/) 理论上就能看到 OpenClaw 的 webUI 页面，首次使用需要配对。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=24fe29beb63846ac8e94b878d698d9cb&docGuid=OZhaWn3rdPxhy3)
输入 token 进行连接，需要在容器中进行验证配对

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=0e1176cdbba94679bf9cd108cff53e0e&docGuid=OZhaWn3rdPxhy3)
在容器内使用命令：`openclaw devices list` 获取待配对设备，使用命令：`openclaw devices approve <requestId>` 对设备进行配对确认。

配对完成后回到 webUI 页面，再连接一次即可进入网关页面，看到如下页面就可以愉快的和你的小龙虾对话了～。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=eac2047804c840bbb2ceffb91583ed96&docGuid=OZhaWn3rdPxhy3)
**⚠️ 注意：**

- 使用 docker 部署，OpenClaw 相关的运维命令（如上面使用到的`openclaw devices list`等）需要在容器中执行，在本机的终端无法生效。

## 3.3 开发机（VPS，云服务器等）安全部署方案

在服务器上部署和本地部署整体流程是一致的，主要有以下不同：

1. 服务器的网络环境可以类比为更广的局域网，那么部署的服务会存在公网暴露的问题，如果有漏洞被攻击容易造成资料泄露等风险。
2. 服务器一般是 Linux 系统且没有图形化环境，很多文件操作不方便，一般会依赖一些额外的服务操作

所以，服务器部署采用以下方案：

1. 依然使用 docker 来简化部署流程
2. 使用 nginx 反向代理在外网和内部 OpenClaw 网关中间做一层安全校验（https+用户验证）
3. 开启防火墙，禁用不该暴露到公网的端口，如网关的 18789 端口
4. 借助 filebrowser 等工具来实现服务器文件的可视化管理

### 3.3.1 Docker 安装

公司开发机安装参考：[10.1 公司开发机安装 docker（Ubuntu）](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/KhnB5s-mff/B-2iAEL1mG2qFD?t=mention&mt=doc&dt=doc)

推荐安装图形化 docker 部署工具：portainer

### 3.3.2 部署 Docker 容器

1、镜像编排文件`docker-compose.yml`可参考如下配置：

```json
volumes:
  openclaw-home-node:
    name: openclaw-home-node
  filebrowser-data:
    name: openclaw-filebrowser-data
  npm-data:
    name: openclaw-npm-data
  npm-letsencrypt:
    name: openclaw-npm-letsencrypt

networks:
  openclaw-network:
    name: openclaw-network
    driver: bridge

services:
  openclaw-gateway:
    image: ${OPENCLAW_IMAGE:-openclaw:local}
    container_name: openclaw-gateway
    environment:
      PUID: 1000
      PGID: 1000
      TZ: Asia/Shanghai
      HOME: /home/node
      TERM: xterm-256color
      OPENCLAW_GATEWAY_TOKEN: ${OPENCLAW_GATEWAY_TOKEN:-}
      OPENCLAW_ALLOW_INSECURE_PRIVATE_WS: ${OPENCLAW_ALLOW_INSECURE_PRIVATE_WS:-}
    volumes:
      - openclaw-home-node:/home/node
      - /var/run/docker.sock:/var/run/docker.sock
    group_add:
      - "${DOCKER_GID:-999}"
    ports:
      - "127.0.0.1:${OPENCLAW_GATEWAY_PORT:-18789}:18789"
      - "127.0.0.1:${OPENCLAW_BRIDGE_PORT:-18790}:18789"
    init: true
    restart: unless-stopped
    command:
      [
        "node",
        "dist/index.js",
        "gateway",
        "--bind",
        "${OPENCLAW_GATEWAY_BIND:-lan}",
        "--port",
        "${OPENCLAW_GATEWAY_PORT:-18789}",
      ]
    healthcheck:
      test:
        [
          "CMD",
          "node",
          "-e",
          "fetch('http://127.0.0.1:18789/healthz').then((r)=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))",
        ]
      interval: 30s
      timeout: 5s
      retries: 5
      start_period: 20s
    networks:
      - openclaw-network

  nginx-proxy-manager:
    image: jc21/nginx-proxy-manager:latest
    container_name: openclaw-nginx-proxy-manager
    restart: unless-stopped
    privileged: true
    ports:
      - 8501:81
      - 8500:80
    environment:
      PUID: 1000
      PGID: 1000
      TZ: Asia/Shanghai
    volumes:
      - npm-data:/data
      - npm-letsencrypt:/etc/letsencrypt
    networks:
      - openclaw-network

  filebrowser:
    image: filebrowser/filebrowser:latest
    container_name: openclaw-filebrowser
    environment:
      PUID: 1000
      PGID: 1000
      TZ: Asia/Shanghai
    volumes:
      - filebrowser-data:/config
      - filebrowser-data:/database
      - openclaw-home-node:/srv
    ports:
      - 8600:80
    restart: unless-stopped
    privileged: true
    networks:
      - openclaw-network
```

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=21df7bf4307c4ced84b9e9e3d5b4a5a4&docGuid=OZhaWn3rdPxhy3)
2、环境变量直接填到 web 页中即可

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=8fbb9ad66c1d41c0890199023cabaa01&docGuid=OZhaWn3rdPxhy3)
**⚠️ 说明：**

- `docker-compose.yml` 中的 `volumes` 代表 docker 的数据卷，类比电脑的硬盘；
- `openclaw-gateway` 容器为 OpenClaw 网关
- `openclaw-nginx-proxy-manager` 为可视化 nginx 管理容器
- `openclaw-filebrowser` 为可视化文件管理容器，可用于操作 openclaw-home-node 的所有文件

### 3.3.3 Nginx 配置

1、打开`nginx-proxy-manager`后台管理页面，增加如下反向代理：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=b4eecd556e7e4b7cb445c590215afd82&docGuid=OZhaWn3rdPxhy3)
**⚠️ 说明：**

- `Forward Hostname / IP` 需要填写容器的 ip，使用 docker compose 部署，填写 service name 即可，如：openclaw-gateway
- `Access List`增加用户密码校验，也可以使用第三方 OAuth 服务
- `Websockets Support` 需要打开，因为 OpenClaw 使用 websocket 进行通信

2、HTTPS 证书配置

略过，公司内网使用零信任，默认提供 https 支持。

### 3.3.4 准备 OpenClaw 配置文件

访问 filebrowser 管理后台，通过 web 可视化网页新建 `.openclaw/openclaw.json` 和 `.openclaw/agents/main/agent/auth-profiles.json` 文件，配置文件内容可参考章节 3.2.3。

**⚠️ 注意：**

- `gateway.controlUi.allowedOrigins` 需要配置为你访问的正式域名。

### 3.3.5 容器启动和 OpenClaw 配置

参考章节 3.2.4，有新设备访问时需要配对，配对完成即可正常访问。

**⚠️ 说明：**

- 可以通过 portainer 提供的 web 页进入容器内部执行命令。

## 3.4 总结

| 架构方案                          | 安全性  | 部署成本 | 性能  | 适用场景                                                                        |
| --------------------------------- | ------- | -------- | ----- | ------------------------------------------------------------------------------- |
| **裸机部署**                      | 🟡 中   | 🟢 低    | 🟢 高 | 可以使用单独的机器来部署，如：MacMini，如果你有闲置的机器，也可以采用这种方案。 |
| **Docker 部署**                   | 🟠 中高 | 🟡 中    | 🟡 中 | 部署成本适中，且可以支持快速扩容，但部分功能受限（docker 和宿主机隔离导致）     |
| **Virtual Machine（虚拟机）部署** | 🟢 高   | 🟠 中高  | 🟡 中 | 部署和运维成本较高，暂未深入研究。                                              |

---

# 4 进阶配置

## 4.1 自定义镜像构建

下载 Openclaw 源码，在根目录使用以下脚本指定参数进行镜像编译：

```json
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --build-arg OPENCLAW_DOCKER_APT_PACKAGES='git curl wget jq python3-pip build-essential ffmpeg sqlite3 libpq-dev default-libmysqlclient-dev' \
  --build-arg OPENCLAW_EXTENSIONS='ollama,openai,anthropic,gemini,signal,docker-sandbox,browser,nostr-tools,python-interpreter' \
  --build-arg OPENCLAW_INSTALL_DOCKER_CLI=1 \
  -t ghcr.io/openclaw/openclaw:latest \
  -f Dockerfile . --no-cache
```

**⚠️ 说明：**

- `OPENCLAW_DOCKER_APT_PACKAGES` 额外编译到镜像中的 apt 包，常用如 git，wget，pip 等。
- `OPENCLAW_EXTENSIONS`额外编译到镜像中的插件，常用如：docker-sandbox（容器执行 Agent 任务）、browser（浏览器插件）等；
- `OPENCLAW_INSTALL_DOCKER_CLI` 是否安装 docker cli 环境，打开可支持在 docker 中调用 docker。
- `-t`参数可以指定镜像名。

## 4.2 沙盒环境（Sandbox）

官方文档：[https://docs.openclaw.ai/gateway/sandboxing](https://docs.openclaw.ai/gateway/sandboxing)

OpenClaw 可以在沙箱后端中运行工具（Tools），当启用沙箱时，工具执行会在隔离的沙箱环境中进行，以降低潜在影响范围。这是一个可选功能，通过配置项（`agents.defaults.sandbox` 或 `agents.list[].sandbox`）进行控制。

---

# 5 参考资料

1. [https://bbs.huaweicloud.com/blogs/474393](https://bbs.huaweicloud.com/blogs/474393)
2. [https://lumadock.com/tutorials/openclaw-docker-kubernetes](https://lumadock.com/tutorials/openclaw-docker-kubernetes)
3. [https://docs.openclaw.ai/install/docker#docker](https://docs.openclaw.ai/install/docker#docker)
4. [https://docs.openclaw.ai/gateway/sandboxing](https://docs.openclaw.ai/gateway/sandboxing)
5. [https://github.com/jhaertf/openclaw-sandboxed](https://github.com/jhaertf/openclaw-sandboxed)
