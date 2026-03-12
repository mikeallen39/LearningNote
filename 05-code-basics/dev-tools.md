# 开发工具笔记

---

## Docker

### Docker vs 虚拟机
虚拟机需要占用大量资源，启动速度慢。Docker 只是容器的一种实现，容器也是一种虚拟化技术。

### 基本概念
- **镜像**：只读模板，用来创建容器
- **容器**：Docker 的运行实例
- **仓库**：存储镜像的地方（如 DockerHub）

镜像和容器的关系就像 Java 中的类和实例。

Docker 使用 client-server 架构模式，容器数据不是持久化的，需要通过逻辑卷实现数据持久化。

### 容器化步骤

```bash
# 1. 创建 Dockerfile
# 2. 构建镜像
docker build -t hello-docker .

# 3. 查看镜像
docker images

# 4. 运行容器
docker run hello-docker

# 5. 下载镜像
docker pull username/hello-docker
```

示例 Dockerfile：
```dockerfile
FROM node:14-alpine
COPY index.js /index.js
CMD node /index.js
```

### Docker Compose
通过 `docker-compose.yaml` 配置文件将应用程序的多个模块（前端、后端、数据库、缓存、负载均衡）关联起来。

```bash
docker compose up
```

在线环境：Play with Docker

---

## Git 常用命令

### 创建分支并添加项目

```bash
git init
git remote add origin https://xxx.git
git checkout -b 分支名
git add .
git commit -m "初始化"
git push -u origin 分支名
```

### 下载项目分支代码

```bash
git clone -b 分支名 https://xxx.git

# 上传代码
git checkout 分支名
git add .
git commit -m ""
git push
```

---

## FastAPI

### 安装

```bash
pip install "fastapi[all]"
```

### 运行

```bash
uvicorn main:app --reload
```
