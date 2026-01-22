# kyc-test-tool

KYC 自动化测试工具 - 支持身份证认证和视频认证的完整流程测试。

## Docker 部署

### 前置要求

- Docker 19.03+ (支持 GPU)
- Docker Compose
- NVIDIA GPU 驱动 (GPU 版本)
- nvidia-docker 或 Docker 19.03+ 原生 GPU 支持

### 快速启动

1. **创建必要目录**
   ```bash
   mkdir -p cache/liveportrait
   ```
   
   > **说明**：`logs` 和 `kyc_test` 目录由程序自动创建，无需手动操作。

2. **构建并启动服务**
   ```bash
   # 构建 Docker 镜像
   docker-compose build
   
   # 启动服务（后台运行）
   docker-compose up -d
   
   # 查看启动日志
   docker-compose logs -f
   ```

3. **等待模型加载**
   
   服务启动后需要 5-10 秒加载深度学习模型，看到以下日志表示启动成功：
   ```
   LivePortrait 模型预加载完成
   Application startup complete.
   Uvicorn running on http://0.0.0.0:9000
   ```

### 服务管理

```bash
# 停止服务
docker-compose stop

# 重启服务
docker-compose restart

# 停止并删除容器
docker-compose down

# 查看运行状态
docker-compose ps

# 查看实时日志
docker logs -f kyc-api-service
```

## 测试使用

### 方式1：通过 Swagger 页面测试（推荐）

1. **访问 API 文档**
   
   打开浏览器访问：http://localhost:9000/docs

2. **测试自动 KYC 流程**
   
   - 找到 `GET /auto_kyc` 接口
   - 点击 "Try it out"
   - 在 `action` 参数中输入动作类型（可选值：`mouth_open`, `left_shake`, `right_shake`, `nod`）
   - 点击 "Execute" 执行测试
   - 查看响应结果中的 `final_status` 字段：
     - `2` = 已完成 ✅
     - `1` = 已认证视频
     - `0` = 未完成

3. **查看生成的测试数据**
   
   测试完成后，在 `kyc_test/<user_id>/` 目录下可以找到：
   - `avatar.png` - 用户头像
   - `idcard_front.png` - 身份证正面
   - `idcard_back.png` - 身份证反面
   - `mouth_open.mp4` - 生成的认证视频

### 方式2：通过命令行测试

```bash
# 测试张嘴动作
curl "http://localhost:9000/auto_kyc?action=mouth_open"

# 测试左摇头动作
curl "http://localhost:9000/auto_kyc?action=left_shake"

# 测试右摇头动作
curl "http://localhost:9000/auto_kyc?action=right_shake"

# 测试点头动作
curl "http://localhost:9000/auto_kyc?action=nod"
```

### 响应示例

**成功响应**：
```json
{
  "user_id": "8527306768",
  "message": "KYC流程测试完成",
  "final_status": 2,
  "status_text": "已完成",
  "details": {
    "idcard_front": "✅ 通过",
    "idcard_back": "✅ 通过",
    "face_collection": "✅ 通过",
    "video_verification": {
      "mouth_open": "✅ 通过"
    }
  }
}
```

## 配置说明

### 端口配置

- API 服务：`9000`（可在 docker-compose.yml 中修改）

### 目录挂载

| 宿主机目录 | 容器目录 | 说明 |
|-----------|---------|------|
| `./LivePortrait/pretrained_weights` | `/app/LivePortrait/pretrained_weights` | 模型权重（只读） |
| `./LivePortrait/assets/examples/driving` | `/app/LivePortrait/assets/examples/driving` | 驱动视频（只读） |
| `./cache/liveportrait` | `/app/cache/liveportrait` | 模板缓存（可写） |
| `./logs` | `/app/logs` | 日志输出 |
| `./kyc_test` | `/app/kyc_test` | 测试数据输出 |
| `./avatars` | `/app/avatars` | 头像库 |

### GPU 配置

服务默认使用 GPU 0，可通过环境变量修改：
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0  # 修改为其他 GPU ID
  - CUDA_VISIBLE_DEVICES=0
```

## 故障排查

### 服务启动失败

```bash
# 查看详细日志
docker-compose logs kyc-api-service

# 检查 GPU 是否可用
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 视频生成失败

1. 确认 `cache/liveportrait` 目录存在且可写
2. 检查 GPU 内存是否充足（建议至少 4GB）
3. 查看容器日志中的详细错误信息

### 常见错误

- **"No module named 'torch'"**: 镜像构建不完整，重新执行 `docker-compose build --no-cache`
- **"Read-only file system"**: 缓存目录挂载配置错误，确认 docker-compose.yml 配置正确
- **"CUDA out of memory"**: GPU 内存不足，尝试减少并发请求或使用更大显存的 GPU

 