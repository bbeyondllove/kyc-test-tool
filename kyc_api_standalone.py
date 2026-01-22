"""
KYC 测试 API 服务
使用现有功能模块
运行：uvicorn kyc_api_standalone:app --host 0.0.0.0 --port 9000
"""
import os
import sys
import io
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# 配置日志
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "kyc_api.log")

# 禁用 stdout 包装，避免与 FastAPI 异步冲突
os.environ['DISABLE_STDOUT_WRAP'] = '1'

# 设置环境变量，抑制 TensorFlow 等库的日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_POLL_STRATEGY'] = 'poll'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

# 抑制所有警告
import warnings
warnings.filterwarnings('ignore')

# 配置日志系统
def setup_logging():
    """配置日志记录"""
    # 创建 logger
    logger = logging.getLogger("kyc_api")
    logger.setLevel(logging.INFO)

    # 清除已有处理器
    logger.handlers.clear()

    # 文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 初始化日志
logger = setup_logging()
logger.info("=" * 60)
logger.info(f"KYC API 服务启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"日志文件: {LOG_FILE}")

# 重定向标准输出到日志
class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

if os.environ.get('DISABLE_STDOUT_WRAP'):
    import sys
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.WARNING)

# 导入路径设置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import time

app = FastAPI(
    title="KYC 测试 API",
    description="KYC 身份证和视频认证测试服务",
    version="1.0.0"
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有请求和响应"""
    start_time = time.time()

    # 记录请求
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    url = str(request.url)
    query_params = dict(request.query_params)

    logger.info(f"[请求] {client_ip} - {method} {url} - 参数: {query_params}")

    try:
        response = await call_next(request)

        # 记录响应
        process_time = time.time() - start_time
        status_code = response.status_code

        logger.info(f"[响应] {client_ip} - {method} {url} - 状态码: {status_code} - 耗时: {process_time:.2f}s")

        return response

    except Exception as e:
        # 记录异常
        process_time = time.time() - start_time
        logger.error(f"[异常] {client_ip} - {method} {url} - 错误: {str(e)} - 耗时: {process_time:.2f}s")
        raise

# 导入 KYC 测试模块
import kyc_full_test
import liveportrait_server

# ========== LivePortrait 模型预加载 ==========


@app.on_event("startup")
async def startup_event():
    """应用启动时预加载 LivePortrait 模型"""
    try:
        logger.info("开始预加载 LivePortrait 模型...")
        service = liveportrait_server.get_service()
        kyc_full_test.set_liveportrait_service(service)
        logger.info("LivePortrait 模型预加载完成")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"LivePortrait 模型预加载失败: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    logger.info("=" * 60)
    logger.info(f"KYC API 服务关闭 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ========== API 端点 ==========


@app.get("/auto_kyc", tags=["自动KYC"])
async def auto_kyc(
    action: str = Query("mouth_open", description="视频动作类型: mouth_open, left_shake, right_shake, nod")
):
    """
    自动KYC完整流程（生成身份证正/反面照片 + 身份证认证 + 人脸采集 + 生成动作视频 + 视频认证）

    **参数:**
    - **action**: 视频动作类型 (mouth_open, left_shake, right_shake, nod)
    """
    try:
        logger.info(f"[Auto KYC] 开始 - action: {action}")

        actions = [action]

        logger.info("[Auto KYC] 生成随机用户...")
        user_id, _, _, _, output_dir = kyc_full_test.generate_random_user_with_avatar()

        logger.info(f"[Auto KYC] 运行 KYC 流程 - user_id: {user_id}, actions: {actions}")
        results = kyc_full_test.run_full_kyc_flow(user_id, output_dir, actions, False)

        status_map = {0: "未完成", 1: "认证中", 2: "已完成"}
        final_status = results.get("final_status")
        final_status_text = status_map.get(final_status, "未知")

        logger.info(f"[Auto KYC] 完成 - user_id: {user_id}, final_status: {final_status}({final_status_text})")

        # 如果 KYC 通过，获取用户详细信息
        if final_status == 2:
            logger.info(f"[Auto KYC] KYC 通过，获取用户详细信息 - user_id: {user_id}")
            client = kyc_full_test.KYCFullTestClient()
            info_result = client.get_user_info(user_id)
            logger.info(f"[Auto KYC] KYC流程后 get_user_info 返回 - user_id: {user_id}, result: {info_result}")

            if info_result and info_result.get("code") == 0:
                info_data = info_result.get("data", {})
                logger.info(f"[Auto KYC] 用户信息获取成功 - user_id: {user_id}")

                # 删除不需要的字段
                info_data.pop("CardFaceFeature", None)
                info_data.pop("VideoFaceFeature", None)
                info_data.pop("LastVideoFaceFeature", None)
                info_data.pop("VideoFace", None)
                info_data.pop("VideoPath", None)
                info_data.pop("VideoFaceFeatureType", None)
                info_data.pop("Operate", None)
                info_data.pop("PassportImage", None)

                # 直接返回用户信息
                return info_data
            else:
                # KYC 通过但获取详细信息失败（网络错误等）
                logger.warning(f"[Auto KYC] KYC 通过但获取用户详细信息失败 - user_id: {user_id}")
                return {
                    "success": True,
                    "user_id": user_id,
                    "final_status": 2,
                    "final_status_text": "已完成",
                    "message": "KYC 认证成功，但获取详细信息失败（可能为网络问题）"
                }

        # KYC 未通过，返回基础信息
        return {
            "success": False,
            "user_id": user_id,
            "final_status": final_status
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Auto KYC] 异常 - user_id: {user_id}, error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
