"""
KYC 测试 API 服务
使用现有功能模块
运行：uvicorn kyc_api_standalone:app --host 0.0.0.0 --port 8003
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

# 重定向标准输出，避免 emoji 编码问题
class NullWriter:
    def write(self, *args, **kwargs):
        pass
    def flush(self, *args, **kwargs):
        pass

if os.environ.get('DISABLE_STDOUT_WRAP'):
    import sys
    sys.stdout = NullWriter()
    sys.stderr._original_stderr = sys.stderr.__class__

# 导入路径设置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import json
from enum import Enum

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
import kyc_idcard_test
import kyc_video_test
import kyc_full_test
import liveportrait_server

idcard_client = kyc_idcard_test.KYCTestClient()
video_client = kyc_video_test.KYCVideoTestClient()

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


class ActionEnum(str, Enum):
    """支持的认证动作"""
    mouth_open = "mouth_open"
    left_shake = "left_shake"
    right_shake = "right_shake"
    nod = "nod"
    all = "all"


# ========== API 端点 ==========


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "KYC 测试 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/idcard/verify", tags=["身份证认证"])
async def verify_id_card(user_id: str = Query(None, description="用户ID，不指定则自动生成随机用户")):
    """
    身份证认证接口

    **参数:**
    - **user_id**: 可选，指定用户ID，不指定则自动生成随机用户
    """
    try:
        logger.info(f"[身份证认证] 开始 - user_id: {user_id}")

        if user_id:
            output_dir = os.path.join("./kyc_test", user_id)
            front_path = os.path.join(output_dir, "idcard_front.png")
            back_path = os.path.join(output_dir, "idcard_back.png")

            if not os.path.exists(front_path) or not os.path.exists(back_path):
                logger.warning(f"[身份证认证] 用户身份证文件不存在 - user_id: {user_id}")
                raise HTTPException(status_code=404, detail="未找到用户身份证文件")

            success = idcard_client.run_full_idcard_test(user_id, front_path, back_path)
        else:
            # 使用 kyc_full_test 生成随机用户并测试
            logger.info("[身份证认证] 生成随机用户...")
            user_id, front_image, back_image, _, _ = kyc_full_test.generate_random_user_with_avatar()
            success = idcard_client.run_full_idcard_test(user_id, front_image, back_image)

        logger.info(f"[身份证认证] 完成 - user_id: {user_id}, success: {success}")
        return {
            "success": success,
            "user_id": user_id,
            "message": "认证成功" if success else "认证失败"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[身份证认证] 异常 - user_id: {user_id}, error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/video/verify", tags=["视频认证"])
async def verify_video(
    user_id: str = Query(..., description="用户ID"),
    action: ActionEnum = Query(ActionEnum.left_shake, description="动作类型")
):
    """
    视频认证接口

    **参数:**
    - **user_id**: 用户ID（必填）
    - **action**: 动作类型 (mouth_open, left_shake, right_shake, nod, all)
    """
    try:
        logger.info(f"[视频认证] 开始 - user_id: {user_id}, action: {action}")

        output_dir = os.path.join("./kyc_test", user_id)
        os.makedirs(output_dir, exist_ok=True)

        avatar_path = os.path.join(output_dir, "avatar.png")
        if not os.path.exists(avatar_path):
            logger.warning(f"[视频认证] 用户头像不存在 - user_id: {user_id}")
            raise HTTPException(status_code=404, detail="未找到用户头像，请先进行身份证认证")

        # 确定动作
        if action == ActionEnum.all:
            actions = list(kyc_full_test.ACTION_DRIVERS.keys())
        else:
            actions = [action.value]

        logger.info(f"[视频认证] 动作列表: {actions}")

        # 人脸采集
        collect_result = video_client.collect_face(user_id, avatar_path)
        time.sleep(1)

        # 生成视频（如果需要）
        video_paths = {}
        for act in actions:
            video_path = os.path.join(output_dir, f"{act}.mp4")
            # 使用 kyc_full_test 生成视频
            driving_video = kyc_full_test.ACTION_DRIVERS.get(act)
            if driving_video:
                driving_path = os.path.join(kyc_full_test.LIVEPORTRAIT_DIR, "assets", "examples", "driving", driving_video)
                if os.path.exists(driving_path):
                    kyc_full_test.generate_video_with_liveportrait(avatar_path, driving_path, video_path)
            if os.path.exists(video_path):
                video_paths[act] = video_path

        if not video_paths:
            logger.error(f"[视频认证] 视频生成失败 - user_id: {user_id}")
            raise HTTPException(status_code=500, detail="视频生成失败")

        # 验证视频
        results = video_client.run_all_actions_test(user_id, video_paths, avatar_path=avatar_path)

        success = any(r.get("code") == 0 for r in results.values())
        logger.info(f"[视频认证] 完成 - user_id: {user_id}, success: {success}, results: {results}")

        return {
            "success": success,
            "user_id": user_id,
            "videos": {k: v.get("code") == 0 for k, v in results.items()},
            "message": "认证成功" if success else "认证失败"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[视频认证] 异常 - user_id: {user_id}, error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/full/test/sync", tags=["完整流程"])
async def full_test_sync(
    user_id: str = Query(None, description="用户ID，不指定则自动生成随机用户"),
    action: ActionEnum = Query(ActionEnum.left_shake, description="视频动作类型")
):
    """
    同步完整流程测试（等待完成）

    **参数:**
    - **user_id**: 可选，指定用户ID
    - **action**: 视频动作类型
    """
    try:
        logger.info(f"[完整流程] 开始 - user_id: {user_id}, action: {action}")

        actions = [action.value] if action != ActionEnum.all else list(kyc_full_test.ACTION_DRIVERS.keys())

        if not user_id:
            logger.info("[完整流程] 生成随机用户...")
            user_id, _, _, _, output_dir = kyc_full_test.generate_random_user_with_avatar()
        else:
            output_dir = os.path.join("./kyc_test", user_id)
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f"[完整流程] 运行 KYC 流程 - user_id: {user_id}, actions: {actions}")
        results = kyc_full_test.run_full_kyc_flow(user_id, output_dir, actions, False)

        status_map = {0: "未完成", 1: "认证中", 2: "已完成"}
        final_status = results.get("final_status")
        final_status_text = status_map.get(final_status, "未知")

        logger.info(f"[完整流程] 完成 - user_id: {user_id}, final_status: {final_status}({final_status_text})")
        return {
            "success": results.get("final_status") == 2,
            "user_id": user_id,
            "idcard_front": results.get("idcard_front", False),
            "idcard_back": results.get("idcard_back", False),
            "collect_face": results.get("collect_face", False),
            "videos": results.get("videos", {}),
            "final_status": results.get("final_status"),
            "final_status_text": final_status_text,
            "message": "测试完成" if results.get("final_status") == 2 else "未完成"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[完整流程] 异常 - user_id: {user_id}, error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
