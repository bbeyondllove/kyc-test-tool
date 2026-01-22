"""
LivePortrait 视频生成服务
模型启动时加载，持续提供服务
运行：python liveportrait_server.py
"""
import os
import sys
import io

# 禁用 tyro 输出，避免与 stdout 包装冲突
if os.environ.get('DISABLE_STDOUT_WRAP'):
    import tyro
    try:
        tyro.extras.set_quiet(True)  # 禁用所有输出
        tyro.extras.set_accent_color("")
    except AttributeError:
        pass  # tyro 版本可能不支持这些方法
from pathlib import Path

# 移除 stdout 包装，避免与服务器冲突
# if sys.platform == 'win32':
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 添加 LivePortrait 路径
LIVEPORTRAIT_DIR = Path(__file__).parent / "LivePortrait"
sys.path.insert(0, str(LIVEPORTRAIT_DIR))

os.environ["PYTHONPATH"] = str(LIVEPORTRAIT_DIR)


class LivePortraitService:
    """LivePortrait 视频生成服务（模型预加载）"""

    def __init__(self):
        self.pipeline = None
        self.inference_cfg = None
        self.crop_cfg = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        from src.config.argument_config import ArgumentConfig
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline

        # 默认配置
        self.inference_cfg = InferenceConfig()
        self.crop_cfg = CropConfig()

        # 初始化 pipeline
        self.pipeline = LivePortraitPipeline(
            inference_cfg=self.inference_cfg,
            crop_cfg=self.crop_cfg
        )

    def generate_video(self, source_image_path, driving_video_path, output_path):
        """
        生成视频

        Args:
            source_image_path: 源图片路径（头像）
            driving_video_path: 驱动视频路径
            output_path: 输出视频路径

        Returns:
            bool: 是否成功
        """
        from src.config.argument_config import ArgumentConfig
        import shutil

        if not os.path.exists(source_image_path):
            return False

        if not os.path.exists(driving_video_path):
            return False

        # 创建参数
        args = ArgumentConfig(
            source=source_image_path,
            driving=driving_video_path,
            output_dir=os.path.dirname(output_path),
            flag_write_concat=False,  # KYC流程不需要concat视频
        )

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            self.pipeline.execute(args)

            # 重命名输出文件
            source_name = Path(source_image_path).stem
            driving_name = Path(driving_video_path).stem
            expected_output = Path(output_path).parent / f"{source_name}--{driving_name}.mp4"

            if expected_output.exists():
                shutil.move(str(expected_output), str(output_path))
                return True
            else:
                return False

        except Exception as e:
            import traceback
            print(f"[LivePortrait] 生成视频失败: {str(e)}")
            print(f"[LivePortrait] 源图片: {source_image_path}, 驱动视频: {driving_video_path}, 输出: {output_path}")
            traceback.print_exc()
            return False

# 全局服务实例
_service = None


def get_service():
    """获取服务实例（单例）"""
    global _service
    if _service is None:
        _service = LivePortraitService()
    return _service


# 驱动视频目录
DRIVING_DIR = LIVEPORTRAIT_DIR / "assets" / "examples" / "driving"

# ========== 动作映射配置（统一定义） ==========
# 其他模块应从此处导入，避免多处重复定义
ACTION_DRIVERS = {
    "mouth_open": "mouth_open.mp4",   # 张嘴
    "left_shake": "left_shake.mp4",   # 左摇头
    "right_shake": "right_shake.mp4", # 右摇头
    "nod": "nod.mp4",                 # 点头
}


def generate_video_by_action(avatar_path, action, output_dir):
    """
    根据动作生成视频

    Args:
        avatar_path: 头像路径
        action: 动作类型
        output_dir: 输出目录

    Returns:
        str: 生成的视频路径，失败返回 None
    """
    print(f"[LivePortrait] 开始生成视频: action={action}, avatar={avatar_path}, output_dir={output_dir}")

    driving_filename = ACTION_DRIVERS.get(action)
    if not driving_filename:
        print(f"[LivePortrait] 错误: 不支持的动作 {action}")
        return None

    driving_path = DRIVING_DIR / driving_filename
    if not driving_path.exists():
        print(f"[LivePortrait] 错误: 驱动视频不存在: {driving_path}")
        return None

    output_path = os.path.join(output_dir, f"{action}.mp4")

    service = get_service()
    result = service.generate_video(avatar_path, str(driving_path), output_path)
    if result:
        print(f"[LivePortrait] 生成成功: {output_path}")
        return output_path
    else:
        print(f"[LivePortrait] 生成失败: {output_path}")
        return None


if __name__ == "__main__":
    # 测试代码（禁用，避免 print 风险）
    pass
