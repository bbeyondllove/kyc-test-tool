"""
测试 flag_write_concat 参数是否生效
验证不生成 concat 视频
"""
import os
import sys
from pathlib import Path

# 添加 LivePortrait 路径
LIVEPORTRAIT_DIR = Path(__file__).parent / "LivePortrait"
sys.path.insert(0, str(LIVEPORTRAIT_DIR))

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

def test_concat_flag():
    """测试 concat 视频生成控制"""
    
    # 准备测试文件
    avatar_path = "avatars/male/adyqfc.png"
    driving_path = "LivePortrait/assets/examples/driving/mouth_open.mp4"
    output_dir = "test_output"
    
    if not os.path.exists(avatar_path):
        print(f"❌ 头像文件不存在: {avatar_path}")
        return False
        
    if not os.path.exists(driving_path):
        print(f"❌ 驱动视频不存在: {driving_path}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("测试 flag_write_concat=False（不生成 concat 视频）")
    print("=" * 60)
    
    # 初始化配置（强制使用CPU）
    inference_cfg = InferenceConfig()
    inference_cfg.flag_force_cpu = True  # 强制CPU模式
    crop_cfg = CropConfig()
    
    # 创建 pipeline
    pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )
    
    # 创建参数（禁用 concat）
    args = ArgumentConfig(
        source=avatar_path,
        driving=driving_path,
        output_dir=output_dir,
        flag_write_concat=False,  # 关键：不生成 concat 视频
        flag_force_cpu=True,  # 强制CPU模式
    )
    
    print(f"源图片: {avatar_path}")
    print(f"驱动视频: {driving_path}")
    print(f"输出目录: {output_dir}")
    print(f"flag_write_concat: {args.flag_write_concat}")
    print()
    
    # 执行生成
    try:
        print("开始生成视频...")
        wfp, wfp_concat = pipeline.execute(args)
        print(f"✅ 生成完成")
        print(f"主视频: {wfp}")
        print(f"Concat视频: {wfp_concat}")
        print()
        
        # 检查文件
        avatar_name = Path(avatar_path).stem
        driving_name = Path(driving_path).stem
        expected_main = Path(output_dir) / f"{avatar_name}--{driving_name}.mp4"
        expected_concat = Path(output_dir) / f"{avatar_name}--{driving_name}_concat.mp4"
        
        print("文件检查:")
        print(f"  主视频存在: {expected_main.exists()} - {expected_main}")
        print(f"  Concat视频存在: {expected_concat.exists()} - {expected_concat}")
        print()
        
        if expected_main.exists() and not expected_concat.exists():
            print("✅ 测试通过！")
            print("   - 主视频已生成")
            print("   - Concat视频未生成（符合预期）")
            return True
        elif expected_main.exists() and expected_concat.exists():
            print("❌ 测试失败！")
            print("   - 主视频已生成")
            print("   - Concat视频也生成了（不符合预期，flag_write_concat未生效）")
            return False
        else:
            print("❌ 测试失败！")
            print("   - 主视频未生成")
            return False
            
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_concat_flag()
    sys.exit(0 if success else 1)
