"""
KYC 视频认证测试脚本
使用 LivePortrait 生成视频并进行视频验证
支持多种动作：张嘴、左转头、右转头、点头
"""
import requests
import json
import time
import os
import sys
import argparse
import subprocess
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == 'win32' and not os.environ.get('DISABLE_STDOUT_WRAP'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# KYC API 配置
KYC_API_URL = "https://kyc-testnet.chainlessdw20.com/api/process"

# LivePortrait 路径
LIVEPORTRAIT_DIR = Path(__file__).parent / "LivePortrait"

# 从 liveportrait_server 导入统一的动作映射配置
from liveportrait_server import ACTION_DRIVERS

# 检查根目录是否有自定义 driving 视频（覆盖默认配置）
CUSTOM_DRIVERS_DIR = Path(__file__).parent
if CUSTOM_DRIVERS_DIR.exists():
    for action in ACTION_DRIVERS.keys():
        custom_driver = CUSTOM_DRIVERS_DIR / f"{action}.mp4"
        if custom_driver.exists():
            ACTION_DRIVERS[action] = str(custom_driver)
            print(f"使用自定义 driving 视频: {action} -> {custom_driver}")


class KYCVideoTestClient:
    """KYC 视频认证测试客户端"""

    def __init__(self, api_url=KYC_API_URL):
        self.api_url = api_url

    def collect_face(self, user_id, avatar_path):
        """
        采集人脸（用于视频验证前）

        Args:
            user_id: 用户ID
            avatar_path: 头像图片路径

        Returns:
            dict: 响应结果
        """
        print(f"\n{'='*60}")
        print(f"采集人脸 - 用户: {user_id}")
        print(f"{'='*60}")

        if not os.path.exists(avatar_path):
            return {
                "error": True,
                "message": f"头像文件不存在: {avatar_path}"
            }

        params = {"user_id": user_id}
        request_data = {
            "api": "collect_face",
            "version": "1.0",
            "params": json.dumps(params)
        }

        try:
            with open(avatar_path, 'rb') as f:
                files = {'user_file': (os.path.basename(avatar_path), f, 'image/png')}
                response = requests.post(
                    self.api_url,
                    data={"request": json.dumps(request_data)},
                    files=files,
                    timeout=30
                )
                return response.json()
        except requests.exceptions.ConnectionError:
            return {
                "error": True,
                "message": "无法连接到 KYC 服务器"
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }

    def verify_video(self, user_id, video_path, action, nation="China"):
        """
        验证视频

        Args:
            user_id: 用户唯一标识
            video_path: 视频文件路径
            action: 动作类型 (mouth_open, left_shake, right_shake, nod)
            nation: 国家类型

        Returns:
            dict: {'code': int, 'msg': str, 'data': dict}
        """
        print(f"\n验证视频 - 用户: {user_id}, 动作: {action}")

        params = {
            "user_id": user_id,
            "action": action,
            "nation": nation
        }

        request_data = {
            "api": "detection_file",
            "version": "1.0",
            "params": json.dumps(params)
        }

        try:
            with open(video_path, 'rb') as f:
                files = {'user_file': (os.path.basename(video_path), f, 'video/mp4')}
                response = requests.post(
                    self.api_url,
                    data={"request": json.dumps(request_data)},
                    files=files,
                    timeout=30
                )
                result = response.json()
                code = result.get("code", -1)
                msg = result.get("msg", "")
                data = result.get("data", {})
                return {'code': code, 'msg': msg, 'data': data}
        except requests.exceptions.ConnectionError:
            return {'code': -1, 'msg': '无法连接到 KYC 服务器'}
        except Exception as e:
            return {'code': -1, 'msg': f'验证失败: {str(e)}'}

    def get_user_status(self, user_id):
        """获取用户状态"""
        params = {"user_id": user_id}
        request_data = {
            "api": "get_user_status",
            "version": "1.0",
            "params": json.dumps(params)
        }

        try:
            with open("", 'rb') as f:
                files = {"file": ("", io.BytesIO(b""), "application/octet-stream")}
                response = requests.post(
                    self.api_url,
                    data={"request": json.dumps(request_data)},
                    files=files,
                    timeout=30
                )
                return response.json()
        except Exception as e:
            return {'code': -1, 'msg': str(e)}

    def run_all_actions_test(self, user_id, video_paths, avatar_path=None):
        """
        运行所有动作的视频验证测试

        Args:
            user_id: 用户ID
            video_paths: {action: video_path} 字典
            avatar_path: 可选，重新采集人脸

        Returns:
            {action: result} 字典
        """
        if avatar_path:
            # 重新采集人脸
            collect_result = self.collect_face(user_id, avatar_path)
            time.sleep(1)

        results = {}
        for action, video_path in video_paths.items():
            if not os.path.exists(video_path):
                results[action] = {'code': -1, 'msg': f'视频文件不存在: {video_path}'}
                continue

            result = self.verify_video(user_id, video_path, action)
            results[action] = result

            status = "✅ 通过" if result['code'] == 0 else "❌ 失败"
            print(f"  {action}: {status} (code={result['code']}, msg={result['msg']})")

        return results


def generate_video_with_liveportrait(source_image, driving_source, output_path):
    """
    使用 LivePortrait 生成视频（简化版本，直接调用 subprocess）

    Args:
        source_image: 源图片路径（头像）
        driving_source: 驱动视频/模板路径（可以是绝对路径或LivePortrait assets下的相对路径）
        output_path: 输出视频路径

    Returns:
        bool: 是否成功
    """
    liveportrait_inference = LIVEPORTRAIT_DIR / "inference.py"

    if not liveportrait_inference.exists():
        print(f"❌ LivePortrait 不存在: {liveportrait_inference}")
        return False

    driving_path = Path(driving_source)
    if not driving_path.is_absolute():
        driving_path = LIVEPORTRAIT_DIR / "assets" / "examples" / "driving" / driving_source

    if not driving_path.exists():
        print(f"❌ 驱动视频不存在: {driving_path}")
        return False

    source_image_abs = os.path.abspath(source_image)
    driving_path_abs = str(driving_path)
    output_parent_abs = os.path.abspath(output_path.parent)

    # 使用 venv 中的 Python (CUDA 支持)
    python_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = r"C:\Python312\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    cmd = [
        python_exe,
        str(liveportrait_inference),
        "-s", source_image_abs,
        "-d", driving_path_abs,
        "-o", output_parent_abs
    ]

    print(f"执行命令: {' '.join(cmd)}")

    try:
        env = dict(os.environ, PYTHONPATH=str(LIVEPORTRAIT_DIR))
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(
            cmd,
            cwd=str(LIVEPORTRAIT_DIR),
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='ignore',
            env=env
        )

        if result.returncode != 0:
            print(f"❌ LivePortrait执行失败 (返回码: {result.returncode})")
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if 'ERROR' in line.upper() or 'Exception' in line or ('Error' in line and 'Deprecation' not in line and 'SSL' not in line):
                        line = line.strip()
                        if line:
                            print(f"     错误: {line}")
            return False

        # 检查输出文件
        source_name = Path(source_image).stem
        driving_name = Path(driving_source).stem
        expected_output = output_path.parent / f"{source_name}--{driving_name}.mp4"

        if expected_output.exists():
            import shutil
            shutil.move(str(expected_output), str(output_path))
            print(f"✅ 视频已生成: {output_path}")
            return True
        else:
            print(f"❌ 输出文件不存在: {expected_output}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ LivePortrait执行超时")
        return False
    except Exception as e:
        print(f"❌ 生成视频失败: {e}")
        return False


def extract_avatar_from_idcard(idcard_front_path, output_path):
    """
    从身份证正面图片中提取头像
    """
    try:
        from PIL import Image
        import numpy as np
        import cv2

        img = Image.open(idcard_front_path)
        img_array = np.array(img)

        # 身份证上头像的大致位置
        avatar_x, avatar_y = 1500, 690
        avatar_w, avatar_h = 500, 670

        avatar = img_array[avatar_y:avatar_y+avatar_h, avatar_x:avatar_x+avatar_w]
        avatar_img = Image.fromarray(avatar)
        avatar_img.save(output_path)

        print(f"✅ 头像已提取: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 提取头像失败: {e}")
        return False


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='KYC 视频认证测试脚本')
    parser.add_argument('--user_id', type=str, required=True, help='用户ID（与身份证测试使用相同的ID）')
    parser.add_argument('--action', type=str,
                        choices=['mouth_open', 'left_shake', 'right_shake', 'nod', 'all'],
                        default='left_shake',
                        help='动作类型')
    parser.add_argument('--skip-video-generate', action='store_true',
                        help='跳过视频生成（使用现有视频文件）')

    args = parser.parse_args()

    print(f"{'='*60}")
    print("KYC 视频认证测试")
    print(f"{'='*60}")
    print(f"API地址: {KYC_API_URL}")
    print(f"用户ID: {args.user_id}")
    print(f"动作: {args.action}")
    print(f"{'='*60}")

    # 确定要测试的动作
    if args.action == 'all':
        actions_to_test = list(ACTION_DRIVERS.keys())
    else:
        actions_to_test = [args.action]

    # 输出目录
    output_dir = Path("./kyc_test") / args.user_id
    os.makedirs(output_dir, exist_ok=True)

    # 获取头像
    avatar_path = output_dir / "avatar.png"
    if not avatar_path:
        # 尝试从身份证提取头像
        front_path = output_dir / "idcard_front.png"
        if front_path.exists():
            print("\n从身份证中提取头像...")
            extract_avatar_from_idcard(front_path, avatar_path)
        else:
            print(f"\n❌ 未找到头像，请先运行: python kyc_idcard_test.py {args.user_id}")
            return

    # 生成视频文件
    video_paths = {}

    for action in actions_to_test:
        video_path = output_dir / f"{action}.mp4"

        if args.skip_video_generate:
            # 跳过视频生成，检查现有文件
            if os.path.exists(video_path):
                video_paths[action] = str(video_path)
                print(f"{action} 视频已存在，跳过生成")
            else:
                print(f"❌ {action} 视频不存在，无法跳过")
                continue
        else:
            # 使用 LivePortrait 生成视频
            if generate_video_with_liveportrait(avatar_path, ACTION_DRIVERS[action], video_path):
                video_paths[action] = str(video_path)
            else:
                print(f"❌ 生成 {action} 视频失败")

    if not video_paths:
        print("\n❌ 没有可用的视频文件")
        return

    # 执行视频验证测试
    client = KYCVideoTestClient()
    client.run_all_actions_test(args.user_id, video_paths, avatar_path=avatar_path)


if __name__ == "__main__":
    main()
