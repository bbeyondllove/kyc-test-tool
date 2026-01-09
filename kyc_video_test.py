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
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def auto_fix_cuda(python_exe):
    """自动修复 CUDA 环境"""
    print("\n" + "="*60)
    print("检测到 CUDA 问题，尝试自动修复...")
    print("="*60)

    fix_script = Path(__file__).parent / "fix_cuda.py"

    if not fix_script.exists():
        print("❌ 修复脚本不存在")
        return False

    try:
        # 运行修复脚本
        result = subprocess.run(
            [sys.executable, str(fix_script)],
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # 再次检查 CUDA
        test_cmd = [python_exe, "-c", "import torch; exit(0 if torch.cuda.is_available() else 1)"]
        test_result = subprocess.run(test_cmd, capture_output=True, timeout=10)
        return test_result.returncode == 0

    except Exception as e:
        print(f"修复失败: {e}")
        return False


# KYC API 配置
KYC_API_URL = "http://localhost:8080/api/process"

# LivePortrait 路径
LIVEPORTRAIT_DIR = Path(__file__).parent / "LivePortrait"

# 动作对应的 driving video/template
ACTION_DRIVERS = {
    "mouth_open": "d20.mp4",     # 张嘴
    "left_shake": "d3.mp4",      # 左摇头
    "right_shake": "d10.mp4",    # 右摇头
    "nod": "d11.mp4",            # 点头
}

# 检查根目录是否有自定义 driving 视频
CUSTOM_DRIVERS_DIR = Path(__file__).parent
if CUSTOM_DRIVERS_DIR.exists():
    for action, default_driver in ACTION_DRIVERS.items():
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

        # 构建请求参数
        params = {
            "user_id": user_id
        }

        request_data = {
            "api": "collect_face",
            "version": "1.0",
            "params": json.dumps(params)
        }

        print(f"   请求参数: {json.dumps(request_data, ensure_ascii=False)}")

        try:
            with open(avatar_path, 'rb') as f:
                files = {'user_file': (os.path.basename(avatar_path), f, 'image/png')}

                # 使用双重编码格式（与 auto_kyc_test.py 保持一致）
                response = requests.post(
                    self.api_url,
                    data={"request": json.dumps(request_data)},
                    files=files,
                    timeout=30
                )

                result = response.json()
                print(f"   响应状态码: {response.status_code}")
                print(f"   响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
                return result
        except requests.exceptions.ConnectionError:
            return {
                "error": True,
                "message": "无法连接到 KYC 服务器，请确认服务已启动"
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
            dict: 响应结果
        """
        if not os.path.exists(video_path):
            return {
                "error": True,
                "message": f"视频文件不存在: {video_path}"
            }

        # 构建请求参数
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

        print(f"   请求参数: {json.dumps(request_data, ensure_ascii=False)}")
        print(f"   发送请求到: {self.api_url}")

        try:
            with open(video_path, 'rb') as f:
                files = {'user_file': (os.path.basename(video_path), f, 'video/mp4')}

                # 使用双重编码格式（与 auto_kyc_test.py 保持一致）
                headers = {
                    'accept': 'application/json',
                    'X-Content-Encrypted': 'none'
                }
                response = requests.post(
                    self.api_url,
                    data={"request": json.dumps(request_data)},
                    files=files,
                    headers=headers,
                    timeout=60
                )

                result = response.json()
                print(f"   响应状态码: {response.status_code}")
                print(f"   响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
                return result
        except requests.exceptions.ConnectionError:
            return {
                "error": True,
                "message": "无法连接到 KYC 服务器，请确认服务已启动"
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }

    def _print_result(self, action, result):
        """打印结果"""
        print(f"\n{'='*60}")
        if result.get("error"):
            print(f"❌ {action} 验证失败: {result.get('message')}")
        else:
            code = result.get("code", -1)
            msg = result.get("msg", "")
            data = result.get("data", {})

            if code == 0:
                print(f"✅ {action} 验证成功")
                if data.get("distance"):
                    print(f"   相似度: {data['distance']}")
                if data.get("repeate_id"):
                    print(f"   重复用户: {data['repeate_id']}")
                if data.get("action"):
                    print(f"   检测动作: {data['action']}")
            else:
                print(f"❌ {action} 验证失败 (code={code}): {msg}")
        print(f"{'='*60}")

    def run_all_actions_test(self, user_id, video_paths, avatar_path=None, nation="China"):
        """
        执行所有动作的验证测试

        Args:
            user_id: 用户ID
            video_paths: 字典 {action: video_path}
            avatar_path: 头像路径（可选，用于人脸采集）
            nation: 国家类型

        Returns:
            dict: 每个动作的测试结果
        """
        print(f"\n{'#'*60}")
        print(f"# 开始 KYC 视频认证测试 - 用户: {user_id}")
        print(f"{'#'*60}")

        results = {}

        # 先进行人脸采集（如果有头像）
        if avatar_path and os.path.exists(avatar_path):
            print(f"\n{'='*60}")
            print(f"第一步：采集人脸")
            print(f"{'='*60}")
            collect_result = self.collect_face(user_id, avatar_path)
            if collect_result.get("code") == 0:
                print(f"✅ 人脸采集成功")
            else:
                print(f"⚠️ 人脸采集失败 (code={collect_result.get('code')}): {collect_result.get('msg')}")

            time.sleep(1)

        for action, video_path in video_paths.items():
            print(f"\n{'='*60}")
            print(f"测试动作: {action}")
            print(f"视频文件: {video_path}")
            print(f"{'='*60}")

            result = self.verify_video(user_id, video_path, action, nation)
            self._print_result(action, result)
            results[action] = result

            # 间隔
            time.sleep(1)

        # 打印总结
        print(f"\n{'#'*60}")
        print(f"# 视频测试完成 - 用户: {user_id}")
        print(f"{'#'*60}")
        print(f"\n测试结果总结:")
        for action, result in results.items():
            code = result.get("code", -1)
            msg = result.get("msg", "")
            status = "✅ 通过" if code == 0 else "❌ 失败"
            print(f"  {action}: {status}")
            if code != 0:
                print(f"    错误码: {code}, 消息: {msg}")

        # 查询用户认证状态
        self.get_user_status(user_id)

        return results

    def get_user_status(self, user_id):
        """获取用户认证状态"""
        print(f"\n{'='*60}")
        print(f"查询用户状态 - 用户: {user_id}")
        print(f"{'='*60}")

        params = {
            "user_id": user_id
        }

        request_data = {
            "api": "get_user_status",
            "version": "1.0",
            "params": json.dumps(params)
        }

        try:
            import io
            response = requests.post(
                self.api_url,
                files={"file": ("", io.BytesIO(b""), "application/octet-stream")},
                data={"request": json.dumps(request_data)},
                timeout=30
            )
            result = response.json()
            self._print_status_result(result)
            return result
        except Exception as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }

    def _print_status_result(self, result):
        """打印状态结果"""
        if result.get("error"):
            print(f"❌ 状态查询失败: {result.get('message')}")
        else:
            code = result.get("code", -1)
            msg = result.get("msg", "")
            data = result.get("data", {})

            if code == 0:
                status = data.get("status", -1)
                status_map = {
                    0: "未完成KYC认证",
                    1: "认证中",
                    2: "已完成KYC认证"
                }
                status_text = status_map.get(status, f"未知状态({status})")
                print(f"✅ 状态查询成功: {status_text}")
                if data:
                    print(f"   数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
            else:
                print(f"❌ 状态查询失败 (code={code}): {msg}")
                if data:
                    print(f"   数据: {json.dumps(data, ensure_ascii=False, indent=2)}")


def generate_video_with_liveportrait(source_image, driving_source, output_path):
    """
    使用 LivePortrait 生成视频

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

    # 构建 driving video 路径
    # 如果 driving_source 是绝对路径，直接使用；否则使用 LivePortrait assets 目录
    driving_path = Path(driving_source)
    if not driving_path.is_absolute():
        driving_path = LIVEPORTRAIT_DIR / "assets" / "examples" / "driving" / driving_source

    if not driving_path.exists():
        print(f"❌ 驱动视频不存在: {driving_path}")
        return False

    # 转换为绝对路径
    source_image_abs = os.path.abspath(source_image)
    driving_path_abs = str(driving_path)
    output_parent_abs = os.path.abspath(output_path.parent)

    # 使用有 CUDA 支持的 Python
    # 检测可能的 Python 路径
    possible_pythons = [
        r"C:\Python312\python.exe",              # 有 CUDA 支持
        str(LIVEPORTRAIT_DIR / "venv" / "Scripts" / "python.exe"),  # LivePortrait 虚拟环境
        sys.executable,                           # 当前 Python
    ]

    python_exe = None
    cuda_available = False

    print("\n检测 CUDA 支持的 Python...")
    for py in possible_pythons:
        if os.path.exists(py):
            try:
                # 测试 CUDA 支持
                test_cmd = [py, "-c", "import torch; print('CUDA:', torch.cuda.is_available()); exit(0 if torch.cuda.is_available() else 1)"]
                result = subprocess.run(
                    test_cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                print(f"  检查 {py}:")
                if result.stdout:
                    print(f"    {result.stdout.strip()}")
                if result.returncode == 0:
                    python_exe = py
                    cuda_available = True
                    print(f"  ✓ 使用 CUDA Python: {python_exe}")
                    break
            except subprocess.TimeoutExpired:
                print(f"    超时")
            except Exception as e:
                print(f"    错误: {e}")

    if not python_exe:
        # 没有找到 CUDA，但可能有 CPU 可以用
        print("\n未找到 CUDA Python，尝试使用第一个可用的 Python...")
        for py in possible_pythons:
            if os.path.exists(py):
                python_exe = py
                print(f"使用 Python: {python_exe} (无 CUDA，可能很慢或失败)")
                break

    if not python_exe:
        python_exe = sys.executable
        print(f"使用默认 Python: {python_exe} (无 CUDA)")

    cmd = [
        python_exe,
        str(liveportrait_inference),
        "-s", source_image_abs,
        "-d", driving_path_abs,
        "-o", output_parent_abs
    ]

    print(f"\n执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(LIVEPORTRAIT_DIR),
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ, PYTHONPATH=str(LIVEPORTRAIT_DIR))
        )

        if result.returncode != 0:
            print(f"❌ LivePortrait 执行失败 (返回码: {result.returncode})")
            if result.stdout:
                print(f"标准输出:\n{result.stdout}")
            if result.stderr:
                print(f"标准错误:\n{result.stderr}")

            # 检查是否是 CUDA 相关错误，自动修复
            if "CUDA" in result.stderr or "cuda" in result.stderr.lower():
                print("\n💡 检测到 CUDA 问题，尝试自动修复...")
                if auto_fix_cuda(python_exe):
                    print("\n✅ CUDA 修复成功，重新尝试生成视频...")
                    # 重新执行
                    result = subprocess.run(
                        cmd,
                        cwd=str(LIVEPORTRAIT_DIR),
                        capture_output=True,
                        text=True,
                        timeout=120,
                        env=dict(os.environ, PYTHONPATH=str(LIVEPORTRAIT_DIR))
                    )
                    if result.returncode != 0:
                        print(f"❌ 修复后仍然失败")
                        if result.stderr:
                            print(f"标准错误:\n{result.stderr}")
                        return False
                else:
                    print("\n❌ CUDA 修复失败")
                    print("   请手动运行: python fix_cuda.py")
                    return False

        # LivePortrait 会生成两个文件：xxx.mp4 和 xxx_concat.mp4
        # 我们使用 xxx.mp4（只有结果）
        source_name = Path(source_image).stem
        driving_name = Path(driving_source).stem
        expected_output = output_path.parent / f"{source_name}--{driving_name}.mp4"

        if expected_output.exists():
            # 重命名到指定路径
            import shutil
            shutil.move(str(expected_output), str(output_path))
            print(f"✅ 视频已生成: {output_path}")
            return True
        else:
            print(f"❌ 输出文件不存在: {expected_output}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ LivePortrait 执行超时")
        return False
    except Exception as e:
        print(f"❌ 生成视频失败: {e}")
        return False


def extract_avatar_from_idcard(idcard_front_path, output_path):
    """
    从身份证正面图片中提取头像

    Args:
        idcard_front_path: 身份证正面图片路径
        output_path: 输出头像路径

    Returns:
        bool: 是否成功
    """
    try:
        from PIL import Image
        import numpy as np
        import cv2

        # 读取身份证图片
        img = Image.open(idcard_front_path)
        img_array = np.array(img)

        # 身份证上头像的大致位置（根据身份证生成器模板）
        # 头像在 (1500, 690) 位置，大小 500x670
        avatar_x, avatar_y = 1500, 690
        avatar_w, avatar_h = 500, 670

        # 裁剪头像
        avatar = img_array[avatar_y:avatar_y+avatar_h, avatar_x:avatar_x+avatar_w]

        # 保存头像
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
    parser.add_argument('--user-id', type=str, required=True, help='用户ID（与身份证测试使用相同的ID）')
    parser.add_argument('--avatar', type=str, help='头像图片路径（如果不提供，将自动查找）')
    parser.add_argument('--idcard-front', type=str, help='身份证正面图片路径（用于提取头像）')
    parser.add_argument('--output-dir', type=str, default='./kyc_test', help='输出目录（与身份证测试共用）')
    parser.add_argument('--actions', type=str, nargs='+',
                        choices=['mouth_open', 'left_shake', 'right_shake', 'nod', 'all'],
                        default=['mouth_open'],
                        help='要测试的动作（默认: mouth_open）')
    parser.add_argument('--skip-generate', action='store_true',
                        help='跳过视频生成，使用已有的视频文件')

    args = parser.parse_args()

    # 确定要测试的动作
    if 'all' in args.actions:
        actions_to_test = ['mouth_open', 'left_shake', 'right_shake', 'nod']
    else:
        actions_to_test = args.actions

    # 创建输出目录
    output_dir = Path(args.output_dir) / args.user_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KYC 视频认证测试")
    print("=" * 60)
    print(f"用户ID: {args.user_id}")
    print(f"测试动作: {', '.join(actions_to_test)}")
    print(f"输出目录: {output_dir}")
    print(f"API地址: {KYC_API_URL}")
    print("=" * 60)

    # 确定头像路径
    avatar_path = args.avatar

    # 可能的头像目录（新版优先）
    avatar_search_dirs = [
        Path("./kyc_test"),      # 新版 kyc_test.py 输出目录
        Path("./kyc_test_random") # 旧版 kyc_idcard_test.py 输出目录
    ]

    if not avatar_path:
        # 1. 优先尝试从 kyc_test 目录查找已保存的头像（新版）
        for search_dir in avatar_search_dirs:
            possible_avatar = search_dir / args.user_id / "avatar.png"
            if possible_avatar.exists():
                avatar_path = str(possible_avatar)
                print(f"\n找到已保存的头像: {avatar_path}")
                break

        if not avatar_path and args.idcard_front:
            # 2. 从指定身份证提取头像
            avatar_path = str(output_dir / "avatar.png")
            print(f"\n从身份证提取头像...")
            if not extract_avatar_from_idcard(args.idcard_front, avatar_path):
                print("❌ 无法提取头像")
                return
        elif not avatar_path:
            # 3. 尝试从目录查找身份证正面并提取
            for search_dir in avatar_search_dirs:
                possible_idcard = search_dir / args.user_id / "idcard_front.png"
                if possible_idcard.exists():
                    avatar_path = str(output_dir / "avatar.png")
                    print(f"\n从身份证图片提取头像: {possible_idcard}")
                    if not extract_avatar_from_idcard(str(possible_idcard), avatar_path):
                        print("❌ 无法提取头像")
                        return
                    break

            if not avatar_path:
                print(f"\n❌ 请提供头像图片 (--avatar) 或身份证正面图片 (--idcard-front)")
                print(f"   或确保已运行身份证测试: ./kyc_test/{args.user_id}/avatar.png")
                return

    print(f"使用头像: {avatar_path}")

    # 生成或查找视频文件
    video_paths = {}

    for action in actions_to_test:
        video_path = output_dir / f"{action}.mp4"

        if args.skip_generate:
            # 跳过生成，检查文件是否存在
            if video_path.exists():
                video_paths[action] = str(video_path)
                print(f"使用现有视频: {video_path}")
            else:
                print(f"❌ 视频文件不存在: {video_path}")
        else:
            # 使用 LivePortrait 生成视频
            print(f"\n正在生成 {action} 视频...")
            driving_source = ACTION_DRIVERS.get(action)
            if not driving_source:
                print(f"❌ 不支持的动作: {action}")
                continue

            if generate_video_with_liveportrait(avatar_path, driving_source, video_path):
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
