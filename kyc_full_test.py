"""
KYC 完整认证流程测试脚本（重写版）
整合身份证认证和视频认证，一条命令完成完整KYC流程

使用方法:
  # 随机模式 - 自动生成新用户并完成完整认证
  python kyc_full_test.py --random --count 1

  # 使用已有用户ID
  python kyc_full_test.py --user-id 1234567890

  # 跳过视频生成，使用已有视频
  python kyc_full_test.py --user-id 1234567890 --skip-video-generate
"""
import requests
import json
import time
import os
import sys
import random
import argparse
import subprocess
from pathlib import Path
import urllib.request
import ssl
import io
import string
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# 修复 Windows 控制台编码问题
if sys.platform == 'win32' and not os.environ.get('DISABLE_STDOUT_WRAP'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# KYC API 配置
KYC_API_URL = "https://kyc-testnet.chainlessdw20.com/api/process"

# 导入身份证生成器模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idcard_generator import id_card_utils, name_utils, region_data, utils

# 资源目录
asserts_dir = os.path.join(utils.get_base_path(), 'asserts')

# LivePortrait 路径
LIVEPORTRAIT_DIR = Path(__file__).parent / "LivePortrait"

# 动作对应的 driving video/template
ACTION_DRIVERS = {
    "mouth_open": "d20.mp4",               # 张嘴
    "left_shake": "left_shake.mp4",        # 左摇头 (LivePortrait assets下)
    "right_shake": "d10.mp4",              # 右摇头
    "nod": "d11.mp4",                      # 点头
}

# ========== 工具函数 ==========

def print_section(title):
    """打印分隔线"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

def print_step(step_num, step_name):
    """打印步骤"""
    print(f"\n{'#'*70}")
    print(f"# 步骤 {step_num}: {step_name}")
    print(f"{'#'*70}")

def resize_image_for_ocr(image_path, max_width=1240):
    img = Image.open(image_path)
    width, height = img.size
    if width > max_width:
        new_width = max_width
        new_height = int(height * max_width / width)
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized.save(image_path)
        print(f"   图片已缩放: {width}x{height} -> {new_width}x{new_height}")
    return image_path

def prepare_test_images(input_path, output_dir, resize_for_ocr=True):
    if not os.path.exists(input_path):
        print(f"源图片不存在: {input_path}")
        return None, None

    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(input_path)
    width, height = img.size

    print(f"准备测试图片: {input_path} ({width}x{height})")

    front = img.crop((0, 0, width, height // 2))
    front_path = os.path.join(output_dir, "idcard_front.png")
    front.save(front_path)
    print(f"  正面已保存: {front_path}")

    back = img.crop((0, height // 2, width, height))
    back_path = os.path.join(output_dir, "idcard_back.png")
    back.save(back_path)
    print(f"  反面已保存: {back_path}")

    if resize_for_ocr:
        resize_image_for_ocr(front_path, max_width=800)
        resize_image_for_ocr(back_path, max_width=800)

    return front_path, back_path

def generate_random_user_data(sex=None):
    """生成随机用户数据"""
    if sex:
        # 如果指定了性别，直接使用 random_name_with_sex 生成对应性别的姓名
        name_info = name_utils.random_name_with_sex(sex)
    else:
        # 未指定性别，随机生成
        name_info = name_utils.random_name()

    year = random.randint(1960, 2005)
    month = random.randint(1, 12)
    day = id_card_utils.random_day(year, month)
    region_info = region_data.random_full_data()

    # 生成身份证号（不指定sex，让校验码自动计算）
    id_card = id_card_utils.random_card_no(
        prefix=region_info["code"],
        year=str(year),
        month=str(month),
        day=str(day)
    )
    user_id = str(random.randint(1000000000, 9999999999))

    start_time = id_card_utils.get_start_time()
    expire_time = id_card_utils.get_expire_time()

    return {
        "user_id": user_id,
        "name": name_info.get("name_full", "测试"),
        "sex": name_info.get("sex_text", "男"),
        "nation": "汉",
        "year": year,
        "month": month,
        "day": day,
        "address": region_info.get("address", ""),
        "id_card": id_card,
        "issuing_authority": region_info.get("issuing_authority", ""),
        "valid_period": f"{start_time}-{expire_time}"
    }

def get_random_avatar(gender=None):
    """
    从本地 avatars 文件夹获取随机头像

    Args:
        gender: 性别 '男' 或 '女'，不指定则随机

    Returns:
        tuple: (PIL.Image 头像图片, 性别 '男'/'女'/None)
    """
    avatars_dir = Path("./avatars")
    if not avatars_dir.exists():
        raise Exception("avatars 文件夹不存在，请先运行 download_avatars.py 下载头像")

    if gender is None:
        if random.random() < 0.5:
            gender = '男'
        else:
            gender = '女'

    if gender == '男':
        folder = avatars_dir / "male"
    elif gender == '女':
        folder = avatars_dir / "female"
    else:
        folder = avatars_dir / "unknown"

    avatar_files = list(folder.glob("*.png"))
    if not avatar_files:
        raise Exception(f"{gender} 文件夹为空，请先运行 download_avatars.py 下载头像")

    avatar_path = random.choice(avatar_files)
    avatar = Image.open(avatar_path)

    return avatar, gender


def change_background(img, img_back, zoom_size, center):
    """抠图并粘贴到背景"""
    img = cv2.resize(img, zoom_size)
    rows, cols, channels = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    diff = [5, 30, 30]
    gb = hsv[0, 0]
    lower_blue = np.array(gb - diff)
    upper_blue = np.array(gb + diff)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    for i in range(rows):
        for j in range(cols):
            if dilate[i, j] == 0:
                img_back[center[0] + i, center[1] + j] = img[i, j]

    return img_back

def generate_idcard_image(user_data, avatar_image, output_path, avatar_output_path=None):
    """生成身份证图片"""
    if avatar_image is None:
        raise Exception("头像图片必须提供")

    print("生成身份证图片...")

    saved_avatar_path = None
    if avatar_output_path:
        try:
            avatar_image.save(avatar_output_path)
            saved_avatar_path = avatar_output_path
            print(f"  头像已保存: {avatar_output_path}")
        except Exception as e:
            print(f"  保存头像失败: {e}")

    empty_image = Image.open(os.path.join(asserts_dir, 'empty.png'))
    name_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/hei.ttf'), 72)
    other_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/hei.ttf'), 64)
    birth_date_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/fzhei.ttf'), 60)
    id_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/ocrb10bt.ttf'), 90)

    draw = ImageDraw.Draw(empty_image)

    draw.text((630, 690), user_data["name"], fill=(0, 0, 0), font=name_font)
    draw.text((630, 840), user_data["sex"], fill=(0, 0, 0), font=other_font)
    draw.text((1030, 840), user_data["nation"], fill=(0, 0, 0), font=other_font)
    draw.text((630, 975), str(user_data["year"]), fill=(0, 0, 0), font=birth_date_font)
    draw.text((950, 975), str(user_data["month"]).zfill(2), fill=(0, 0, 0), font=birth_date_font)
    draw.text((1150, 975), str(user_data["day"]).zfill(2), fill=(0, 0, 0), font=birth_date_font)

    addr_loc_y = 1115
    addr = user_data["address"]
    if addr:
        addr_lines = []
        start = 0
        while start < utils.get_show_len(addr):
            show_txt = utils.get_show_txt(addr, start, start + 22)
            addr_lines.append(show_txt)
            start = start + 22

        for addr_line in addr_lines:
            draw.text((630, addr_loc_y), addr_line, fill=(0, 0, 0), font=other_font)
            addr_loc_y += 100

    draw.text((900, 1475), user_data["id_card"], fill=(0, 0, 0), font=id_font)
    draw.text((1050, 2750), user_data["issuing_authority"], fill=(0, 0, 0), font=other_font)
    draw.text((1050, 2895), user_data["valid_period"], fill=(0, 0, 0), font=other_font)

    # 添加头像
    avatar = cv2.cvtColor(np.asarray(avatar_image), cv2.COLOR_RGBA2BGRA)
    empty_image_cv = cv2.cvtColor(np.asarray(empty_image), cv2.COLOR_RGBA2BGRA)
    empty_image_cv = change_background(avatar, empty_image_cv, (500, 670), (690, 1500))
    empty_image = Image.fromarray(cv2.cvtColor(empty_image_cv, cv2.COLOR_BGRA2RGBA))

    color_path = output_path
    bw_path = output_path.replace('.png', '_bw.png')

    empty_image.save(color_path)
    empty_image.convert('L').save(bw_path)

    print(f"  身份证图片已生成: {color_path}")

    return color_path, bw_path, saved_avatar_path

# ========== KYC API 客户端 ==========

class KYCFullTestClient:
    """KYC 完整测试客户端"""

    def __init__(self, api_url=KYC_API_URL):
        self.api_url = api_url
        self.results = {
            "idcard_front": None,
            "idcard_back": None,
            "collect_face": None,
            "videos": {},
            "final_status": None
        }

    def _make_request(self, api_type, user_id, file_path, file_type, params=None):
        """发送文件请求"""
        if params is None:
            params = {}
        params["user_id"] = user_id

        request_data = {
            "api": api_type,
            "version": "1.0",
            "params": json.dumps(params)
        }

        try:
            with open(file_path, 'rb') as f:
                files = {'user_file': (os.path.basename(file_path), f, file_type)}
                response = requests.post(
                    self.api_url,
                    data={"request": json.dumps(request_data)},
                    files=files,
                    timeout=30
                )
                return response.json()
        except Exception as e:
            return {"error": True, "message": f"请求失败: {str(e)}"}

    def _make_request_no_file(self, api_type, user_id):
        """发送无文件请求"""
        params = {"user_id": user_id}
        request_data = {
            "api": api_type,
            "version": "1.0",
            "params": json.dumps(params)
        }

        try:
            response = requests.post(
                self.api_url,
                files={"file": ("", io.BytesIO(b""), "application/octet-stream")},
                data={"request": json.dumps(request_data)},
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"error": True, "message": f"请求失败: {str(e)}"}

    def verify_idcard_front(self, user_id, front_path):
        """验证身份证正面"""
        print(f"  验证身份证正面...")
        result = self._make_request("verify_idcard_front", user_id, front_path, "image/png", {"nation": "China"})
        code = result.get("code", -1)

        if code == 0:
            print(f"  -> 正面认证成功")
            data = result.get("data", {})
            if data:
                print(f"     身份证号: {data.get('id_card', '')}")
                print(f"     姓名: {data.get('real_name', '')}")
            self.results["idcard_front"] = True
        else:
            print(f"  -> 正面认证失败 (code={code}): {result.get('msg', '')}")
            self.results["idcard_front"] = False

        return result

    def verify_idcard_back(self, user_id, back_path):
        """验证身份证反面"""
        print(f"  验证身份证反面...")
        result = self._make_request("verify_idcard_back", user_id, back_path, "image/png", {"nation": "China"})
        code = result.get("code", -1)

        if code == 0:
            print(f"  -> 反面认证成功")
            self.results["idcard_back"] = True
        else:
            print(f"  -> 反面认证失败 (code={code}): {result.get('msg', '')}")
            self.results["idcard_back"] = False

        return result

    def collect_face(self, user_id, avatar_path):
        """采集人脸"""
        print(f"  采集人脸...")

        if not os.path.exists(avatar_path):
            print(f"  -> 头像文件不存在: {avatar_path}")
            self.results["collect_face"] = False
            return {"error": True, "message": "头像文件不存在"}

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
                result = response.json()
        except Exception as e:
            result = {"error": True, "message": f"请求失败: {str(e)}"}

        code = result.get("code", -1)
        if code == 0:
            print(f"  -> 人脸采集成功")
            self.results["collect_face"] = True
        else:
            print(f"  -> 人脸采集失败 (code={code}): {result.get('msg', '')}")
            self.results["collect_face"] = False

        return result

    def verify_video(self, user_id, video_path, action):
        """验证视频"""
        print(f"  验证视频 {action}...")

        if not os.path.exists(video_path):
            print(f"  -> 视频文件不存在: {video_path}")
            self.results["videos"][action] = False
            return {"error": True, "message": "视频文件不存在"}

        params = {"action": action, "nation": "China"}
        result = self._make_request("detection_file", user_id, video_path, "video/mp4", params)
        code = result.get("code", -1)

        if code == 0:
            print(f"  -> {action} 验证成功")
            data = result.get("data", {})
            if data.get("distance"):
                print(f"     相似度: {data['distance']}")
            if data.get("action"):
                print(f"     检测动作: {data['action']}")
            self.results["videos"][action] = True
        else:
            print(f"  -> {action} 验证失败 (code={code}): {result.get('msg', '')}")
            if result.get('data'):
                print(f"     数据: {json.dumps(result.get('data'), ensure_ascii=False, indent=2)}")
            if result.get('error'):
                print(f"     错误: {result.get('error')} - {result.get('message', '')}")
            self.results["videos"][action] = False

        return result

    def get_user_status(self, user_id):
        """获取用户状态"""
        print(f"  查询用户状态...")
        result = self._make_request_no_file("get_user_status", user_id)

        if not result.get("error"):
            code = result.get("code", -1)
            data = result.get("data", {})
            if code == 0:
                status = data.get("status", -1)
                status_map = {
                    0: "未完成KYC认证",
                    1: "认证中",
                    2: "已完成KYC认证"
                }
                status_text = status_map.get(status, f"未知状态({status})")
                print(f"  -> 状态: {status_text}")
                self.results["final_status"] = status
            else:
                print(f"  -> 状态查询失败 (code={code}): {result.get('msg', '')}")
        else:
            print(f"  -> 状态查询失败: {result.get('message', '')}")

        return result

    def get_user_info(self, user_id):
        """获取用户信息"""
        result = self._make_request_no_file("get_user_info", user_id)
        return result

# ========== 视频生成 ==========

# 全局 LivePortrait 服务实例
_liveportrait_service = None


def set_liveportrait_service(service):
    """设置 LivePortrait 服务实例"""
    global _liveportrait_service
    _liveportrait_service = service


def get_liveportrait_service():
    """获取 LivePortrait 服务实例"""
    return _liveportrait_service


def generate_video_with_liveportrait(source_image, driving_source, output_path):
    """使用LivePortrait生成视频"""
    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    print(f"  生成视频: {Path(driving_source).stem} -> {output_path.name}")

    # 如果有预加载的服务，使用它（API 模式）
    if _liveportrait_service is not None:
        # 从 driving_source 提取 action 名称
        driving_name = Path(driving_source).stem
        # 反查 action
        action = None
        for a, d in ACTION_DRIVERS.items():
            if driving_name == Path(d).stem:
                action = a
                break

        if action:
            import liveportrait_server
            output_dir = os.path.dirname(output_path)
            result_path = liveportrait_server.generate_video_by_action(source_image, action, output_dir)
            if result_path and os.path.exists(result_path):
                # 移动到目标路径
                if result_path != output_path:
                    import shutil
                    shutil.move(result_path, output_path)
                print(f"  -> 视频生成成功: {output_path.name}")
                return True
            return False

    # 原来的 subprocess 方式（命令行模式）
    liveportrait_inference = LIVEPORTRAIT_DIR / "inference.py"

    if not liveportrait_inference.exists():
        print(f"  -> LivePortrait不存在: {liveportrait_inference}")
        return False

    driving_path = Path(driving_source)
    if not driving_path.is_absolute():
        driving_path = LIVEPORTRAIT_DIR / "assets" / "examples" / "driving" / driving_source

    if not driving_path.exists():
        print(f"  -> 驱动视频不存在: {driving_path}")
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
            print(f"  -> LivePortrait执行失败 (返回码: {result.returncode})")
            if result.stderr:
                # 显示关键错误，过滤警告
                for line in result.stderr.split('\n'):
                    if 'ERROR' in line.upper() or 'Exception' in line or ('Error' in line and 'Deprecation' not in line and 'SSL' not in line):
                        print(f"     错误: {line}")
            return False

        source_name = Path(source_image).stem
        driving_name = Path(driving_source).stem
        expected_output = output_path.parent / f"{source_name}--{driving_name}.mp4"

        if expected_output.exists():
            import shutil
            shutil.move(str(expected_output), str(output_path))
            print(f"  -> 视频生成成功: {output_path.name}")
            return True
        else:
            print(f"  -> 输出文件不存在: {expected_output}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  -> LivePortrait执行超时")
        return False
    except Exception as e:
        print(f"  -> 生成视频失败: {e}")
        return False

# ========== 完整流程 ==========

def run_full_kyc_flow(user_id, output_dir, actions_to_test, skip_video_generate=False):
    """运行完整 KYC 流程（身份证认证失败则停止）"""
    client = KYCFullTestClient()

    # ========== 第一阶段: 身份证认证 ==========
    print_step(1, "身份证认证")

    front_path = os.path.join(output_dir, "idcard_front.png")
    back_path = os.path.join(output_dir, "idcard_back.png")

    if not os.path.exists(front_path):
        print(f"错误: 身份证正面不存在: {front_path}")
        return client.results

    client.verify_idcard_front(user_id, front_path)

    if not client.results["idcard_front"]:
        print("\n❌ 身份证认证失败，终止流程，不进行动作认证")
        print_section("测试结果")
        print(f"用户ID: {user_id}")
        print(f"身份证正面: ❌ 失败")
        return client.results

    time.sleep(1)

    if not os.path.exists(back_path):
        print(f"错误: 身份证反面不存在: {back_path}")
        return client.results

    client.verify_idcard_back(user_id, back_path)

    if not client.results["idcard_back"]:
        print("\n❌ 身份证反面认证失败，终止流程，不进行动作认证")
        print_section("测试结果")
        print(f"用户ID: {user_id}")
        print(f"身份证正面: ✅ 通过")
        print(f"身份证反面: ❌ 失败")
        return client.results

    time.sleep(1)
    client.get_user_status(user_id)

    # ========== 第二阶段: 视频认证 ==========
    print_step(2, "视频认证")

    avatar_path = os.path.join(output_dir, "avatar.png")

    # 人脸采集
    if not os.path.exists(avatar_path):
        print(f"警告: 头像不存在: {avatar_path}")
        print("      将尝试继续视频认证，但可能失败")
    else:
        client.collect_face(user_id, avatar_path)
        time.sleep(1)

    # 生成/使用视频
    video_paths = {}
    for action in actions_to_test:
        video_path = os.path.join(output_dir, f"{action}.mp4")

        if skip_video_generate:
            if os.path.exists(video_path):
                video_paths[action] = video_path
                print(f"  使用现有视频: {video_path}")
            else:
                print(f"  视频文件不存在: {video_path}")
        else:
            driving_source = ACTION_DRIVERS.get(action)
            if driving_source:
                if generate_video_with_liveportrait(avatar_path, driving_source, video_path):
                    video_paths[action] = video_path
                else:
                    print(f"  生成 {action} 视频失败")
            else:
                print(f"  不支持的动作: {action}")

    # 验证视频
    if video_paths:
        print(f"\n开始验证视频...")
        for action, video_path in video_paths.items():
            client.verify_video(user_id, video_path, action)
            time.sleep(1)
    else:
        print("\n❌ 没有可用的视频文件")

    # ========== 第三阶段: 最终状态查询 ==========
    print_step(3, "最终状态查询")

    client.get_user_status(user_id)

    # ========== 结果汇总 ==========
    print_section("测试结果汇总")

    print(f"用户ID: {user_id}")
    print(f"身份证正面: {'✅ 通过' if client.results['idcard_front'] else '❌ 失败'}")
    print(f"身份证反面: {'✅ 通过' if client.results['idcard_back'] else '❌ 失败'}")
    print(f"人脸采集: {'✅ 通过' if client.results['collect_face'] else '❌ 失败'}")

    if client.results['videos']:
        print("视频验证:")
        for action, passed in client.results['videos'].items():
            print(f"  {action}: {'✅ 通过' if passed else '❌ 失败'}")

    status_map = {0: "未完成", 1: "认证中", 2: "已完成"}
    print(f"最终状态: {status_map.get(client.results['final_status'], '未知')}")

    return client.results

# ========== 主程序 ==========

def main():
    parser = argparse.ArgumentParser(description='KYC 完整认证流程测试脚本')
    parser.add_argument('--user_id', type=str, default=None, help='用户ID（可选，不指定则自动生成随机用户）')
    parser.add_argument('--action', type=str,
                        choices=['mouth_open', 'left_shake', 'right_shake', 'nod', 'all'],
                        default='left_shake',
                        help='要测试的视频动作（默认: left_shake）')

    args = parser.parse_args()

    # 确定要测试的动作
    if args.action == 'left_shake':
        actions_to_test = ['left_shake']
    elif args.action == 'all':
        actions_to_test = ['mouth_open', 'left_shake', 'right_shake', 'nod']
    else:
        actions_to_test = [args.action]

    print_section("KYC 完整认证流程测试")
    print(f"API地址: {KYC_API_URL}")
    print(f"测试动作: {', '.join(actions_to_test)}")

    # ========== 随机模式 ==========
    if not args.user_id:
        print("随机模式：生成新用户...")

        try:
            # 1. 生成用户数据（包含随机性别）
            print_step(1, "生成用户数据")
            user_data = generate_random_user_data()
            user_id = user_data['user_id']
            user_gender = user_data['sex']

            print(f"  用户ID: {user_id}")
            print(f"  姓名: {user_data['name']}")
            print(f"  性别: {user_gender}")
            print(f"  身份证号: {user_data['id_card']}")

            # 2. 根据性别获取头像
            print_step(2, "获取头像")
            avatar_image, detected_gender = get_random_avatar(gender=user_gender)

            if avatar_image is None:
                print("无法获取头像，测试终止")
                return

            # 3. 创建输出目录
            output_dir = os.path.join("./kyc_test", user_id)
            os.makedirs(output_dir, exist_ok=True)

            # 4. 生成身份证
            print_step(3, "生成身份证")
            avatar_path = os.path.join(output_dir, "avatar.png")
            temp_idcard_path = os.path.join(output_dir, f"temp_idcard.png")

            color_path, bw_path, saved_avatar_path = generate_idcard_image(
                user_data, avatar_image, temp_idcard_path, avatar_output_path=avatar_path
            )

            # 5. 裁剪正反面
            prepare_test_images(color_path, output_dir)

            # 6. 删除临时文件
            for f in [color_path, bw_path]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except:
                    pass

            # 7. 运行完整流程
            print_step(4, "运行KYC认证流程")
            run_full_kyc_flow(user_id, output_dir, actions_to_test, False)

        except Exception as e:
            print(f"\n❌ 测试失败: {e}")

    # ========== 使用模式 ==========
    else:
        user_id = args.user_id
        output_dir = os.path.join("./kyc_test", user_id)
        os.makedirs(output_dir, exist_ok=True)

        print(f"使用已有用户: {user_id}")

        # 检查身份证文件
        front_path = os.path.join(output_dir, "idcard_front.png")
        back_path = os.path.join(output_dir, "idcard_back.png")

        if not os.path.exists(front_path) or not os.path.exists(back_path):
            print(f"❌ 未找到身份证文件，请先运行: python kyc_idcard_test.py {user_id}")
            return

        # 运行完整流程
        run_full_kyc_flow(user_id, output_dir, actions_to_test, False)


# ========== 辅助函数（用于 API 调用） ==========


def generate_random_user_with_avatar():
    """
    生成随机用户并获取头像（简化版，用于 API）

    Returns:
        tuple: (user_id, front_image, back_image, avatar_path, output_dir)
    """
    # 先生成用户数据（包含随机性别）
    user_data = generate_random_user_data()
    user_id = user_data['user_id']
    user_gender = user_data['sex']

    # 根据性别获取头像
    avatar_image, detected_gender = get_random_avatar(gender=user_gender)
    if avatar_image is None:
        raise Exception("无法获取头像")

    # 创建输出目录
    output_dir = os.path.join("./kyc_test", user_id)
    os.makedirs(output_dir, exist_ok=True)

    # 生成身份证
    avatar_path = os.path.join(output_dir, "avatar.png")
    temp_color_path = os.path.join(output_dir, f"temp_idcard.png")

    color_path, bw_path, saved_avatar_path = generate_idcard_image(
        user_data,
        avatar_image=avatar_image,
        output_path=temp_color_path,
        avatar_output_path=avatar_path
    )

    # 裁剪正反面
    front_image, back_image = prepare_test_images(color_path, output_dir)

    # 删除临时文件
    for f in [color_path, bw_path]:
        try:
            if os.path.exists(f):
                os.remove(f)
        except:
            pass

    return user_id, front_image, back_image, avatar_path, output_dir


if __name__ == "__main__":
    main()
