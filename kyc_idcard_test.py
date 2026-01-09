"""
KYC 证件认证测试脚本
测试身份证正面、反面认证功能
支持随机生成测试数据
"""
import requests
import json
import time
import os
import sys
import random
import datetime
import calendar
import urllib.request
import ssl
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# 修复 Windows 控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# KYC API 配置
KYC_API_URL = "http://localhost:8080/api/process"

# 导入身份证生成器模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idcard_generator import id_card_utils, name_utils, region_data, utils

# 资源目录
asserts_dir = os.path.join(utils.get_base_path(), 'asserts')


def resize_image_for_ocr(image_path, max_width=1240):
    """
    调整图片大小以加速OCR处理

    Args:
        image_path: 输入图片路径
        max_width: 最大宽度（高度按比例缩放）

    Returns:
        str: 调整后的图片路径（覆盖原文件）
    """
    img = Image.open(image_path)
    width, height = img.size

    if width > max_width:
        # 计算新的尺寸
        new_width = max_width
        new_height = int(height * max_width / width)

        # 调整大小
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized.save(image_path)
        print(f"   📐 图片已缩放: {width}x{height} -> {new_width}x{new_height}")

    return image_path


def prepare_test_images(input_path="color.png", output_dir="./kyc_test", resize_for_ocr=True):
    """
    准备测试图片：从身份证生成器输出的合成图中裁剪出正面和反面

    Args:
        input_path: 身份证生成器输出的彩色图片路径
        output_dir: 测试图片输出目录
        resize_for_ocr: 是否缩放图片以加速OCR处理

    Returns:
        tuple: (正面图片路径, 反面图片路径)
    """
    # 检查源图片是否存在
    if not os.path.exists(input_path):
        print(f"⚠️ 源图片不存在: {input_path}")
        print("请先运行身份证生成器生成彩色身份证图片")
        return None, None

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开并裁剪图片
    img = Image.open(input_path)
    width, height = img.size

    print(f"📁 准备测试图片...")
    print(f"   源图片: {input_path} ({width}x{height})")

    # 裁剪位置 (上半部分是正面，下半部分是反面)
    front_y_start = 0
    front_y_end = height // 2
    back_y_start = height // 2
    back_y_end = height

    # 裁剪正面
    front = img.crop((0, front_y_start, width, front_y_end))
    front_path = os.path.join(output_dir, "idcard_front.png")
    front.save(front_path)
    print(f"   ✅ 正面已保存: {front_path}")

    # 裁剪反面
    back = img.crop((0, back_y_start, width, back_y_end))
    back_path = os.path.join(output_dir, "idcard_back.png")
    back.save(back_path)
    print(f"   ✅ 反面已保存: {back_path}")

    # 调整图片大小以加速OCR
    if resize_for_ocr:
        print(f"   📐 缩放图片以加速OCR处理...")
        resize_image_for_ocr(front_path, max_width=800)
        resize_image_for_ocr(back_path, max_width=800)

    return front_path, back_path


def generate_random_user_data(sex=None):
    """
    生成随机的用户身份信息

    Args:
        sex: 指定性别 ('男' 或 '女')，如果为 None 则随机生成

    Returns:
        dict: 包含用户ID、姓名、性别、民族、出生日期、住址、身份证号、签发机关、有效期限
    """
    # 生成姓名（根据性别）
    name_info = name_utils.random_name()

    # 如果指定了性别，重新生成直到匹配
    if sex:
        target_sex_code = 0 if sex == '女' else 1
        max_attempts = 10

        for _ in range(max_attempts):
            name_info = name_utils.random_name()
            if name_info['sex'] == target_sex_code:
                break

        # 如果重试后仍不匹配，手动修正性别文本
        if name_info['sex'] != target_sex_code:
            name_info['sex'] = target_sex_code
            name_info['sex_text'] = sex

    # 随机出生日期
    year = random.randint(1960, 2005)
    month = random.randint(1, 12)
    day = id_card_utils.random_day(year, month)

    # 随机省份地址和签发机关
    region_info = region_data.random_full_data()

    # 使用正确的地区代码生成身份证号
    id_card = id_card_utils.random_card_no(
        prefix=region_info["code"],
        year=str(year),
        month=str(month),
        day=str(day)
    )

    # 生成随机用户ID (10位数字)
    user_id = str(random.randint(1000000000, 9999999999))

    # 随机有效期限
    start_time = id_card_utils.get_start_time()
    expire_time = id_card_utils.get_expire_time()

    return {
        "user_id": user_id,
        "name": name_info["name_full"],
        "sex": name_info["sex_text"],
        "nation": "汉",
        "year": year,
        "month": month,
        "day": day,
        "address": region_info["address"],
        "id_card": id_card,
        "issuing_authority": region_info["issuing_authority"],
        "valid_period": f"{start_time}-{expire_time}"
    }


def download_random_avatar():
    """
    从 thispersondoesnotexist.com 下载随机 AI 生成的头像并检测性别

    Returns:
        tuple: (PIL.Image 头像图片, 性别 '男'/'女'/None)
    """
    # 尝试导入 DeepFace 进行性别检测
    try:
        from deepface import DeepFace
        deepface_available = True
    except ImportError:
        deepface_available = False
        print("警告: DeepFace 未安装，将使用随机性别")
        print("      安装: pip install deepface")

    try:
        url = "https://thispersondoesnotexist.com"
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(request, context=context, timeout=30) as response:
            image_data = response.read()

        avatar = Image.open(io.BytesIO(image_data))

        # 检测性别
        detected_gender = None
        if deepface_available:
            try:
                img_array = np.array(avatar.convert('RGB'))
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                result = DeepFace.analyze(img_bgr, actions=['gender'], enforce_detection=False)

                if isinstance(result, list):
                    result = result[0]

                dominant_gender = result.get('dominant_gender', None)
                if dominant_gender == 'Woman':
                    detected_gender = '女'
                elif dominant_gender == 'Man':
                    detected_gender = '男'

                if detected_gender:
                    print(f"   检测到头像性别: {detected_gender}")

            except Exception as e:
                print(f"   性别检测失败: {e}")
        else:
            # 没有安装 DeepFace，随机生成一个性别
            detected_gender = random.choice(['男', '女'])
            print(f"   使用随机性别: {detected_gender}")

        return avatar, detected_gender

    except Exception as e:
        print(f"下载头像失败: {e}")
        return None, None


def get_local_avatar(target_gender=None, avatar_dir="./avatars"):
    """
    从本地目录随机选择一个头像

    Args:
        target_gender: 目标性别 ('男' 或 '女')，如果为 None 则从主目录选择
        avatar_dir: 头像目录路径

    Returns:
        PIL.Image: 头像图片，失败返回 None
    """
    # 如果指定了性别，尝试从性别子目录选择
    if target_gender:
        gender_dir = os.path.join(avatar_dir, 'male' if target_gender == '男' else 'female')
        if os.path.exists(gender_dir):
            avatar_files = [f for f in os.listdir(gender_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            if avatar_files:
                avatar_path = os.path.join(gender_dir, random.choice(avatar_files))
                try:
                    return Image.open(avatar_path)
                except Exception as e:
                    print(f"加载头像失败: {e}")

    # 从主目录选择
    if not os.path.exists(avatar_dir):
        return None

    avatar_files = [f for f in os.listdir(avatar_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    if not avatar_files:
        return None

    avatar_path = os.path.join(avatar_dir, random.choice(avatar_files))
    try:
        return Image.open(avatar_path)
    except Exception as e:
        print(f"加载头像失败: {e}")
        return None


def generate_idcard_image(user_data, avatar_image, output_path="color.png", auto_bg=True, avatar_output_path=None):
    """
    自动生成身份证图片（无GUI）

    Args:
        user_data: 用户数据字典（包含 sex 字段）
        avatar_image: 头像 PIL.Image 对象（必须提供）
        output_path: 输出图片路径
        auto_bg: 是否自动抠图
        avatar_output_path: 头像保存路径，如果为 None 则不保存

    Returns:
        tuple: (彩色图片路径, 黑白图片路径, 头像路径(如果保存))
    """
    if avatar_image is None:
        raise Exception("头像图片必须提供")

    print(f"正在生成身份证图片...")

    # 保存原始头像（用于视频生成）
    saved_avatar_path = None
    if avatar_output_path:
        try:
            avatar_copy = avatar_image.copy()
            avatar_copy.save(avatar_output_path)
            saved_avatar_path = avatar_output_path
            print(f"头像已保存: {avatar_output_path}")
        except Exception as e:
            print(f"保存头像失败: {e}")

    # 加载空白身份证模板
    empty_image = Image.open(os.path.join(asserts_dir, 'empty.png'))

    # 加载字体
    name_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/hei.ttf'), 72)
    other_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/hei.ttf'), 64)
    birth_date_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/fzhei.ttf'), 60)
    id_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/ocrb10bt.ttf'), 90)

    # 绘制信息
    draw = ImageDraw.Draw(empty_image)

    # 姓名
    draw.text((630, 690), user_data["name"], fill=(0, 0, 0), font=name_font)

    # 性别
    draw.text((630, 840), user_data["sex"], fill=(0, 0, 0), font=other_font)

    # 民族
    draw.text((1030, 840), user_data["nation"], fill=(0, 0, 0), font=other_font)

    # 出生日期
    draw.text((630, 975), str(user_data["year"]), fill=(0, 0, 0), font=birth_date_font)
    draw.text((950, 975), str(user_data["month"]).zfill(2), fill=(0, 0, 0), font=birth_date_font)
    draw.text((1150, 975), str(user_data["day"]).zfill(2), fill=(0, 0, 0), font=birth_date_font)

    # 住址（分多行显示）
    addr_loc_y = 1115
    addr = user_data["address"]
    addr_lines = []
    start = 0
    while start < utils.get_show_len(addr):
        show_txt = utils.get_show_txt(addr, start, start + 22)
        addr_lines.append(show_txt)
        start = start + 22

    for addr_line in addr_lines:
        draw.text((630, addr_loc_y), addr_line, fill=(0, 0, 0), font=other_font)
        addr_loc_y += 100

    # 身份证号
    draw.text((900, 1475), user_data["id_card"], fill=(0, 0, 0), font=id_font)

    # 背面信息
    draw.text((1050, 2750), user_data["issuing_authority"], fill=(0, 0, 0), font=other_font)
    draw.text((1050, 2895), user_data["valid_period"], fill=(0, 0, 0), font=other_font)

    # 添加头像
    if auto_bg:
        # 使用抠图方式
        avatar = cv2.cvtColor(np.asarray(avatar_image), cv2.COLOR_RGBA2BGRA)
        empty_image_cv = cv2.cvtColor(np.asarray(empty_image), cv2.COLOR_RGBA2BGRA)

        # 调用抠图函数
        empty_image_cv = change_background(avatar, empty_image_cv, (500, 670), (690, 1500))
        empty_image = Image.fromarray(cv2.cvtColor(empty_image_cv, cv2.COLOR_BGRA2RGBA))
    else:
        # 直接粘贴
        avatar = avatar_image.resize((500, 670))
        avatar = avatar.convert('RGBA')
        empty_image.paste(avatar, (1500, 690), mask=avatar)

    # 保存彩色和黑白图片
    color_path = output_path
    bw_path = output_path.replace('.png', '_bw.png').replace('color', 'bw')
    if bw_path == color_path:
        bw_path = output_path.replace('.png', '_bw.png')

    empty_image.save(color_path)
    empty_image.convert('L').save(bw_path)

    print(f"身份证图片已生成: {color_path} (彩色), {bw_path} (黑白)")

    return color_path, bw_path, saved_avatar_path


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


class KYCTestClient:
    """KYC 测试客户端"""

    def __init__(self, api_url=KYC_API_URL):
        self.api_url = api_url

    def _make_request(self, api_type, user_id, image_path, nation="China"):
        """
        发送 KYC 认证请求

        Args:
            api_type: API 类型 (verify_idcard_front, verify_idcard_back, verify_passport)
            user_id: 用户唯一标识
            image_path: 图片路径
            nation: 国家类型

        Returns:
            dict: 响应结果
        """
        if not os.path.exists(image_path):
            return {
                "error": True,
                "message": f"图片文件不存在: {image_path}"
            }

        # 构建请求参数
        params = {
            "user_id": user_id,
            "nation": nation
        }

        request_data = {
            "api": api_type,
            "version": "1.0",
            "params": json.dumps(params)
        }

        # 准备文件
        files = None
        try:
            with open(image_path, 'rb') as f:
                files = {'user_file': (os.path.basename(image_path), f, 'image/png')}

                # 使用双重编码格式（与 auto_kyc_test.py 保持一致）
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
                "message": "无法连接到 KYC 服务器，请确认服务已启动"
            }
        except Exception as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }

    def verify_idcard_front(self, user_id, front_image_path, nation="China"):
        """
        验证身份证正面

        Returns:
            {
                "code": 0,  # 0表示成功
                "msg": "success",
                "data": {
                    "user_id": "...",
                    "id_card": "...",
                    "real_name": "...",
                    "file_url": "..."
                }
            }
        """
        print(f"\n{'='*60}")
        print(f"验证身份证正面 - 用户: {user_id}")
        print(f"{'='*60}")

        result = self._make_request("verify_idcard_front", user_id, front_image_path, nation)
        self._print_result("正面认证", result)
        return result

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
        print(f"   实际发送的 JSON: {json.dumps({'request': json.dumps(request_data)}, ensure_ascii=False)}")

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

    def verify_idcard_back(self, user_id, back_image_path, nation="China"):
        """
        验证身份证反面
        """
        print(f"\n{'='*60}")
        print(f"验证身份证反面 - 用户: {user_id}")
        print(f"{'='*60}")

        result = self._make_request("verify_idcard_back", user_id, back_image_path, nation)
        self._print_result("反面认证", result)
        return result

    def verify_passport(self, user_id, passport_image_path, nation=""):
        """
        验证护照
        """
        print(f"\n{'='*60}")
        print(f"验证护照 - 用户: {user_id}")
        print(f"{'='*60}")

        result = self._make_request("verify_passport", user_id, passport_image_path, nation)
        self._print_result("护照认证", result)
        return result

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
            # 使用 multipart/form-data 格式
            import io
            response = requests.post(
                self.api_url,
                files={"file": ("", io.BytesIO(b""), "application/octet-stream")},
                data={"request": json.dumps(request_data)},
                timeout=30
            )
            result = response.json()
            self._print_result("用户状态", result)
            return result
        except Exception as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }

    def get_user_info(self, user_id):
        """获取用户详细信息"""
        print(f"\n{'='*60}")
        print(f"查询用户信息 - 用户: {user_id}")
        print(f"{'='*60}")

        params = {
            "user_id": user_id
        }

        request_data = {
            "api": "get_user_info",
            "version": "1.0",
            "params": json.dumps(params)
        }

        try:
            # 使用 multipart/form-data 格式
            import io
            response = requests.post(
                self.api_url,
                files={"file": ("", io.BytesIO(b""), "application/octet-stream")},
                data={"request": json.dumps(request_data)},
                timeout=30
            )
            result = response.json()
            self._print_result("用户信息", result)
            return result
        except Exception as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }

    def _print_result(self, title, result):
        """打印结果"""
        if result.get("error"):
            print(f"❌ {title}失败: {result.get('message')}")
        else:
            code = result.get("code", -1)
            msg = result.get("msg", "")
            data = result.get("data", {})

            if code == 0:
                print(f"✅ {title}成功")
                if data:
                    print(f"   数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
            else:
                print(f"❌ {title}失败 (code={code}): {msg}")
                if data:
                    print(f"   数据: {json.dumps(data, ensure_ascii=False, indent=2)}")

    def run_full_idcard_test(self, user_id, front_image_path, back_image_path, avatar_path=None, nation="China"):
        """
        执行完整的身份证认证流程

        Args:
            user_id: 用户ID
            front_image_path: 正面图片路径
            back_image_path: 反面图片路径
            avatar_path: 头像路径（可选，用于人脸采集）
            nation: 国家类型

        Returns:
            bool: 是否全部成功
        """
        print(f"\n{'#'*60}")
        print(f"# 开始 KYC 身份证认证测试 - 用户: {user_id}")
        print(f"{'#'*60}")

        # 1. 验证正面
        front_result = self.verify_idcard_front(user_id, front_image_path, nation)
        if front_result.get("code") != 0:
            print("\n⚠️ 正面认证失败，停止测试")
            return False

        time.sleep(1)  # 间隔1秒

        # 2. 验证反面（人脸采集将在视频认证时进行）
        back_result = self.verify_idcard_back(user_id, back_image_path, nation)
        if back_result.get("code") != 0:
            print("\n⚠️ 反面认证失败")
            return False

        time.sleep(1)

        # 4. 查询状态
        self.get_user_status(user_id)

        # 5. 查询详细信息
        self.get_user_info(user_id)

        print(f"\n{'#'*60}")
        print(f"# 测试完成 - 用户: {user_id}")
        print(f"{'#'*60}")

        return True


def main():
    """主测试函数"""
    import argparse

    parser = argparse.ArgumentParser(description='KYC 证件认证测试脚本')
    parser.add_argument('--random', action='store_true', help='使用随机生成的测试数据')
    parser.add_argument('--count', type=int, default=1, help='随机测试次数（仅在使用 --random 时有效）')
    parser.add_argument('--user-id', type=str, default=None, help='指定用户ID（用于与视频测试保持一致）')
    parser.add_argument('--source', type=str, default='color.png', help='身份证图片路径（不使用随机模式时）')
    parser.add_argument('--no-avatar', action='store_true', help='不使用自动抠图（仅随机模式）')

    args = parser.parse_args()

    client = KYCTestClient()

    if args.random:
        # 随机模式：生成多组随机数据进行测试
        print("=" * 60)
        print(f"KYC 证件认证随机测试 (共 {args.count} 次)")
        print("=" * 60)
        print(f"API地址: {KYC_API_URL}")
        if args.user_id:
            print(f"指定用户ID: {args.user_id}")
        print("=" * 60)

        success_count = 0
        fail_count = 0

        for i in range(args.count):
            print(f"\n{'#'*60}")
            print(f"# 第 {i+1}/{args.count} 次测试")
            print(f"{'#'*60}")

            try:
                # 1. 先下载头像并检测性别
                print(f"\n正在下载头像并检测性别...")
                avatar_image, detected_gender = download_random_avatar()

                if avatar_image is None:
                    print("\n❌ 无法下载头像")
                    fail_count += 1
                    continue

                # 2. 根据检测到的性别生成用户数据
                user_data = generate_random_user_data(sex=detected_gender)

                # 如果指定了 user_id，使用指定的；否则使用随机生成的
                if args.user_id:
                    user_data['user_id'] = args.user_id

                # 打印生成的用户信息
                print(f"\n📋 生成的用户信息:")
                print(f"   用户ID: {user_data['user_id']}")
                print(f"   姓名: {user_data['name']}")
                print(f"   性别: {user_data['sex']}")
                print(f"   身份证号: {user_data['id_card']}")
                print(f"   地址: {user_data['address']}")

                # 为每个用户创建独立文件夹（与视频测试共用）
                output_dir = os.path.join("./kyc_test", user_data['user_id'])
                os.makedirs(output_dir, exist_ok=True)

                # 头像保存路径
                avatar_path = os.path.join(output_dir, "avatar.png")

                # 生成临时身份证图片（用于裁剪）
                temp_color_path = os.path.join(output_dir, f"temp_idcard_{user_data['user_id']}.png")

                color_path, bw_path, saved_avatar_path = generate_idcard_image(
                    user_data,
                    avatar_image=avatar_image,
                    output_path=temp_color_path,
                    auto_bg=not args.no_avatar,
                    avatar_output_path=avatar_path
                )

                # 裁剪正面和反面，保存为 idcard_front.png 和 idcard_back.png
                front_image, back_image = prepare_test_images(color_path, output_dir)

                # 删除临时完整身份证图片
                try:
                    if os.path.exists(color_path):
                        os.remove(color_path)
                    if os.path.exists(bw_path):
                        os.remove(bw_path)
                except:
                    pass

                if front_image is None or back_image is None:
                    print("\n❌ 无法准备测试图片")
                    fail_count += 1
                    continue

                # 执行测试（包含人脸采集）
                success = client.run_full_idcard_test(
                    user_data['user_id'],
                    front_image,
                    back_image,
                    avatar_path=saved_avatar_path
                )

                if success:
                    success_count += 1
                else:
                    fail_count += 1

                print(f"\n💡 视频测试命令:")
                if saved_avatar_path:
                    print(f"   python kyc_video_test.py --user-id {user_data['user_id']}")
                else:
                    print(f"   python kyc_video_test.py --user-id {user_data['user_id']}")

            except Exception as e:
                print(f"\n❌ 测试失败: {e}")
                fail_count += 1

            # 测试间隔
            if i < args.count - 1:
                time.sleep(2)

        # 打印测试总结
        print(f"\n{'='*60}")
        print(f"测试完成!")
        print(f"成功: {success_count}, 失败: {fail_count}")
        print(f"{'='*60}")

    else:
        # 传统模式：使用现有的身份证图片
        user_id = args.user_id if args.user_id else "1001529777"
        source_image = args.source

        # 为用户创建独立文件夹
        output_dir = os.path.join("./kyc_test", user_id)
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("KYC 证件认证测试")
        print("=" * 60)
        print(f"用户ID: {user_id}")
        print(f"源图片: {source_image}")
        print(f"API地址: {KYC_API_URL}")
        print("=" * 60)

        # 自动准备测试图片
        front_image, back_image = prepare_test_images(source_image, output_dir)

        if front_image is None or back_image is None:
            print("\n❌ 无法准备测试图片，测试终止")
            print("提示: 请先运行身份证生成器 (main.py) 生成 color.png 图片")
            print("       或者使用 --random 参数进行随机测试")
            return

        print(f"\n正面图片: {front_image}")
        print(f"反面图片: {back_image}")
        print("=" * 60)

        # 执行完整测试
        success = client.run_full_idcard_test(user_id, front_image, back_image)

        if success:
            print("\n✅ 所有测试通过!")
            print(f"\n💡 视频测试命令:")
            print(f"   python kyc_video_test.py --user-id {user_id} --idcard-front {front_image}")
        else:
            print("\n❌ 测试未完全通过")


if __name__ == "__main__":
    main()
