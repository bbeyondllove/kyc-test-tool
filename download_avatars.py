"""
头像预下载脚本
功能：
1. 下载指定数量的头像
2. 使用 DeepFace 判断性别
3. 按性别分类存储到不同文件夹

使用方法：
  python download_avatars.py --count 100
  python download_avatars.py --count 50 --output-dir ./avatars
"""
import os
import sys
import argparse
import urllib.request
import ssl
import json
from pathlib import Path
from datetime import datetime
import concurrent.futures
import io
import random
import string
import time

# 抑制警告
import warnings
warnings.filterwarnings('ignore')

# 导入 PIL
try:
    from PIL import Image
except ImportError:
    print("错误: 需要安装 Pillow")
    sys.exit(1)


class AvatarDownloader:
    """头像下载和分类器"""

    def __init__(self, output_dir="./avatars", max_workers=5):
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.male_dir = self.output_dir / "male"
        self.female_dir = self.output_dir / "female"
        self.unknown_dir = self.output_dir / "unknown"
        self.metadata_file = self.output_dir / "metadata.json"

        # 创建目录
        self.male_dir.mkdir(parents=True, exist_ok=True)
        self.female_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_dir.mkdir(parents=True, exist_ok=True)

        # 加载元数据
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """加载已下载头像的元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _download_avatar(self, index):
        """下载单张头像"""
        try:
            url = "https://thispersondoesnotexist.com"
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(request, context=context, timeout=30) as response:
                image_data = response.read()

            image = Image.open(io.BytesIO(image_data))
            return image, None
        except Exception as e:
            return None, str(e)

    def _detect_gender(self, image):
        """使用 DeepFace 检测性别"""
        try:
            from deepface import DeepFace
            # 将 PIL 图像转换为临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                image.save(tmp_file, format='JPEG')
                tmp_path = tmp_file.name

            try:
                # 检测性别
                result = DeepFace.analyze(
                    tmp_path,
                    actions=['gender'],
                    enforce_detection=False,
                    silent=True
                )

                if isinstance(result, list):
                    gender = result[0].get('dominant_gender', 'unknown').lower()
                else:
                    gender = result.get('dominant_gender', 'unknown').lower()

                # 返回中文性别
                if 'woman' in gender or 'female' in gender:
                    return '女'
                elif 'man' in gender or 'male' in gender:
                    return '男'
                else:
                    return 'unknown'
            finally:
                # 删除临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except ImportError:
            print("警告: DeepFace 未安装，无法检测性别")
            return 'unknown'
        except Exception as e:
            return 'unknown'

    def _generate_random_filename(self):
        """生成6位随机字母文件名"""
        return ''.join(random.choices(string.ascii_lowercase, k=6))

    def _save_avatar(self, image, gender, index, error=None):
        """保存头像到对应文件夹"""
        if error:
            return False, error

        try:
            # 确定保存目录
            if gender == '男':
                save_dir = self.male_dir
            elif gender == '女':
                save_dir = self.female_dir
            else:
                save_dir = self.unknown_dir

            # 生成随机文件名并确保不重复
            max_attempts = 10
            for _ in range(max_attempts):
                filename = self._generate_random_filename() + ".png"
                filepath = save_dir / filename
                if not filepath.exists():
                    break
            else:
                # 如果重复太多，用时间戳
                filename = self._generate_random_filename() + str(int(time.time())) + ".png"
                filepath = save_dir / filename

            # 保存头像
            image.save(filepath)

            # 更新元数据
            self.metadata[str(filepath)] = {
                'gender': gender,
                'filename': filename
            }

            return True, str(filepath)
        except Exception as e:
            return False, str(e)

    def download_and_classify(self, count, skip_existing=False):
        """下载并分类指定数量的头像"""
        print(f"\n{'='*60}")
        print(f"开始下载和分类头像")
        print(f"目标数量: {count}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")

        stats = {
            'male': 0,
            'female': 0,
            'unknown': 0,
            'failed': 0
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for i in range(count):
                # 检查是否已存在足够的头像（仅当 skip_existing=True）
                if skip_existing:
                    if stats['male'] + stats['female'] >= count:
                        break

                future = executor.submit(self._process_single_avatar, i)
                futures.append(future)

            # 处理结果
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                success, result = future.result()
                if success:
                    gender = result.get('gender', 'unknown')
                    stats[gender] = stats.get(gender, 0) + 1
                else:
                    stats['failed'] += 1

                # 显示进度
                total = i + 1
                print(f"\r进度: {total}/{len(futures)} | "
                      f"男: {stats['male']} | "
                      f"女: {stats['female']} | "
                      f"未知: {stats['unknown']} | "
                      f"失败: {stats['failed']}", end='', flush=True)

        print(f"\n\n{'='*60}")
        print(f"下载完成!")
        print(f"男性头像: {stats['male']}")
        print(f"女性头像: {stats['female']}")
        print(f"未知: {stats['unknown']}")
        print(f"失败: {stats['failed']}")
        print(f"{'='*60}\n")

        # 保存元数据
        self._save_metadata()

        return stats

    def _process_single_avatar(self, index):
        """处理单张头像（下载、检测、保存）"""
        # 下载头像
        image, error = self._download_avatar(index)
        if error:
            return False, {'error': error}

        # 检测性别
        gender = self._detect_gender(image)

        # 保存头像
        success, result = self._save_avatar(image, gender, index)

        if success:
            return True, {'gender': gender, 'filepath': result}
        else:
            return False, {'error': result}

    def get_avatar(self, gender='all'):
        """获取指定性别的头像"""
        if gender == 'all':
            dir_to_search = self.output_dir
        elif gender == '男':
            dir_to_search = self.male_dir
        elif gender == '女':
            dir_to_search = self.female_dir
        else:
            dir_to_search = self.unknown_dir

        # 获取所有头像文件
        avatar_files = list(dir_to_search.glob('avatar_*.png'))
        if not avatar_files:
            return None

        import random
        return random.choice(avatar_files)

    def get_avatar_count(self, gender='all'):
        """获取指定性别的头像数量"""
        if gender == 'all':
            dir_to_search = self.output_dir
        elif gender == '男':
            dir_to_search = self.male_dir
        elif gender == '女':
            dir_to_search = self.female_dir
        else:
            dir_to_search = self.unknown_dir

        return len(list(dir_to_search.glob('avatar_*.png')))


def main():
    parser = argparse.ArgumentParser(description='下载和分类头像')
    parser.add_argument('--count', type=int, default=100, help='下载头像数量 (默认: 100)')
    parser.add_argument('--output-dir', type=str, default='./avatars', help='输出目录 (默认: ./avatars)')
    parser.add_argument('--max-workers', type=int, default=5, help='并发下载数 (默认: 5)')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已存在的头像')

    args = parser.parse_args()

    downloader = AvatarDownloader(
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )

    stats = downloader.download_and_classify(args.count, args.skip_existing)

    # 显示统计信息
    print(f"\n当前头像统计:")
    print(f"  男性: {downloader.get_avatar_count('男')}")
    print(f"  女性: {downloader.get_avatar_count('女')}")
    print(f"  未知: {downloader.get_avatar_count('unknown')}")
    print(f"  总计: {downloader.get_avatar_count('all')}")


if __name__ == "__main__":
    main()
