"""
预先缩小所有头像50%并保存到新文件夹
"""
import os
from pathlib import Path
from PIL import Image

# 源目录和目标目录
source_dir = Path("./avatars")
target_dir = Path("./avatars_resized")

# 缩放比例
scale = 0.5

# 创建目标目录结构
target_dir.mkdir(exist_ok=True)
for subdir in ["male", "female", "unknown"]:
    (target_dir / subdir).mkdir(exist_ok=True)

# 处理每个子目录
for subdir in ["male", "female", "unknown"]:
    source_folder = source_dir / subdir
    target_folder = target_dir / subdir

    if not source_folder.exists():
        print(f"跳过不存在的目录: {source_folder}")
        continue

    # 处理文件夹中的所有图片
    for source_file in source_folder.glob("*.png"):
        print(f"处理: {source_folder.name}/{source_file.name}")

        try:
            # 加载图片
            img = Image.open(source_file)

            # 计算新尺寸
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)

            # 缩放
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 保存到目标目录
            target_file = target_folder / source_file.name
            img_resized.save(target_file)

            print(f"  -> {img.width}x{img.height} -> {new_width}x{new_height}")

        except Exception as e:
            print(f"  -> 失败: {e}")

print("\n处理完成!")
print(f"缩小后的头像保存在: {target_dir}")
