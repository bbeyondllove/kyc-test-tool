import random
import sys
import threading
import tkinter
from tkinter.filedialog import *
from tkinter.messagebox import *
from tkinter.ttk import *
import platform
import urllib.request
import io
import ssl

import PIL.Image as PImage
import cv2
import numpy
from PIL import ImageFont, ImageDraw, ImageTk
import os
from idcard_generator import id_card_utils, name_utils, utils, loading_alert, region_data

# 尝试导入 DeepFace 用于性别检测
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("警告: DeepFace 未安装，无法自动检测性别。请运行: pip install deepface")

asserts_dir = os.path.join(utils.get_base_path(), 'asserts')
print("asserts_dir", asserts_dir)


def set_entry_value(entry, value):
    entry.delete(0, tkinter.END)
    entry.insert(0, value)


def change_background(img, img_back, zoom_size, center):
    # 缩放
    img = cv2.resize(img, zoom_size)
    rows, cols, channels = img.shape

    # 转换hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 获取mask
    # lower_blue = np.array([78, 43, 46])
    # upper_blue = np.array([110, 255, 255])
    diff = [5, 30, 30]
    gb = hsv[0, 0]
    lower_blue = numpy.array(gb - diff)
    upper_blue = numpy.array(gb + diff)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.imshow('Mask', mask)

    # 腐蚀膨胀
    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    # 粘贴
    for i in range(rows):
        for j in range(cols):
            if dilate[i, j] == 0:  # 0代表黑色的点
                img_back[center[0] + i, center[1] + j] = img[i, j]  # 此处替换颜色，为BGR通道

    return img_back


def paste(avatar, bg, zoom_size, center):
    avatar = cv2.resize(avatar, zoom_size)
    rows, cols, channels = avatar.shape
    for i in range(rows):
        for j in range(cols):
            bg[center[0] + i, center[1] + j] = avatar[i, j]
    return bg


class IDGen:
    def __init__(self):
        self.f_name = None
        self.avatar_image = None  # 存储下载的头像

    def download_random_avatar(self):
        """从 thispersondoesnotexist.com 下载随机 AI 生成的头像"""
        self.loading_bar = loading_alert.LoadingBar(title="提示", content="正在下载 AI 头像...")
        self.loading_bar.show(self.root)

        download_thread = threading.Thread(target=self._download_avatar_thread)
        download_thread.setDaemon(True)
        download_thread.start()

    def _detect_gender(self, pil_image):
        """使用 DeepFace 检测图片中人脸的性别"""
        if not DEEPFACE_AVAILABLE:
            return None
        
        try:
            # 将 PIL Image 转换为 numpy 数组 (BGR 格式供 OpenCV/DeepFace 使用)
            img_array = numpy.array(pil_image.convert('RGB'))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # 使用 DeepFace 分析性别
            result = DeepFace.analyze(img_bgr, actions=['gender'], enforce_detection=False)
            
            # 结果可能是列表或字典
            if isinstance(result, list):
                result = result[0]
            
            dominant_gender = result.get('dominant_gender', None)
            if dominant_gender == 'Woman':
                return '女'
            elif dominant_gender == 'Man':
                return '男'
            return None
        except Exception as e:
            print(f"性别检测失败: {e}")
            return None

    def _on_avatar_downloaded(self, detected_gender):
        """头像下载完成后的回调"""
        if detected_gender:
            set_entry_value(self.eSex, detected_gender)
            showinfo('成功', f'AI 头像已下载，检测到性别: {detected_gender}\n点击"生成身份证"按钮生成图片')
        else:
            showinfo('成功', 'AI 头像已下载，点击"生成身份证"按钮生成图片')

    def _download_avatar_thread(self):
        """在后台线程中下载头像"""
        try:
            url = "https://thispersondoesnotexist.com"
            # 创建请求，添加 User-Agent 避免被拒绝
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            # 忽略 SSL 证书验证（某些系统可能有问题）
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(request, context=context, timeout=30) as response:
                image_data = response.read()

            # 将下载的数据转换为 PIL Image
            self.avatar_image = PImage.open(io.BytesIO(image_data))
            self.f_name = None  # 清除文件路径，表示使用内存中的图片

            # 先关闭加载框，快速响应用户
            self.loading_bar.close()
            self.root.after(0, lambda: showinfo('成功', 'AI 头像已下载，正在检测性别...'))
            
            # 后台检测性别（不阻塞用户操作）
            detected_gender = self._detect_gender(self.avatar_image)
            if detected_gender:
                self.root.after(0, lambda g=detected_gender: self._update_gender(g))
        except Exception as e:
            self.loading_bar.close()
            self.root.after(0, lambda: showerror('错误', f'下载头像失败: {str(e)}'))

    def _update_gender(self, gender):
        """更新性别字段"""
        set_entry_value(self.eSex, gender)
        showinfo('提示', f'检测到性别: {gender}')

    def random_data(self):
        # 随机姓名和性别
        random_name = name_utils.random_name()
        set_entry_value(self.eName, random_name["name_full"])
        set_entry_value(self.eSex, '女' if random_name['sex'] == 0 else "男")
        set_entry_value(self.eNation, "汉")
        
        # 随机出生日期
        year = random.randint(1960, 2005)
        month = random.randint(1, 12)
        day = id_card_utils.random_day(year, month)
        set_entry_value(self.eYear, year)
        set_entry_value(self.eMon, month)
        set_entry_value(self.eDay, day)
        
        # 随机省份地址和签发机关
        region_info = region_data.random_full_data()
        set_entry_value(self.eAddr, region_info["address"])
        set_entry_value(self.eOrg, region_info["issuing_authority"])
        
        # 使用正确的地区代码生成身份证号
        set_entry_value(self.eIdn, id_card_utils.random_card_no(
            prefix=region_info["code"],
            year=str(year),
            month=str(month),
            day=str(day)
        ))
        
        # 随机有效期限
        start_time = id_card_utils.get_start_time()
        expire_time = id_card_utils.get_expire_time()
        set_entry_value(self.eLife, start_time + "-" + expire_time)

    def select_avatar(self):
        """选择本地头像文件"""
        self.f_name = askopenfilename(initialdir=os.getcwd(), title='选择头像')
        if len(self.f_name) == 0:
            return
        self.avatar_image = None  # 清除下载的头像，使用文件
        showinfo('成功', f'已选择头像: {os.path.basename(self.f_name)}\n点击"生成身份证"按钮生成图片')

    def generator_image(self):
        """生成身份证图片"""
        # 检查是否有头像
        if self.f_name is None and self.avatar_image is None:
            showerror('错误', '请先选择头像或下载 AI 头像')
            return

        self.loading_bar = loading_alert.LoadingBar(title="提示", content="图片正在生成...")
        self.loading_bar.show(self.root)

        # 开启新线程保持滚动条显示
        wait_thread = threading.Thread(target=self.handle_image)
        wait_thread.setDaemon(True)
        wait_thread.start()

    def handle_image(self):
        # 从文件或内存加载头像
        if self.avatar_image is not None:
            avatar = self.avatar_image.copy()
        else:
            avatar = PImage.open(self.f_name)  # 500x670
        empty_image = PImage.open(os.path.join(asserts_dir, 'empty.png'))

        name_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/hei.ttf'), 72)
        other_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/hei.ttf'), 64)
        birth_date_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/fzhei.ttf'), 60)
        id_font = ImageFont.truetype(os.path.join(asserts_dir, 'fonts/ocrb10bt.ttf'), 90)

        draw = ImageDraw.Draw(empty_image)
        draw.text((630, 690), self.eName.get(), fill=(0, 0, 0), font=name_font)
        draw.text((630, 840), self.eSex.get(), fill=(0, 0, 0), font=other_font)
        draw.text((1030, 840), self.eNation.get(), fill=(0, 0, 0), font=other_font)
        draw.text((630, 975), self.eYear.get(), fill=(0, 0, 0), font=birth_date_font)
        draw.text((950, 975), self.eMon.get(), fill=(0, 0, 0), font=birth_date_font)
        draw.text((1150, 975), self.eDay.get(), fill=(0, 0, 0), font=birth_date_font)

        # 住址
        addr_loc_y = 1115
        addr_lines = self.get_addr_lines()
        for addr_line in addr_lines:
            draw.text((630, addr_loc_y), addr_line, fill=(0, 0, 0), font=other_font)
            addr_loc_y += 100

        # 身份证号
        draw.text((900, 1475), self.eIdn.get(), fill=(0, 0, 0), font=id_font)

        # 背面
        draw.text((1050, 2750), self.eOrg.get(), fill=(0, 0, 0), font=other_font)
        draw.text((1050, 2895), self.eLife.get(), fill=(0, 0, 0), font=other_font)

        if self.eBgvar.get():
            avatar = cv2.cvtColor(numpy.asarray(avatar), cv2.COLOR_RGBA2BGRA)
            empty_image = cv2.cvtColor(numpy.asarray(empty_image), cv2.COLOR_RGBA2BGRA)
            empty_image = change_background(avatar, empty_image, (500, 670), (690, 1500))
            empty_image = PImage.fromarray(cv2.cvtColor(empty_image, cv2.COLOR_BGRA2RGBA))
        else:
            avatar = avatar.resize((500, 670))
            avatar = avatar.convert('RGBA')
            empty_image.paste(avatar, (1500, 690), mask=avatar)
            # im = paste(avatar, im, (500, 670), (690, 1500))

        empty_image.save('color.png')
        empty_image.convert('L').save('bw.png')

        self.loading_bar.close()
        showinfo('成功', '文件已生成到目录下,黑白bw.png和彩色color.png')

    def show_ui(self, root):
        self.root = root
        root.title('AIRobot身份证图片生成器')
        # root.geometry('640x480')
        root.resizable(width=False, height=False)
        Label(root, text='请遵守法律法规', foreground="#FF0000").grid(row=0, column=0, sticky=tkinter.W, padx=3, pady=3, columnspan=6)

        Label(root, text='姓名:').grid(row=1, column=0, sticky=tkinter.W, padx=3, pady=3)
        self.eName = Entry(root, width=8)
        self.eName.grid(row=1, column=1, sticky=tkinter.W, padx=3, pady=3)
        Label(root, text='性别:').grid(row=1, column=2, sticky=tkinter.W, padx=3, pady=3)
        self.eSex = Entry(root, width=8)
        self.eSex.grid(row=1, column=3, sticky=tkinter.W, padx=3, pady=3)
        Label(root, text='民族:').grid(row=1, column=4, sticky=tkinter.W, padx=3, pady=3)
        self.eNation = Entry(root, width=8)
        self.eNation.grid(row=1, column=5, sticky=tkinter.W, padx=3, pady=3)
        Label(root, text='出生年:').grid(row=2, column=0, sticky=tkinter.W, padx=3, pady=3)
        self.eYear = Entry(root, width=8)
        self.eYear.grid(row=2, column=1, sticky=tkinter.W, padx=3, pady=3)
        Label(root, text='月:').grid(row=2, column=2, sticky=tkinter.W, padx=3, pady=3)
        self.eMon = Entry(root, width=8)
        self.eMon.grid(row=2, column=3, sticky=tkinter.W, padx=3, pady=3)
        Label(root, text='日:').grid(row=2, column=4, sticky=tkinter.W, padx=3, pady=3)
        self.eDay = Entry(root, width=8)
        self.eDay.grid(row=2, column=5, sticky=tkinter.W, padx=3, pady=3)
        Label(root, text='住址:').grid(row=3, column=0, sticky=tkinter.W, padx=3, pady=3)
        self.eAddr = Entry(root, width=32)
        self.eAddr.grid(row=3, column=1, sticky=tkinter.W, padx=3, pady=3, columnspan=5)
        Label(root, text='证件号码:').grid(row=4, column=0, sticky=tkinter.W, padx=3, pady=3)
        self.eIdn = Entry(root, width=32)
        self.eIdn.grid(row=4, column=1, sticky=tkinter.W, padx=3, pady=3, columnspan=5)
        Label(root, text='签发机关:').grid(row=5, column=0, sticky=tkinter.W, padx=3, pady=3)
        self.eOrg = Entry(root, width=32)
        self.eOrg.grid(row=5, column=1, sticky=tkinter.W, padx=3, pady=3, columnspan=5)
        Label(root, text='有效期限:').grid(row=6, column=0, sticky=tkinter.W, padx=3, pady=3)
        self.eLife = Entry(root, width=32)
        self.eLife.grid(row=6, column=1, sticky=tkinter.W, padx=3, pady=3, columnspan=5)
        Label(root, text='选项:').grid(row=7, column=0, sticky=tkinter.W, padx=3, pady=3)
        self.eBgvar = tkinter.IntVar()
        self.eBgvar.set(1)
        self.ebg = Checkbutton(root, text='自动抠图', variable=self.eBgvar)
        self.ebg.grid(row=7, column=1, sticky=tkinter.W, padx=3, pady=3, columnspan=5)

        random_btn = Button(root, text='随机数据', width=8, command=self.random_data)
        random_btn.grid(row=8, column=0, sticky=tkinter.W, padx=16, pady=3, columnspan=2)

        # 头像选择按钮
        select_avatar_btn = Button(root, text='选择本地头像', width=12, command=self.select_avatar)
        select_avatar_btn.grid(row=9, column=0, sticky=tkinter.W, padx=16, pady=3, columnspan=2)
        ai_avatar_btn = Button(root, text='AI随机头像', width=12, command=self.download_random_avatar)
        ai_avatar_btn.grid(row=9, column=2, sticky=tkinter.W, padx=1, pady=3, columnspan=2)

        # 生成按钮
        generator_btn = Button(root, text='生成身份证', width=12, command=self.generator_image)
        generator_btn.grid(row=9, column=4, sticky=tkinter.W, padx=1, pady=3, columnspan=2)

        # 触发随机生成
        self.random_data()

    # 获得要显示的住址数组
    def get_addr_lines(self):
        addr = self.eAddr.get()
        addr_lines = []
        start = 0
        while start < utils.get_show_len(addr):
            show_txt = utils.get_show_txt(addr, start, start + 22)
            addr_lines.append(show_txt)
            start = start + 22

        return addr_lines

    def run(self):
        root = tkinter.Tk()
        self.show_ui(root)
        # 跨平台图标设置
        current_os = platform.system()
        if current_os == 'Windows':
            ico_path = os.path.join(asserts_dir, 'ico.ico')
            root.iconbitmap(ico_path)
        elif current_os == 'Darwin':  # macOS
            # macOS 使用 iconbitmap 可能不生效，但不会报错
            ico_path = os.path.join(asserts_dir, 'ico.icns')
            try:
                root.iconbitmap(ico_path)
            except tkinter.TclError:
                pass
        else:  # Linux 及其他系统
            # Linux 使用 PhotoImage 加载 PNG 图标
            ico_png_path = os.path.join(asserts_dir, 'ico.png')
            if os.path.exists(ico_png_path):
                icon = ImageTk.PhotoImage(PImage.open(ico_png_path))
                root.iconphoto(True, icon)
        root.protocol('WM_DELETE_WINDOW', lambda: sys.exit(0))
        root.mainloop()
