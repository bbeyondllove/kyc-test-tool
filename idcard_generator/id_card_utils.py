import datetime
import random
import calendar


# 计算最后一位校验码
class IdentityCard:
    __Wi = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    __Ti = ['1', '0', 'x', '9', '8', '7', '6', '5', '4', '3', '2']

    @staticmethod
    def check(code):
        return len(code) == 17

    @staticmethod
    def calculate(code):
        sum_value = 0
        for i in range(17):
            sum_value += int(code[i]) * IdentityCard.__Wi[i]
        return IdentityCard.__Ti[sum_value % 11]


# 随机生成身份证号
def random_card_no(prefix='', year='', month='', day='', sex=None):
    """
    生成随机身份证号码

    Args:
        prefix: 地区代码（6位）
        year: 出生年份
        month: 出生月份
        day: 出生日期
        sex: 性别（'男'、'女'、0 或 1），0/女=偶数，1/男=奇数

    Returns:
        str: 18位身份证号码
    """
    if len(prefix) < 6:
        prefix = '513701'
    else:
        prefix = prefix[0:6]

    if len(year) > 0:
        year = str(year).zfill(4)
    else:
        year = str(random.randint(1900, 2020))

    if len(month) > 0:
        month = str(month[0:2]).zfill(2)
    else:
        month = str(random.randint(1, 12)).zfill(2)

    if len(day) > 0:
        day = str(day).zfill(2)
    else:
        day = str(random.randint(1, 28)).zfill(2)

    # 生成前15位和第17位（性别位）
    if sex is not None:
        # 根据性别确定第17位：奇数=男，偶数=女
        # sex: 0/女=偶数，1/男=奇数，或 '男'/'女'
        sex_code = 0 if sex in ['女', 0] else 1
        if sex_code == 0:  # 女，偶数
            seq_digit = random.choice([0, 2, 4, 6, 8])
        else:  # 男，奇数
            seq_digit = random.choice([1, 3, 5, 7, 9])
        # 第16位随机
        rand_int1 = random.randint(0, 99)
        rand_value = str(rand_int1).zfill(2) + str(seq_digit)
    else:
        # 不指定性别，完全随机
        rand_int = random.randint(0, 999)
        rand_value = str(rand_int).zfill(3)

    card_17 = prefix + year + month + day + rand_value
    vi_code = IdentityCard.calculate(card_17)
    return card_17 + vi_code


# 随机生成身份证号
def random_day(year=2020, month=2):
    monthRange = calendar.monthrange(year, month)
    return random.randint(1, monthRange[1])


# 身份证起始日
def get_expire_time():
    expire_time = datetime.datetime.now() + datetime.timedelta(days=365 * 10)
    return expire_time.strftime("%Y.%m.%d")


# 身份证到期日
def get_start_time():
    start_time = datetime.datetime.now() - datetime.timedelta(days=365 * 10)
    return start_time.strftime("%Y.%m.%d")


def test():
    code = random_card_no()  # 17位身份证
    print(code)
    if IdentityCard.check(code):
        print("你的校验位为:%s" % IdentityCard.calculate(code))


if __name__ == '__main__':
    test()
