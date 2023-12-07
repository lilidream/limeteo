"""
实用函数
"""

import time

def timeit(func):
    """
    计算函数运行时间的装饰器
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print("运行时间：", time.time() - start, "秒")
    return wrapper


def used_time(func):
    """
    计算语句运行时间
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        return time.time() - start
    return wrapper