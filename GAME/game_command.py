# 引入 Enum 模块，用于创建枚举
from enum import Enum


# 创建一个枚举类Color，从Python内置的枚举类Enum继承
class Command(Enum):
    # 定义 GREEN 数值为 2
    ATACK = 1
    #狮子头 暂时用不着，直接逻辑里判断合并在移动里了
    LION = 2
    # 定义 RED 数值为 1
    MOVE = 3


    #再次挑战
    AGAIN = 4

    # 定义 BLUE 数值为 3
    RERTURN = 5
    #放技能
    SKILL = 6
    #等待时间
    SLEEP = 7


