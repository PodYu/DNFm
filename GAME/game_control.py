import time
from typing import Tuple

from ADB.scrcpy_adb_1 import ScreenController
import math
import random

from dnfm.game.role import Role
from dnfm.game.skill import Skill


class GameControl:
    def __init__(self, adb: ScreenController):
        self.adb = adb

        self.is_move = False
        #初始化为轮船的中心点
        self.last_move = [445, 848]

    #计算基于x1,y1移动的角度 移动的时间暂时没计算
    def calc_angle(x1, y1, x2, y2):
        angle = math.atan2(y1 - y2, x1 - x2)
        return 180 - int(angle * 180 / math.pi)

    def calc_mov_point(self, angle: float) -> Tuple[int, int]:
        rx, ry = (445, 848)  #小米8的轮船坐标中心点
        r = 150 #轮盘半径

        x = rx + r * math.cos(angle * math.pi / 180)
        y = ry - r * math.sin(angle * math.pi / 180)
        return int(x), int(y)

    #开始移动
    def move(self, angle: float, t: float=0):
        # 计算轮盘x, y坐标
        x, y = self.calc_mov_point(angle)
        # self.adb.touch_start(x, y)
        # time.sleep(t)
        # self.adb.touch_end(x, y)
        #结束

        if not self.is_move :
            self.adb.touch_end(x, y, 1)
            self.adb.touch_start(x, y, 1)
            self.is_move = True
            self.adb.touch_move(x, y, 1)
            self.last_move = [x, y]
        else:
            self.adb.touch_move(x, y, 1)
            self.last_move = [x, y]
        if t == 0:
            return
        time.sleep(t)

    #结束移动
    def end_move(self):
        if self.is_move:
            self.is_move = False
            self.adb.touch_end(self.last_move[0], self.last_move[1], 1)
        return
    #普通攻击
    def attack(self, t: float = 0.01):
        x, y = (1910, 910)
        x = self.random_x(x)
        y = self.random_x(y)
        self.adb.touch_start(x, y)
        time.sleep(t)
        self.adb.touch_end(x, y)



    #释放技能 技能有长按有短按
    def skill(self,x: int,y: int,t: float = 0.3,doubleClick: bool=False):
        #给技能触摸点添加偏移
        x = self.random_x(x)
        y = self.random_x(y)

        self.adb.touch_start(x, y)
        time.sleep(t)
        self.adb.touch_end(x, y)
        #如果是双击 就再释放一次
        if(doubleClick):
            time.sleep(0.1)
            self.adb.touch_start(x, y)
            time.sleep(t)
            self.adb.touch_end(x, y)

    def random_x(self,x,):
        randval = random.randint(1,10)
        val = random.randint(1, 10)
        if randval<=5:
            x = x-val
        else:
            x = x+val

        return x

    def randonSkill(self):
        skills = Skill.getSkills()
        skillRange = random.randint(0, 8)
        skill = skills[skillRange]
        self.touch_skill(skill)
    def randonSkillRole(self,role):
        skills = Skill.get_role_skills(role)
        skillRange = random.randint(0, 8)
        skill = skills[skillRange]
        self.touch_skill(skill)
    #放觉醒
    def touchSkillJx(self):
        print("放觉醒")
        skill = Skill.getSkillJX()
        self.touch_skill(skill)

    def skillBuff(self):
        print("开始加buff")
        for skill_buff in Skill.getBuffSkills():
            self.touch_skill(skill_buff)
            time.sleep(1)
    #放最常用的技能 暂时都放在技能3上
    def skill_near(self):
        skill_near = Skill.getSkill3()
        self.touch_skill(skill_near)
    #直接传入一个技能开始放
    def touch_skill(self,skill):
        print(f"释放技能{skill.detail}")
        self.skill(skill.x, skill.y, skill.t, skill.doubleClick)
    #再次挑战
    def startNextGame(self):
        print("。。。。。。。。。。。。。。。再次挑战。。。。。。。。。。。。。")
        x = 1936
        y = 254
        self.click(x,y)
        time.sleep(0.8)
        self.click(1304, 691)
    def returnCity(self):
        print("。。。。。。。。。。。。。。。返回城镇。。。。。。。。。。。。。")
        time.sleep(0.8)
        x = 1900
        y = 450
        self.click(x,y)


    def click(self, x, y, t: float = 0.01):
        x, y = self._ramdon_xy(x, y)
        self.adb.touch_start(x, y)
        time.sleep(t)
        self.adb.touch_end(x, y)

    def _ramdon_xy(self, x, y):
        x = x + random.randint(-5, 5)
        y = y + random.randint(-5, 5)
        return x, y
    def calc_move_point_direction(self, direction: str):
        if direction is None:
            return None
        # 计算轮盘x, y坐标
        angle = 0
        if direction == 'up':
            angle = 90
        if direction == 'down':
            angle = 270
        if direction == 'left':
            angle = 150
        x, y = self.calc_mov_point(angle)
        return x, y

    def touch_skill_idx(self, param,role):
        skill = Skill.getSkills(param)
        self.touch_skill(skill)

    def repair(self):
        print("。。。。。。。。。。。。。。。修理装备。。。。。。。。。。。。。")
        x = 1950
        y = 520
        self.click(x, y)
        time.sleep(0.8)
        self.click(1138, 966)
        time.sleep(0.8)
        #关闭修理框
        self.click(1832, 104)

        pass

    #不同的角色返回不同的技能
    def skill_free(self, role):
        if role == Role.HONGYAN:
            self.touch_skill(Skill.getSkillsHongyan()[2])
        elif role == Role.NAIMA:
            self.touch_skill(Skill.getSkillsNaima()[0])
            time.sleep(0.2)
            self.touch_skill(Skill.getSkillsNaima()[1])
        elif role == Role.QIGONG:
            self.touch_skill(Skill.getSkillsQigong()[2])
            time.sleep(0.2)
            self.touch_skill(Skill.getSkillsQigong()[0])
        elif role == Role.Jianshen:
            self.touch_skill(Skill.getSkillsJianshen()[1])
            time.sleep(0.2)
            self.touch_skill(Skill.getSkillsJianshen()[2])


if __name__ == '__main__':
    ctl = GameControl(ScrcpyADB())

    skills = Skill.getSkills()
    for skill in skills:
        time.sleep(1)
        ctl.touch_skill(skill)



    print("start move 0")
    ctl.move(0, 1)
    time.sleep(0.3)
    print("start move 180")
    ctl.move(90, 1)
    time.sleep(0.3)
    print("start move 180")
    ctl.move(180, 1)
    time.sleep(0.3)
    print("start to attack")
    ctl.attack()
    time.sleep(0.3)
    print("start move 170")
    ctl.move(270, 1)
    time.sleep(0.3)
    ctl.attack(3)




