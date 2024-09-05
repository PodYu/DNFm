import threading
import traceback
from typing import Tuple

import easyocr

from dnfm.game.game_control  import GameControl
from dnfm.adb.scrcpy_adb import ScrcpyADB
from dnfm.game.game_command import  Command
from dnfm.game.label import Label
from dnfm.game.role import Role
from dnfm.utils.room_skill_util import RoomSkillUtil
import time
import cv2 as cv
from ncnn.utils.objects import Detect_Object
import math
import random
import numpy as np

from dnfm.utils import room_calutil
from dnfm.utils.cvmatch import image_match_util
from dnfm.vo.game_param_vo import GameParamVO
from ultralytics import YOLOv10



def get_detect_obj_right(obj: Detect_Object) -> Tuple[int, int]:
    return int(obj.xywh[0][0].item() + obj.xywh[0][2].item()), int(
        obj.xywh[0][1].item() + obj.xywh[0][3].item() / 2)
    # return int(obj.rect.x + obj.rect.w), int(obj.rect.y + obj.rect.h/2)


def get_detect_obj_center(obj: Detect_Object) -> Tuple[int, int]:
    return int(obj.xywh[0][0].item() + obj.xywh[0][2].item() / 2),int(obj.xywh[0][1].item() + obj.xywh[0][3].item()/2)
    # return int(obj.rect.x + obj.rect.w/2), int(obj.rect.y + obj.rect.h/2)

def get_detect_obj_bottom(obj: Detect_Object) -> Tuple[int, int]:
    return int(obj.xywh[0][0].item()+obj.xywh[0][2].item()/2), int(obj.xywh[0][1].item()+obj.xywh[0][3].item())
    # return int(obj.rect.x + obj.rect.w / 2), int(obj.rect.y + obj.rect.h)


def distance_detect_object(a: Detect_Object, b: Detect_Object):
    return math.sqrt((a.xywh[0][0].item() - b.xywh[0][0].item()) ** 2 + (a.xywh[0][1].item() - b.xywh[0][1].item()) ** 2)


def distance(x1, y1, x2, y2):
    dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return dist

def calc_angle(x1, y1, x2, y2):
    angle = math.atan2(y1 - y2, x1 - x2)
    return 180 - int(angle * 180 / math.pi)


class GameAction:
    def __init__(self, ctrl: GameControl):
        self.ctrl = ctrl
        self.yolo = YOLOv10("D:\\repository\\git\\yolov10-main\\runs\detect\\train_v1034\\weights\\best.pt")

        # Create an OCR reader object
        self.ocr = easyocr.Reader(['ch_sim'])

        self.adb = self.ctrl.adb
        self.param = GameParamVO()
        self.sleep_count = 0

        self.class_name=None

        #默认是没进过布万家的狮子头 进过之后要改成True 打完Boss又改为False
        self.bwj_szt_entryed_flag = False
        self.atack_flag = False
        self.move_flag = False
        #当前场景
        self.curl_env = "塞丽亚房间"
        #上一个场景
        self.last_env = ""

        self.hero_x = 0
        self.hero_y = 0
        self.moveto_x = 0
        self.moveto_y = 0





    def move(self,t: float=1.2):
        self.move_flag = True
        moveAngle = calc_angle(self.hero_x, self.hero_y, self.moveto_x, self.moveto_y)

        print("hero x,y:", self.hero_x, self.hero_y)
        print("move to:", self.moveto_x, self.moveto_y)
        print("calc move angle:", moveAngle)
        self.ctrl.move(moveAngle, t)

        self.move_flag = False
    '''
      获取当前的指令
      当前的指令定义只有2个 一个是攻击一个是移动
      如果没有指令则是卡死
      入参为预测的结果
      返回的是指令集 然后按照枚举的顺序 每次只执行一个指令
    '''
    def get_cur_order(self,result):

        rtn_arry = []
        hero_label = 2
        arrow_label = 0
        gate_label = 1
        item_lable = 3
        monster_label = 4
        boss_label = 5
        lion_gate_label = 7
        bwj_room5 = 12
        # 狮子头房间
        bwj_room6 = 13
        # boss房间
        bwj_room9 = 16
        # 下一个门的路径
        next_gate_label = 6

        # 要有个移动优先级 1、是物品 2、是箭头 3才是门 现在门没做具体标记不好按固定路线走
        item_label_flag = False
        arrow_label_flag = False
        next_gate_label = False
        #狮子头
        lion_gate_flag = False
        for box in result[0].boxes:
            if box is None:
                continue

            itemTmp = box.cls.item()
            for labelTmp in Label:
                # match itemTmp:
                #    case Label.hero.value:
                #     if (len(box.xywh[0]) > 0):
                #         self.hero_x = box.xywh[0][0].item()
                #         self.hero_y = box.xywh[0][1].item()
                if labelTmp.value == itemTmp:
                    if Label.hero.value == itemTmp:
                        if (len(box.xywh[0]) > 0):
                            self.hero_x = box.xywh[0][0].item()
                            self.hero_y = box.xywh[0][1].item()
                        continue
                    elif Label.arrow.value == itemTmp or Label.lion_gate.value == itemTmp or Label.item.value == itemTmp or Label.next_gate.value == itemTmp:

                        # 考虑优先级 物品优先级最高
                        if item_label_flag:
                            continue

                        if Label.item.value == itemTmp:
                            rtn_arry.append(
                                Command.MOVE
                            )
                            item_label_flag = True
                            if (len(box.xywh[0]) > 0):
                                self.moveto_x = box.xywh[0][0].item()
                                self.moveto_y = box.xywh[0][1].item()
                            continue

                        #狮子头优先级在过门之前
                        if lion_gate_flag:
                            continue
                        if Label.lion_gate.value == itemTmp:
                            #只有在未进入过狮子头时地和进入  不然会死循环
                            if not self.bwj_szt_entryed_flag:
                                rtn_arry.append(
                                    Command.MOVE
                                )
                                lion_gate_flag = True
                                if (len(box.xywh[0]) > 0):
                                    self.moveto_x = box.xywh[0][0].item()
                                    self.moveto_y = box.xywh[0][1].item()
                            continue

                        # 优先级2 过门 如果有下一个门出现 阤进行移动 就不看后续的箭头位置了
                        if next_gate_label:
                            continue

                        if Label.next_gate.value == box.cls.item():
                            rtn_arry.append(
                                Command.MOVE
                            )
                            next_gate_label = True
                            if (len(box.xywh[0]) > 0):
                                self.moveto_x = box.xywh[0][0].item()
                                self.moveto_y = box.xywh[0][1].item()
                            continue

                        #箭头移动优先级最后面
                        if Label.arrow.value == box.cls.item():
                            rtn_arry.append(
                                Command.MOVE
                            )
                            if (len(box.xywh[0]) > 0):
                                self.moveto_x = box.xywh[0][0].item()
                                self.moveto_y = box.xywh[0][1].item()

                        # # 如果当前已经有物品或箭头了 先捡物品和移动
                        # if item_label_flag or arrow_label_flag or next_gate_label:
                        continue
                    #狮子头 如果没进过，则优先进狮子头，判断逻辑 判断lion_flag =  True.
                    elif Label.monster.value == itemTmp or Label.boss.value == itemTmp:
                        if (len(box.xywh[0]) > 0):
                            self.moveto_x = box.xywh[0][0].item()
                            self.moveto_y = box.xywh[0][1].item()
                        rtn_arry.append(Command.ATACK)
                    #布万家前4图
                    elif Label.bwj_room1.value == itemTmp or Label.bwj_room2.value == itemTmp or Label.bwj_room3.value == itemTmp or Label.bwj_room4.value == itemTmp:
                        #当前在的地图
                        self.curl_env = itemTmp
                    elif  Label.bwj_room5.value == itemTmp or Label.bwj_room6.value == itemTmp or Label.bwj_room7.value == itemTmp or Label.bwj_room8.value == itemTmp or Label.bwj_room9.value == itemTmp :
                        #当前在的地图
                        self.curl_env = itemTmp
                        if Label.bwj_room5.value == itemTmp:
                            print("on bwj_room 5")
                        elif Label.bwj_room6.value == itemTmp:
                            print("on bwj_room 6 狮子头")
                           #在狮子头房间 进去后 改为已经进入过狮子头  并放觉醒
                            if not self.bwj_szt_entryed_flag :
                               self.bwj_szt_entryed_flag = True
                               self.ctrl.touchSkillJx()
                        #Boss 房 把已经 进过狮子头重置
                        elif Label.bwj_room9.value == itemTmp:
                            print("on bwj_room 9 boss房间")
                            self.bwj_szt_entryed_flag = False
                            #boss房 返回再次挑战
                            rtn_arry.append(Command.AGAIN)
        return rtn_arry

    #攻击
    def atack(self):
        # #攻击
        # if self.atack_flag:

            print("start atack")
            # self.move(0.2)
            self.ctrl.skill_near()
            # time.sleep(1)
            self.ctrl.randonSkill()
            # time.sleep(0.6)
            # self.ctrl.randonSkill()
            # time.sleep(1)

            self.ctrl.attack(1)
            # self.atack_flag = False

    #过图成功返回True 否则返回False
    def mov_to_next_room(self,screen):
        t = time.time()
        ada_image = cv.adaptiveThreshold(cv.cvtColor(screen, cv.COLOR_BGR2GRAY), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 3)
        # cv.imshow('ada_image', ada_image)
        # cv.waitKey(1)
        if np.sum(ada_image) == 0:
            print('过图成功')
            self.last_env = self.curl_env
            self.adb.touch_end(0, 0)
            return True

        return False
    def locate_hero(self):
        screen,result = self.find_result()
        hero = self.find_one_tag(result,"hero")
        if hero is not None:
            self.param.hx = hero.xywh[0][0].item()
            self.param.hy = hero.xywh[0][1].item()
            print(f"当前英雄位置：x:{self.param.hx},y:{self.param.hy}")
    #获取预测结果
    def find_result(self):
        while True:
            time.sleep(0.01)
            screen = self.ctrl.adb.last_screen
            if screen is None:
                continue
            s = time.time()
            #verbose不显示日志
            result = self.yolo.predict(source=screen,verbose=False)
            #给标签的值有个暂存的地方，好后续遍历的时候找到
            if self.class_name is None:
                self.class_name = result[0].names
            # print(f'匹配耗时{int((time.time() - s) * 1000)} ms')
            self.display_image(screen, result)
            return screen, result

    #展示预测结果 v10是直接显示结果的
    def display_image(self, screen, result):
        if screen is None:
            return
        ratio = 2
        width = 896 * ratio
        height = 414 * ratio
        cv.namedWindow('screen', cv.WINDOW_NORMAL)
        cv.resizeWindow('screen', width, height)
        cv.imshow('screen', result[0].plot())
        cv.waitKey(1)

    #捡装备
    def pick_up_equipment(self):
        """
        捡装备
        :return:
        """
        # 检查装备的次数
        check_cnt = 0
        hero_no = 0
        while True:
            time.sleep(0.3)
            screen, result = self.find_result()
            hero = self.find_tag(result, 'hero')
            if len(hero) == 0:
                hero_no += 1
                print(f'捡装备 没有找到英雄,{hero_no}次，暂时不做处理。')
                if hero_no > 5:
                    hero_no = 0
                    self.no_hero_handle(result)
                continue

            monster = self.find_tag(result, ['monster', 'boss'])
            if len(monster) > 0:
                print('找到怪物，或者发现卡片，停止捡装备。。')
                return
            hero = hero[0]
            hx, hy = get_detect_obj_bottom(hero)

            equipment = self.find_tag(result, 'item')
            if len(equipment) > 0:
                print('找到装备数量：', len(equipment))
                self.move_to_target(equipment, hero, hx, hy, screen)

            else:
                # 没有装备就跳出去
                check_cnt += 1
                if check_cnt >= 5:
                    print(f'没有装备，停止移动。当前移动状态：{self.param.mov_start}')
                    if self.param.mov_start:
                        self.param.mov_start = False
                        self.adb.touch_end(0, 0)
                    return
                print(f'没有找到装备:{check_cnt} 次。。。')
                continue


    def get_cur_room_index(self):
        """
        获取当前房间的索引，需要看地图
        :return:
        """
        route_map = None
        result = None
        fail_cnt = 0
        while True:
            self.ctrl.click(2105, 128)
            time.sleep(0.3)
            screen = self.ctrl.adb.last_screen
            if screen is None:
                continue
            start_time = time.time()
            result = self.yolo(screen)
            print(f'匹配地图点耗时：{(time.time() - start_time) * 1000}ms...')
            self.display_image(screen, result)
            # route_map = self.find_one_tag(result, 'map')
            # if route_map is None:
            route_map = self.find_one_tag(result, 'blue_hero')
            if route_map is not None:
                break
            else:
                fail_cnt += 1
                time.sleep(0.05)
                if fail_cnt > 8:
                    print('*******************************地图识别失败*******************************')
                    return None, None, None

        if route_map is not None:
            # 关闭地图
            tmp = self.find_one_tag(self.yolo(self.ctrl.adb.last_screen), 'blue_hero')
            if tmp is not None:
                self.ctrl.click(2105, 128)
            point = self.find_one_tag(result, 'blue_hero')
            if point is None:
                return None, None, None
            # 转换成中心点的坐标
            # point = get_detect_obj_center(point)
            #yolov10的处理
            point = (point.xywh[0][0].item(),point.xywh[0][1].item())
            route_id, cur_room = room_calutil.get_cur_room_index(point)
            return route_id, cur_room, point

        return None, None, None

    def move_to_next_room(self):
        """
        过图
        :return:
        """
        # 下一个房间的方向
        direction = None
        # mov_start = False
        lax, lay = 0, 0  # 英雄上次循环的坐标
        move_door_cnt = 0
        hero_no = 0
        no_door_cnt = 0
        while True:
            screen, result = self.find_result()
            # 2 判断是否过图成功
            ada_image = cv.adaptiveThreshold(cv.cvtColor(screen, cv.COLOR_BGR2GRAY), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv.THRESH_BINARY_INV, 13, 3)
            if np.sum(ada_image) <= 600000:
                print('*******************************过图成功*******************************')
                self.param.mov_start = False
                self.adb.touch_end(0, 0)
                self.param.cur_room = self.param.next_room
                return
            # 如果有怪物和装备，就停止过图 只有有装备的时候才停止过图
            if len(self.find_tag(result, [ 'monster','item'])) > 0:
                print('有怪物或装备，停止过图')
                self.param.mov_start = False
                self.adb.touch_end(0, 0)
                return

            # 1 先确定要行走的方向
            if direction is None:
                route_id, cur_room, point = self.get_cur_room_index()
                if route_id is None and cur_room is None:
                    print('没有找到地图和当前房间')
                    return
                elif route_id is None and cur_room is not None:
                    next_room = room_calutil.get_recent_room(cur_room)
                else:
                    self.param.cur_route_id = max(route_id, self.param.cur_route_id)
                    next_route_id, next_room = room_calutil.get_next_room(point, self.param.is_succ_sztroom)
                if next_room is None:
                    print('没有找到下一个房间')
                    return
                self.param.cur_room = cur_room
                self.param.next_room = next_room

                #防止卡图 上面第二个图做特殊处理
                if cur_room == (0,1):
                    cur_room ==(2,1)
                    next_room =(2,2)
                if cur_room == (1, 1):
                    self.param.is_succ_sztroom = True
                direction = room_calutil.get_run_direction(cur_room, next_room)

                mx, my = self.ctrl.calc_move_point_direction(direction)
                self.move_to_xy(mx, my)

            else:
                # 按方向走起来
                mx, my = self.ctrl.calc_move_point_direction(direction)
                self.move_to_xy(mx, my)

            print(f'当前所在房间id：{self.param.cur_route_id},方向：{direction}，当前是否移动：{self.param.mov_start}')

            hero = None
            # 3 先找到英雄位置，在找到对应方向的门进入
            hero = self.find_one_tag(result, 'hero')
            if hero is None:
                hero_no += 1
                print(f'过图中，没有找到英雄,{hero_no}次。')
                # mov_start = False
                # self.adb.touch_end(0, 0)
                if hero_no > 5:
                    hero_no = 0
                    # self.move()
                    self.no_hero_handle(result)
                continue

            hx, hy = get_detect_obj_bottom(hero)
            diff = abs(hx - lax) + abs(hy - lay)
            # 如果数据没什么变化，说明卡墙了
            lax, lay = hx, hy
            print(f'正在过图：英雄位置：{hx},{hy}，与上次的位置变化值：{diff}...')

            # 4 按照对应方向找对应的门
            doortag = room_calutil.get_tag_by_direction(direction)

            door = self.find_tag(result, doortag)
            arrow = self.find_tag(result, 'arrow')

            # if diff < 20 and len(go) > 0:
            #     print('如果数据没什么变化，说明卡墙了，移动到图中间')
            #     mov_start = self.move_to_target(go,hero, hx, hy, mov_start, screen)
            if len(door) > 0:
                self.move_to_target(door, hero, hx, hy, screen)
                # self.move_to_target(door, hero, self.param.hx, self.param.hy, screen)
                time.sleep(0.05)
                # if diff < 20 and len(go) > 0:
                #     print('如果数据没什么变化，说明卡墙了，移动到图中间')
                #     mov_start = self.move_to_target(go, hero, hx, hy, mov_start, screen)
                continue
            else:
                no_door_cnt+=1
                # self.no_hero_handle()
                print(f'没有找到方向门，继续找 次数：｛{no_door_cnt}')

            if no_door_cnt == 30:
                print('没找到方向门按箭头过图')
                #初始化然后按箭头过图
                if len(arrow)>0:
                    self.move_to_target(arrow, hero, self.param.hx, self.param.hy, screen)
                no_door_cnt = 0

            move_door_cnt += 1
            if move_door_cnt > 20:
                print(f'***************过门次数超过{move_door_cnt}次，随机移动一下*******************************')
                self.no_hero_handle(result)
                move_door_cnt = 0


    def move_to_target(self, target: list, hero, hx, hy, screen):
        min_distance_obj = min(target, key=lambda a: distance_detect_object(hero, a))
        cur_hx = hx
        cur_hy = hy
        # 改为直接计算坐标 非常依赖英雄位置的更新 可以新开一线程一直更新英雄的位置
        # min_distance_obj = min(target, key=lambda a: distance(self.param.hx,self.param.hy, a.xywh[0][0].item(),a.xywh[0][1].item()))
        ax, ay = get_detect_obj_bottom(min_distance_obj)

        if self.class_name[int(min_distance_obj.cls[0].item())] == 'opendoor_l':
            ax, ay = get_detect_obj_right(min_distance_obj)
        # 装备标了名称，所以要加40，实际上在下方
        if self.class_name[int(min_distance_obj.cls[0].item())] == 'item':
            ay += 60
        self.craw_line(cur_hx, cur_hy, ax, ay, screen)

        angle = calc_angle(cur_hx, cur_hy, ax, ay)
        # 根据角度计算移动的点击点
        sx, sy = self.ctrl.calc_mov_point(angle)
        # self.ctrl.click(sx, sy, 0.1)
        self.move_to_xy(sx, sy)


    def no_hero_handle(self,result=None, t = 0.8):
        """
        找不到英雄或卡墙了，随机移动，攻击几下
        :param result:
        :param t:
        :return:
        """
        # angle = (self.param.next_angle % 4) * 45 + random.randrange(start=-15, stop=15)
        # print(f'正在随机移动。。。随机角度移动{angle}度。')
        # self.param.next_angle = (self.param.next_angle + 1) % 4
        # sx, sy = self.ctrl.calc_mov_point(angle)
        self.param.mov_start = False
        #卡住了 随机移动
        if self.atack_flag :
          self.ctrl.skill_near()
          self.ctrl.attack(1)
        self.ctrl.move(random.randint(0,360),0.7)

        # self.move_to_xy(sx, sy)


    def move_to_xy(self, x, y, out_time=1.2):
        """
        移动到指定位置,默认2秒超时
        :param x:
        :param y:
        :return:
        """
        if (time.time() - self.param.move_time_out) >= out_time:
            self.param.move_time_out = time.time()
            self.param.mov_start = False
        if not self.param.mov_start:
            self.adb.touch_end(x, y)
            self.adb.touch_start(x, y)
            self.param.mov_start = True
            self.adb.touch_move(x-1, y)
            # 临时加的 因为总有卡死不动的情况 怀疑是这个未结束点击导致的
            #self.adb.touch_end(x, y)
        else:
            self.adb.touch_move(x, y)


    def attack_master(self):
        """
        找到怪物，攻击怪物
        :return:
        """
        attak_cnt = 0
        check_cnt = 0
        jx_cnt = 0
        no_hero_cnt = 0
        print(f'开始攻击怪物,当前房间：{self.param.cur_route_id}')

        cmds = RoomSkillUtil.get_cmd(self.param.cur_route_id)
        if len(cmds) > 0:
            for cmdTmp in cmds:
                cmd = cmdTmp[1]
                match cmd[0]:
                  case Command.MOVE:
                    self.move_to_xy(cmd[1][0],cmd[1][1])
                  case Command.SKILL:
                    self.ctrl.touch_skill_idx(cmd[1],Role.NAIMA)
        #先放觉醒 如果是狮子头
        if self.param.cur_room == (1, 1) and jx_cnt == 0:
            jx_cnt += 1
            self.ctrl.touchSkillJx()

        #开启buff
        if not self.param.skill_buff_start:
            # time.sleep(10)
            self.ctrl.skillBuff()
            self.param.skill_buff_start = True
        while True:
            # 找地图上包含的元素
            screen, result = self.find_result()
            card = self.find_tag(result, ['card'])
            if len(card) > 0:
                print('找到翻牌的卡片，不攻击')
                self.atack_flag = False
                return
            #先攻击了 不移动到怪物面前
            # self.atack()
            hero = self.find_tag(result, 'hero')
            #找不到英雄先攻击
            if len(hero) == 0:
                no_hero_cnt+=1
                # print(f'没有找到英雄,攻击{no_hero_cnt}次')
                if no_hero_cnt%5 == 0:
                    print(f'攻击状态，没有找到英雄,攻击1次')
                    self.ctrl.attack(0.3)
                if no_hero_cnt>20:
                    print(f'没有找到英雄,随机移动攻击')
                    self.no_hero_handle(result)
                    no_hero_cnt = 0
                continue

            hero = hero[0]
            hx, hy = get_detect_obj_bottom(hero)
            # cv.circle(screen, (hx, hy), 5, (0, 0, 125), 5)
            # 开启一次性的技能 todo 只有阿修罗才需要
            # if self.param.skill_start is not True:
            #     self.param.skill_start = True
            #     self.ctrl.skill_right()
            # 有怪物，就攻击怪物
            monster = self.find_tag(result, ['monster', 'boss'])
            if len(monster) > 0:
                print('怪物数量：', len(monster))

                # 最近距离的怪物坐标
                nearest_monster = min(monster, key=lambda a: distance_detect_object(hero, a))
                distance = distance_detect_object(hero, nearest_monster)
                ax, ay = get_detect_obj_bottom(nearest_monster)
                # 判断在一条直线上再攻击
                y_dis = abs(ay - hy)
                print(f'最近距离的怪物坐标：{ax},{ay},距离：{distance},y距离：{y_dis}')

                if distance <= 600 * room_calutil.zoom_ratio and y_dis <= 100 * room_calutil.zoom_ratio:
                    self.adb.touch_end(ax, ay)
                    self.param.mov_start = False
                    # 面向敌人
                    angle = calc_angle(hx, hy, ax, hy)
                    self.ctrl.move(angle, 0.3)
                    print(f'====================敌人与我的角度{angle}==攻击怪物，攻击次数：{attak_cnt},{self.param.cur_room}')
                    attak_cnt += 1
                    # 释放连招
                    self.atack()

                    if jx_cnt > 0:
                        #如果已经放了觉醒 再点一次 用来放觉醒要点两次的
                        self.ctrl.touchSkillJx()
                    # self.ctrl.continuous_attack_axl(attak_cnt)
                    # self.ctrl.kuangzhan_skill(attak_cnt)

                # ax, ay = get_detect_obj_center(nearest_monster)
                # 怪物在右边,就走到怪物走边400的距离
                if ax > hx:
                    ax = int(ax - 500 * room_calutil.zoom_ratio)
                else:
                    ax = int(ax + 500 * room_calutil.zoom_ratio)
                self.craw_line(hx, hy, ax, ay, screen)
                angle = calc_angle(hx, hy, ax, ay)
                sx, sy = self.ctrl.calc_mov_point(angle)
                # self.param.mov_start = False
                self.move_to_xy(sx, sy, 1)


            else:
                check_cnt += 1
                #没找到怪物也攻击
                self.ctrl.attack(0.5)
                if check_cnt >= 5:
                    print(f'没有找到怪物:{check_cnt}次。。。')
                    return





    def craw_line(self, hx, hy, ax, ay, screen):
        # cv.circle(screen, (hx, hy), 5, (0, 0, 125), 5)
        # 计算需要移动到的的坐标
        cv.circle(screen, (hx, hy), 5, (0, 255, 0), 5)
        cv.circle(screen, (ax, ay), 5, (0, 255, 255), 5)
        cv.arrowedLine(screen, (hx, hy), (ax, ay), (255, 0, 0), 3)
        cv.imshow('screen', screen)
        cv.waitKey(1)


    def find_tag(self, result, tag):
        """
        根据标签名称来找到目标
        :param result:
        :param tag:
        :return: list 判断len大于0则是找到了数据
        """
        # hero = [x for x in result if self.yolo.class_names[int(x.label)] in tag]
        hero = [x for x in result[0].boxes if result[0].names[int(x.cls.item())] in tag]
        return hero


    def find_one_tag(self, result, tag):
        """
        根据标签名称来找到目标
        :param result:
        :param tag:
        :return:
        """
        reslist =  [x for x in result[0].boxes if result[0].names[int(x.cls.item())] == tag]

        if len(reslist) == 0:
            print(f'没有找到标签{tag}')
            return None
        else:
            return reslist[0]


    def reset_start_game(self):
        """
        重置游戏，回到初始状态
        :return:
        """
        while True:
            screen, result = self.find_result()

            card = self.find_tag(result, 'card')
            select = self.find_tag(result, 'select')
            start = self.find_tag(result, 'start')


            if len(select) > 0:
                print("发现选择框按扭，开始点击")

                self.ctrl.click(294, 313)
                time.sleep(0.5)
                self.ctrl.click(1640, 834)
                return
            elif len(start) > 0:
                print("发现战斗开始按扭，开始点击")
                time.sleep(0.5)
                self.ctrl.click(1889, 917)
                return
            elif len(card) > 0:
                time.sleep(3)
                # 翻第三个牌子
                self.ctrl.click(1398, 377)
                time.sleep(0.5)
                self.ctrl.click(1398, 377)
                time.sleep(3)


                #翻完牌后 看有没有装备 有的话就先捡
                screen, result = self.find_result()
                if self.find_one_tag(result,'item') is not None:
                    self.pick_up_equipment()

                self.start_next_game_v1()

                # 点击重新挑战 ,指定区域进行模版匹配
                # crop = (1856, 108, 304, 304)
                # crop = tuple(int(value * room_calutil.zoom_ratio) for value in crop)
                # template_img = cv.imread('./template/again.jpg')
                # result = image_match_util.match_template_best(template_img, self.ctrl.adb.last_screen, crop)
                # while result is None:
                #     time.sleep(1)
                #     print('找再次挑战按钮')
                #     # screen, result = self.find_result()
                #     result = image_match_util.match_template_best(template_img, self.ctrl.adb.last_screen, crop)
                #     # self.find_result()
                # x, y, w, h = result['rect']
                # self.ctrl.click((x + w / 2) / self.ctrl.adb.zoom_ratio, (y + h / 2) / self.ctrl.adb.zoom_ratio)
                # # self.ctrl.click(2014,151)
                # time.sleep(0.8)
                # self.ctrl.click(1304, 691)
                # 出现卡片，就是打完了，初始化数值
                # self.param = GameParamVO()

                return
            else:
                return

    def start_next_game_v1(self):
        # 翻牌
        self.ctrl.adb.touch_start(10, 10)
        time.sleep(0.1)
        self.ctrl.adb.touch_end(0, 0)
        time.sleep(0.2)
        self.ctrl.adb.touch_start(0, 0)
        time.sleep(0.1)
        self.ctrl.adb.touch_end(11, 11)

        # 翻完牌后 看有没有装备 有的话就先捡
        screen, result = self.find_result()
        if self.find_one_tag(result, 'item') is not None:
            self.pick_up_equipment()

        time.sleep(5)
        self.ctrl.startNextGame()
        time.sleep(5)
        self.param = GameParamVO()

    def find_text(self, screen, tag):
        """
        根据标签名称来找到目标
        :param result:
        :param tag:
        :return: list 判断len大于0则是找到了数据
        """
        result = self.ocr.readtext(screen)

        if len(result) > 0:
            hero = [x for x in result if tag in x[1]]
            return hero
        else:
            return None
    def start_next_game_ocr(self):
        # #匹配重新挑战
        while True:
            time.sleep(3)
            # 翻完牌后 看有没有装备 有的话就先捡

            screen = self.ctrl.adb.last_screen
            again_result = self.find_text(screen, '再次挑战地下城')
            if again_result is not None and len(again_result) > 0:
                screen, result = self.find_result()
                if self.find_one_tag(result, 'item') is not None:
                    self.pick_up_equipment()

                print("点击再次挑战按扭")
                self.ctrl.startNextGame()
                time.sleep(5)
                self.param = GameParamVO()
    def start_next_game(self):
        # #匹配重新挑战
        while True:
            time.sleep(5)
            # 点击重新挑战 ,指定区域进行模版匹配
            crop = (1856, 108, 304, 304)
            crop = tuple(int(value * room_calutil.zoom_ratio) for value in crop)
            template_img = cv.imread('./template/again.jpg')
            result = image_match_util.match_template_best(template_img, self.ctrl.adb.last_screen, crop)
            while result is None:
                time.sleep(3)
                print('找再次挑战按钮')
                # screen, result = self.find_result()
                result = image_match_util.match_template_best(template_img, self.ctrl.adb.last_screen, crop)
                # self.find_result()
            x, y, w, h = result['rect']
            print("点击再次挑战按扭")
            self.ctrl.startNextGame()
            time.sleep(5)
            self.param = GameParamVO()
            # self.ctrl.click((x + w / 2) / self.ctrl.adb.zoom_ratio, (y + h / 2) / self.ctrl.adb.zoom_ratio)
def run():
    ctrl = GameControl(ScrcpyADB())
    action = GameAction(ctrl)

    # action.start_game()
    mov_start = False
    # 20张图就重新再过走到中间再过
    sleep_frame = 20
    # 开始游戏 查看当前是在哪地方
    #新线程匹配图
    # thread = threading.Thread(target=action.start_next_game)
    # thread.start()

    #新线程定位英雄
    # thread = threading.Thread(target=action.locate_hero)
    # thread.start()
    none_cnt = 0
    while True:
        time.sleep(0.2)
        # screen = action.ctrl.adb.last_screen
        # if screen is None:
        #     continue
        # result = None
        # if action.atack_flag or action.move_flag:
        #     print("攻击或移动中不处理图像")
        #     continue
        # 新线程匹配图
        thread = threading.Thread(target=action.start_next_game)
        thread.start()
        try:
            screen, result = action.find_result()

            # cv.namedWindow('screen', cv.WINDOW_NORMAL)
            # cv.resizeWindow('screen', 800, 600)
            # cv.imshow('screen', result[0].plot())
            # cv.waitKey(1)
            # print(result)

            # 根据出现的元素分配动作
            if len(action.find_tag(result, 'item')) > 0:
                print('--------------------------------发现装备，开始捡起装备--------------------------------')
                action.pick_up_equipment()
                action.param.mov_start = False
                continue

            if len(action.find_tag(result, ['arrow','lion_gate',  'opendoor_d', 'opendoor_r', 'opendoor_u',
                                            'opendoor_l'])) > 0:
                print('--------------------------------发现门，开始移动到下一个房间--------------------------------')
                action.move_to_next_room()
                action.param.mov_start = False
                continue
            if len(action.find_tag(result, 'card')) > 0:
                # 打完就先结束了
                print('打完了，去翻牌子')
                action.reset_start_game()
                # break
                continue
            if len(action.find_tag(result, ['monster', 'boss'])) > 0:
                print('--------------------------------发现怪物，开始攻击--------------------------------')
                action.atack_flag = True
                action.attack_master()
                action.param.mov_start = False
                action.atack_flag = False
                continue
            if len(action.find_tag(result, ['select', 'start'])) > 0:
                print('--------------------------------发现选择框，开始选择--------------------------------')
                action.reset_start_game()
                continue

            none_cnt +=1
            if none_cnt == 30:
                print("-----------------------------------无任何匹配30次随机移动-----------------------------------------------")
                action.no_hero_handle()
                none_cnt = 0
        except Exception as e:
            action.param.mov_start = False
            print(f'出现异常:{e}')
            traceback.print_exc()
if __name__ == '__main__':
    ctrl = GameControl(ScrcpyADB())
    action = GameAction(ctrl)

    # action.ctrl.move(20, 1)
    #
    # action.ctrl.click(2105, 128)
    # action.move
    # action.start_game()