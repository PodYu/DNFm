import logging
import threading
import traceback
from typing import Tuple

from PIL import Image

from dnfm.game.game_control import GameControl
from dnfm.adb.scrcpy_adb import ScrcpyADB
from dnfm.game.game_command import Command
from dnfm.game.label import Label
from dnfm.vo.game_cfg_vo import GameCfgVo
import time
import cv2 as cv
from ncnn.utils.objects import Detect_Object
import math
import random
import numpy as np

from dnfm.game.skill import Skill
from dnfm.utils import room_calutil, ocr_util
from dnfm.utils.cvmatch import image_match_util
from dnfm.utils.room_skill_util import RoomSkillUtil
from dnfm.vo.game_main_vo import GameMainVo
from dnfm.vo.game_param_vo import GameParamVO
from ultralytics import YOLOv10
import easyocr
from dnfm.utils.log_util import logger

def get_detect_obj_right(obj: Detect_Object) -> Tuple[int, int]:
    return int(obj.xywh[0][0].item() + obj.xywh[0][2].item()), int(
        obj.xywh[0][1].item() + obj.xywh[0][3].item() / 2)
    # return int(obj.rect.x + obj.rect.w), int(obj.rect.y + obj.rect.h/2)


def get_detect_obj_center(obj: Detect_Object) -> Tuple[int, int]:
    return int(obj.xywh[0][0].item() + obj.xywh[0][2].item() / 2), int(
        obj.xywh[0][1].item() + obj.xywh[0][3].item() / 2)
    # return int(obj.rect.x + obj.rect.w/2), int(obj.rect.y + obj.rect.h/2)


def get_detect_obj_bottom(obj: Detect_Object) -> Tuple[int, int]:
    return int(obj.xywh[0][0].item() + obj.xywh[0][2].item() / 2), int(obj.xywh[0][1].item() + obj.xywh[0][3].item())
    # return int(obj.rect.x + obj.rect.w / 2), int(obj.rect.y + obj.rect.h)


def distance_detect_object(a: Detect_Object, b: Detect_Object):
    return math.sqrt(
        (a.xywh[0][0].item() - b.xywh[0][0].item()) ** 2 + (a.xywh[0][1].item() - b.xywh[0][1].item()) ** 2)


def find_close_point_to_box(boxes, hero_x,hero_y):
    min_distance = None
    closest_box = None# 遍历所有box，找出最近的box
    for box in boxes:
        tmpx,tmpy = get_detect_obj_bottom(box)
        distanceTmp = distance(tmpx, tmpy,hero_x,hero_y)
        if min_distance is None:
            min_distance = distanceTmp
            closest_box = box
        elif distanceTmp < min_distance:
            min_distance = distanceTmp
            closest_box = box
    return closest_box,min_distance

#找最远的
def find_farthest_point_to_box(boxes, hero_x,hero_y):
    max_distance = None
    farthest_box = None# 遍历所有box，找出最近的box
    for box in boxes:
        tmpx,tmpy = get_detect_obj_center(box)
        distanceTmp = distance(tmpx, tmpy,hero_x,hero_y)
        if max_distance is None:
            max_distance = distanceTmp
            farthest_box = box
        elif distanceTmp > max_distance:
            max_distance = distanceTmp
            farthest_box = box
    return farthest_box,max_distance

def distance(x1, y1, x2, y2):
    dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return dist


def calc_angle(x1, y1, x2, y2):
    angle = math.atan2(y1 - y2, x1 - x2)
    return 180 - int(angle * 180 / math.pi)

bwj_map_cfg=[
    (Label.bwj_room1.value, 'room1.jpg', (2000, 100, 58, 58)),
    (Label.bwj_room2.value, 'room2.jpg', (2000, 100, 58, 58)),
    (Label.bwj_room3.value, 'room3.jpg', (1942, 42, 58, 58)),
    (Label.bwj_room4.value, 'room4.jpg', (2000, 100, 58, 58)),
    (Label.bwj_room5.value, 'room5.jpg', (2000, 158, 58, 58)),
    (Label.bwj_room6.value, 'room6.jpg', (1942, 100, 58, 58)),
    (Label.bwj_room7.value, 'room7.jpg', (1942, 158, 58, 58)),
    (Label.bwj_room8.value, 'room8.jpg', (2000, 100, 58, 58)),
    (Label.bwj_room9.value, 'room9.jpg', (2000, 100, 58, 58))
]
class GameAction:
    def __init__(self, ctrl: GameControl,infer_queue):
        #启动时 是停止状态 要点run才开始
        self.stop_event = True
        #从日志工具类中获取 已经配置好了的
        self.logger = logger
        

        self.infer_queue = infer_queue

        # 有些日志不想看到 暂时和无屏蔽掉
        self.mainVo = GameMainVo()
        # logging.basicConfig(level=logging.DEBUG)

        self.ctrl = ctrl
        # self.yolo = YOLOv10("D:\\repository\\git\\yolov10-main\\runs\detect\\train_v1034\\weights\\best.pt")
        # self.yolo = YOLOv10("..\\..\\runs\\detect\\train_v1039\\weights\\best.pt")

        self.adb = self.ctrl.adb
        self.sleep_count = 0

        # 默认是没进过布万家的狮子头 进过之后要改成True 打完Boss又改为False
        self.bwj_szt_entryed_flag = False
        self.atack_flag = False
        self.move_flag = False
        # 当前场景
        self.curl_env = Label.bwj_room1.value
        # 上一个场景
        self.last_env = ""

        self.hero_x = 0
        self.hero_y = 0
        self.moveto_x = 0
        self.moveto_y = 0

        #用来控制过图时改变地图索引,当为True时 要加地图的判断
        self.is_add_room_num = True

        self.last_screen_result = None

        self.param = GameParamVO()
        self.room_skill_util = RoomSkillUtil()
        #有门的时候的一些操作
        self.find_door = False
        self.stop_event = True
        self.thread_run = True
        thread = threading.Thread(target=self.start_game,name="action")  # 创建线程，并指定目标函数
        # thread = threading.Thread(target=self.select_game,name="action")  # 创建线程，并指定目标函数
        thread.daemon = True  # 设置为守护线程（可选）
        thread.start()

    def reset(self):
        #重置的时候 让线程跑完 再起新线程
        self.thread_run = False
        time.sleep(1)

        self.stop_event = True
        self.param = GameParamVO()
        # 用来控制过图时改变地图索引,当为True时 要加地图的判断
        self.is_add_room_num = True

        self.thread_run = True
        thread = threading.Thread(target=self.start_game, name="action")  # 创建线程，并指定目标函数
        thread.daemon = True  # 设置为守护线程（可选）
        thread.start()

    def find_tag(self, result, tag):
        """
        根据标签名称来找到目标
        :param result:
        :param tag:
        :return: list 判断len大于0则是找到了数据
        """
        hero = [x for x in result[0].boxes if result[0].names[int(x.cls.item())] in tag]
        return hero

    def random_move(self):

        move_angle_cfg=[91,11,181,358]
        moveAngle = move_angle_cfg[random.randint(0, 3)]
        self.logger.info(f"卡死中，随机移动1秒 角度: {moveAngle}")
        self.move_angle(moveAngle,1)

    def start_game(self):
        self.logger.info("开始处理布万家逻辑")
        mov_start = False
        # 20张图就重新再过走到中间再过
        sleep_frame = 50
        # 开始游戏 查看当前是在哪地方

        while self.thread_run:
            if self.stop_event:
                self.ctrl.end_move()
                time.sleep(1)
                continue

            # time.sleep(0.1)
            if self.infer_queue.empty():
                time.sleep(0.001)
                continue
            result = self.find_result()

            #过图中不做任何操作
            if self.move_to_next_room(result):
                continue
            self.get_bwj_map(result)
            # self.logger.info(result)

            # 处理图片结果 并做出相应的反应
            # 1、获取当前角色的位置
            # 2、有怪物就先移动 并攻击怪物
            # 3、如果没怪物有物品就先去捡物品
            # 4、如果没有物品 有箭头和门后 就移动到箭头和门旁边。如果在狮子头旁边就先移动狮子头房间，并攻击狮子头
            if result is None:
                continue

            # 防止卡死
            if self.sleep_count == sleep_frame:
                # 先停止移动 再重新移动 防止卡死
                self.ctrl.end_move()
                self.random_move()

                self.sleep_count = 0
                continue

            cmd_arry = self.get_cur_order(result)

            # 指令为空时处理
            # self.logger.info("cmd order", cmd_arry)
            if len(cmd_arry) == 0:
                self.sleep_count += 1
                continue
            self.sleep_count = 0

            # 处理指令
            if self.process_order(cmd_arry, result):
                return True

    # 执行指令，当程序结束时返回 True 否则返回 False
    def process_order(self, cmd_arry, result):
        # 只执行指令集中按枚举排序 匹配的第一个指令
        break_flag = False
        for cmd in Command:
            if break_flag:
                break
            for cur_cmd in cmd_arry:
                if cur_cmd == cmd:
                    # 只执行指令集中按枚举排序 匹配的第一个指令
                    break_flag = True
                    match cur_cmd:
                        case Command.ATACK:
                            self.atack_flag = True

                            # if not self.param.skill_buff_start:
                            #     #只有在英雄时 才加buff
                            #     if len(self.find_tag(result,['hero']))>0:
                            #         # self.logger.info("开始加buff")
                            #         # time.sleep(5)
                            #         self.ctrl.skillBuff()
                            #         self.param.skill_buff_start = True
                            self.ctrl.end_move()
                            self.atack()
                            self.ctrl.end_move()

                            self.atack_flag = False
                            # time.sleep(0.2)
                        case Command.MOVE:
                            if (self.hero_x == 0 and self.hero_y == 0):
                                # 找不到英雄时要随机移动加判断
                                self.sleep_count += 1
                                continue
                            self.move()


                                # return True
                        case Command.AGAIN:
                            # 如果下一步返回结果 True 则结束游戏 说明已经刷完了
                            return self.start_next_game_v1()

                        case _:
                            # self.atack()
                            self.sleep_count += 1
                    # 执行完后跳出循环
                    break
        return False

    def start_next_game_v1(self):
        self.ctrl.end_move()
        self.logger.info("**************开始翻牌**************")
        self.logger.info(f"翻牌前 上一次点击事件 {self.ctrl.adb.last_click_para}")

        # 翻牌
        self.ctrl.click(500,20)
        time.sleep(0.2)
        self.ctrl.click(400, 100)
        time.sleep(0.5)

        time.sleep(5)
        # 物品检测次数 如果大于30次则不检测
        item_detect_cnt = 0
        while True:
            if item_detect_cnt > 20:
                self.logger.info(f'翻牌后物品检测大于:{item_detect_cnt}次，后续不处理。准备处理打完副本后逻辑')
                self.ctrl.end_move()
                break
            # 1秒钟看5下 检测物品
            time.sleep(0.2)
            # 游戏翻牌后的处理 主要看有没有装备
            result = self.find_result()

            cmd_arry = self.get_cur_order(result)
            move_cmd_arry = []
            # 在翻过牌后只处理移动的指令
            for cmdTmp in cmd_arry:
                if cmdTmp == Command.MOVE:
                    move_cmd_arry.append(cmdTmp)
                    break
            # self.logger.info("cmd move order", move_cmd_arry)
            if len(move_cmd_arry) == 0:
                item_detect_cnt += 1
                continue

            # 处理图片结果 并做出相应的反应
            # 1、获取当前角色的位置
            # 2、有怪物就先移动 并攻击怪物
            # 3、如果没怪物有物品就先去捡物品
            # 4、如果没有物品 有箭头和门后 就移动到箭头和门旁边。如果在狮子头旁边就先移动狮子头房间，并攻击狮子头
            self.process_order(move_cmd_arry, result)


            # self.last_screen_result = result[0].plot()
            # self.show_result(self.last_screen_result)
        self.ctrl.end_move()
        # 找再次挑战按扭 如果找到则返回为 True 找不到则返回False 暂时先一直找吧
        if self.find_again_text():
            cur_pl = self.find_role_pl()
            #现在基本上是疲劳为0时匹配不到 返回-1 所以把-1的也当成是疲劳为0
            #暂时先暂停pl判断
            if  cur_pl == 0 :
                # 角色疲劳为空 结束
                self.logger.info(f"角色{self.mainVo.role} 疲劳为0 结束当前游戏 返回城镇")
                #因为直接点击返回城镇总是点不到
                self.find_screen_text_click("返回城镇")
                # self.ctrl.returnCity()
                return True
            self.mainVo.bwj_cnt += 1
            #把当前副本的挑战时间加进去
            cur_cost_time = time.time() - self.mainVo.cur_start_time
            self.mainVo.fb_cost_arr.append(cur_cost_time)
            self.mainVo.cur_start_time = time.time()
            self.logger.info(f'当前角色:{self.mainVo.role},疲劳:{self.mainVo.pl},布家挑战次数:{self.mainVo.bwj_cnt},当前挑战时间:{cur_cost_time},平均挑战时间:{np.mean(self.mainVo.fb_cost_arr)}')
            # 重新开始游戏

            self.ctrl.startNextGame()
            time.sleep(5)
            self.param = GameParamVO()
            self.curl_env = Label.bwj_room1.value

    '''
     移动指令的执行
     包含 物品 门
    '''
    def move(self, t: float = 0.7):
        # move_cnt = 0
        # while True:
            # if(move_cnt>30):
            #     self.random_move()
            result = self.find_result()




            monsters = self.find_tag(result, ["monster","boss"])
            if len(monsters)>0:
                self.logger.info("找到怪物，停止移动。")
                self.ctrl.end_move()
                return

            monsters = self.find_tag(result, ["card"])
            if len(monsters)>0:
                self.logger.info("找到卡，停止移动。")
                self.ctrl.end_move()
                return


            #更新英雄位置
            heros = self.find_tag(result, ["hero"])
            if len(heros)>0:
                self.hero_x, self.hero_y = get_detect_obj_center(heros[0])
            #判断英雄是否未移动
            self.is_hero_not_move()

            #狮子头额外添加逻辑，不然有时候会被arrow影响
            szt_door = self.find_tag(result,["lion_gate"])
            if Label.bwj_room5.value == self.curl_env and not self.param.is_succ_sztroom and len(szt_door) >0:

                clost_x, clost_y = get_detect_obj_center(szt_door[0])
                # 修饰一下
                clost_y -= 60
                clost_x -= 50
                angle = calc_angle(self.hero_x, self.hero_y, clost_x, clost_y)

                self.move_angle(angle)
                self.logger.info(f'布万家发现狮子头房间门，开始移动...角度:{angle}')
                # self.move_to_target(szt_door[0])
                time.sleep(1)
                return
            #如果在狮子头前一房间 且没发现门，只移动20次 如果20次没移动到狮子头 则不打狮子头了
            if Label.bwj_room5.value == self.curl_env and not self.param.is_succ_sztroom and self.param.sztroom_move_cnt <20 :
                self.param.sztroom_move_cnt += 1
                self.move_angle(170)
                self.logger.info(f"狮子特殊移动 次数{self.param.sztroom_move_cnt} 角度:150")
                time.sleep(0.1)
                return

            #找物品
            items = self.find_tag(result,["item"])
            if len(items)>0:
                # print(items)
                # self.logger.info(f'找到物品:{len(items)}个，英雄位置x:{self.hero_x},y:{self.hero_y}')
                #获取最近的物品 然后计算角度移动
                closest_box,min_distance = find_close_point_to_box(items,self.hero_x,self.hero_y)
                self.move_to_target(closest_box)
                #优先级算高的 直接返回
                return

            doors = self.find_tag(result,
                                 ["opendoor_l", "opendoor_u", "opendoor_b", "opendoor_r",  "next_gate",
                                  "lion_gate"])
            if len(doors)>0:
                #过门逻辑写死
                if Label.bwj_room1.value == self.curl_env:
                    door = self.find_tag(result,["opendoor_b"])
                    if len(door)>0:
                        self.logger.info(f'布万家1房间开始过图')
                        self.move_to_target(door[0])
                        return

                elif Label.bwj_room2.value == self.curl_env:
                    door = self.find_tag(result,["opendoor_r"])
                    if len(door)>0:
                        self.logger.info(f'布万家2房间开始过图')
                        self.move_to_target(door[0])
                        return

                elif Label.bwj_room3.value == self.curl_env:
                    door = self.find_tag(result,["opendoor_r"])
                    if len(door)>0:
                        self.logger.info(f'布万家3房间开始过图')
                        self.move_to_target(door[0])
                        return

                elif Label.bwj_room4.value == self.curl_env:
                    door = self.find_tag(result,["opendoor_u"])
                    if len(door)>0:
                        self.logger.info(f'布万家4房间开始过图')
                        self.move_to_target(door[0])
                        return

                elif Label.bwj_room5.value == self.curl_env:
                    door = self.find_tag(result, ["lion_gate","opendoor_l"])
                    #如果没找到狮子头的门 且过门次数小于20 往狮子头左边那边走
                    # if not self.param.is_succ_sztroom and self.param.sztroom_move_cnt <20  :
                    #     self.param.sztroom_move_cnt+=1
                    #     self.move_angle(170)
                    #     self.logger.info(f"狮子特殊移动 次数{self.param.sztroom_move_cnt} 角度:150")
                    #     time.sleep(0.1)
                    #     return
                    if self.param.is_succ_sztroom:
                        door = self.find_tag(result, ["opendoor_r"])
                    if len(door)>0:
                        self.logger.info(f'布万家5房间开始过图')
                        self.move_to_target(door[0])
                        return

                elif Label.bwj_room6.value == self.curl_env:
                    door = self.find_tag(result, ["opendoor_r"])
                    if len(door)>0:
                        self.logger.info(f'布万家6房间开始过图')
                        self.move_to_target(door[0])
                        return

                elif Label.bwj_room7.value == self.curl_env:
                    door = self.find_tag(result, ["opendoor_r"])
                    if len(door)>0:
                        self.logger.info(f'布万家7房间开始过图')
                        self.move_to_target(door[0])
                        return
                elif Label.bwj_room8.value == self.curl_env:
                    door = self.find_tag(result, ["opendoor_r"])
                    if len(door)>0:
                        self.logger.info(f'布万家8房间开始过图')
                        self.move_to_target(door[0])
                        return


            arrows = self.find_tag(result, ["arrow"])
            if len(arrows)>0:
                # print(arrows)

                # self.logger.info(f'使用箭头过图开始')
                farthest_box,far_distance =find_farthest_point_to_box(arrows,self.hero_x,self.hero_y)
                self.move_to_target(farthest_box)

    def is_hero_not_move(self):

        cur_hx_hy_sum = self.hero_x + self.hero_y
        if abs(self.param.last_hx_hy_sum - cur_hx_hy_sum) < 100:
            self.param.not_move_cnt += 1
        else:
            #只要移动过一次就算移动过 初始化一下
            self.param.not_move_cnt = 0
        if self.param.not_move_cnt > 30:
            self.logger.info(f'未移动次数：{self.param.not_move_cnt}，随机移动')
            self.param.not_move_cnt = 0
            self.ctrl.end_move()
            self.random_move()

        self.param.last_hx_hy_sum = cur_hx_hy_sum

    #过图逻辑
    def move_to_next_room(self,result):
        ada_image = cv.adaptiveThreshold(cv.cvtColor(result[0].plot(), cv.COLOR_BGR2GRAY), 255,
                                         cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 3)
        if np.sum(ada_image) == 0:
            if not self.is_add_room_num :
                self.ctrl.end_move()
                self.logger.info('过图中...')
                self.is_add_room_num = True
                self.param.is_find_cur_room = False
            return True

        if self.is_add_room_num:

            if len(self.find_tag(result,["hero"])) > 0:
                self.last_env = self.curl_env
                self.is_add_room_num = False

                # 过图房间 预设
                match self.curl_env:
                    case Label.bwj_room1.value:
                        self.curl_env = Label.bwj_room2.value
                    case Label.bwj_room2.value:
                        self.curl_env = Label.bwj_room3.value
                    case Label.bwj_room3.value:
                        self.curl_env = Label.bwj_room4.value
                    case Label.bwj_room4.value:
                        self.curl_env = Label.bwj_room5.value
                    case Label.bwj_room5.value:
                        if self.param.is_succ_sztroom:
                            self.curl_env = Label.bwj_room7.value
                        # 不设置狮子头，狮子头会自动放觉醒，走错了会导致出问题，狮子头用识别的逻辑
                        # else:
                        #     self.curl_env=Label.bwj_room6.value
                    case Label.bwj_room6.value:
                        self.curl_env = Label.bwj_room5.value
                    case Label.bwj_room7.value:
                        self.curl_env = Label.bwj_room8.value
                    case Label.bwj_room8.value:
                        self.curl_env = Label.bwj_room9.value
                    case _:
                        pass
                self.get_bwj_map(result)
                # self.logger.info(f'过图成功，当前房间:{self.curl_env} 上一个房间:{self.last_env}')
        return False

    def get_bwj_map(self, result):
        #这个地图九宫匹配 不精准 经常等技能放完了再匹配到
        # 如果当前房间已经找到了 则不做操作直接返回 None 过图后更改为False
        if not self.param.is_find_cur_room:
            cur_env_new = self.find_cur_map(bwj_map_cfg, result[0].plot(), 'bwj')
            if cur_env_new is not None:
                if self.last_env != cur_env_new:
                    self.logger.info(f"新房间检测，当前房间号：{cur_env_new}")
                    self.last_env = self.curl_env
                    self.curl_env = cur_env_new
                    self.param.is_find_cur_room = True

    #移动到检测到的box 传入box
    def move_to_target(self,target_box):
        clost_x, clost_y = get_detect_obj_center(target_box)
        angle = calc_angle(self.hero_x, self.hero_y, clost_x, clost_y)
        self.move_angle(angle)


    '''
     单纯的移动
    '''
    def move_angle(self, angle:int,t: float = 0):
        # moveAngle = calc_angle(self.hero_x, self.hero_y, self.moveto_x, self.moveto_y)
        #需要更新英雄位置 检测到时需要更新
        # cur_hx_hy_sum = self.hero_x + self.hero_y
        #
        # # self.craw_line(self.hero_x, self.hero_y, self.moveto_x, self.moveto_y, self.last_screen_result)
        # # 防止卡死随机移动
        # if abs(self.param.last_hx_hy_sum - cur_hx_hy_sum) < 100:
        #     self.param.not_move_cnt += 1
        #
        # if self.param.not_move_cnt > 30:
        #     self.logger.info(f'未移动次数：{self.param.not_move_cnt}，随机移动1秒')
        #     self.param.not_move_cnt = 0
        #     self.ctrl.end_move()
        #     self.random_move()
        #
        # self.param.last_hx_hy_sum = cur_hx_hy_sum
        # self.logger.info("hero x,y:", self.hero_x, self.hero_y)
        # self.logger.info("move to:", self.moveto_x, self.moveto_y)
        # self.logger.info(f"开始移动 angle:{angle},英雄位置x:{self.hero_x},y:{self.hero_y}" )

        self.ctrl.move(angle, t)

    '''
      获取当前的指令
      当前的指令定义只有2个 一个是攻击一个是移动
      如果没有指令则是卡死
      入参为预测的结果
      返回的是指令集 然后按照枚举的顺序 每次只执行一个指令
    '''

    def get_cur_order(self, result):

        rtn_arry = []

        # 要有个移动优先级 1、是物品 2、是箭头 3才是门 现在门没做具体标记不好按固定路线走
        item_label_flag = False
        arrow_label_flag = False
        next_gate_label = False
        # 狮子头
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
                            # self.hero_x = box.xywh[0][0].item()
                            # self.hero_y = box.xywh[0][1].item()
                            # 获取中心点
                            self.hero_x, self.hero_y = get_detect_obj_center(box)
                        continue
                    elif Label.bwj_room5.value == itemTmp or Label.arrow.value == itemTmp or Label.lion_gate.value == itemTmp or Label.item.value == itemTmp or Label.next_gate.value == itemTmp:
                        if Label.bwj_room5.value == itemTmp:
                            # self.logger.info("on bwj_room 5")
                            # 狮子头优先级在过门之前 如果这个移动没有用 是没找到狮子头房间在布万家5房间再做次判断
                            if lion_gate_flag:
                                continue
                            # 只有在未进入过狮子头时地和进入  不然会死循环，
                            # 在未进入狮子头时 向左侧移动 坐标写死 只进行10次吧
                            if not self.param.is_succ_sztroom and not self.atack_flag and self.param.sztroom_move_cnt <10:
                                self.logger.info(f'狮子头特殊移动次数：{self.param.sztroom_move_cnt}')
                                rtn_arry.append(
                                    Command.MOVE
                                )
                                lion_gate_flag = True
                                self.moveto_x = 282
                                self.moveto_y = 516
                                # self.param.sztroom_move_cnt+=1
                            continue
                        # 狮子头优先级在过门之前 如果这个移动没有用 是没找到狮子头房间在布万家5房间再做次判断
                        if lion_gate_flag:
                            continue
                        if Label.lion_gate.value == itemTmp:
                            # 只有在未进入过狮子头时地和进入  不然会死循环
                            if not self.param.is_succ_sztroom:
                                rtn_arry.append(
                                    Command.MOVE
                                )
                                lion_gate_flag = True
                                if (len(box.xywh[0]) > 0):
                                    # self.moveto_x = box.xywh[0][0].item()
                                    # self.moveto_y = box.xywh[0][1].item()
                                    # 获得中心点
                                    self.moveto_x, self.moveto_y = get_detect_obj_center(box)
                            continue
                        # 考虑优先级 物品优先级最高
                        if item_label_flag:
                            continue

                        if Label.item.value == itemTmp:
                            rtn_arry.append(
                                Command.MOVE
                            )
                            item_label_flag = True
                            if (len(box.xywh[0]) > 0):
                                # self.moveto_x = box.xywh[0][0].item()
                                # self.moveto_y = box.xywh[0][1].item()
                                # 获得中心点
                                self.moveto_x, self.moveto_y = get_detect_obj_center(box)
                                # 修正物品标的名字 y+60
                                self.moveto_y += 100
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
                                # self.moveto_x = box.xywh[0][0].item()
                                # self.moveto_y = box.xywh[0][1].item()
                                # 获得中心点
                                self.moveto_x, self.moveto_y = get_detect_obj_center(box)
                            continue

                        # 箭头移动优先级最后面
                        if Label.arrow.value == box.cls.item():
                            rtn_arry.append(
                                Command.MOVE
                            )
                            if (len(box.xywh[0]) > 0):
                                # self.moveto_x = box.xywh[0][0].item()
                                # self.moveto_y = box.xywh[0][1].item()
                                # 获得中心点
                                self.moveto_x, self.moveto_y = get_detect_obj_center(box)

                        # # 如果当前已经有物品或箭头了 先捡物品和移动
                        # if item_label_flag or arrow_label_flag or next_gate_label:
                        continue
                    # 狮子头 如果没进过，则优先进狮子头，判断逻辑 判断lion_flag =  True.
                    elif Label.monster.value == itemTmp or Label.boss.value == itemTmp:
                        if (len(box.xywh[0]) > 0):
                            # self.moveto_x = box.xywh[0][0].item()
                            # self.moveto_y = box.xywh[0][1].item()
                            # 获得中心点
                            self.moveto_x, self.moveto_y = get_detect_obj_center(box)
                        rtn_arry.append(Command.ATACK)
                    # 有卡片的时候做再次挑战的处理
                    elif Label.card.value == itemTmp:

                        rtn_arry.append(Command.AGAIN)
                    # 布万家前4图
                    elif Label.bwj_room1.value == itemTmp or Label.bwj_room2.value == itemTmp or Label.bwj_room3.value == itemTmp or Label.bwj_room4.value == itemTmp:
                        # 当前在的地图 用新方法获取，暂时先不用
                        self.curl_env = itemTmp
                        pass
                    elif Label.bwj_room6.value == itemTmp or Label.bwj_room7.value == itemTmp or Label.bwj_room8.value == itemTmp or Label.bwj_room9.value == itemTmp:
                        # 当前在的地图
                        self.curl_env = itemTmp

                        if Label.bwj_room6.value == itemTmp:
                            self.curl_env = itemTmp
                            # pass
                            #进图放觉醒取消 由配置放
                            # self.logger.info("on bwj_room 6 狮子头")
                            # 在狮子头房间 进去后 改为已经进入过狮子头  并放觉醒
                            if not self.param.is_succ_sztroom:
                                self.param.is_succ_sztroom = True
                            #     self.ctrl.touchSkillJx()
                            #     self.param.jx_cnt += 1
                        # Boss 房 把已经 进过狮子头重置
                        elif Label.bwj_room9.value == itemTmp:
                            self.curl_env = itemTmp
                            # self.logger.info("on bwj_room 9 boss房间")
                            pass
        return rtn_arry

    # 攻击
    def atack(self):
        #初始化副本开始挑战时间
        if self.mainVo.cur_start_time == 0:
            self.mainVo.cur_start_time = time.time()
        # 一直攻击
        # 不能一直攻击
        monster_cnt = 0
        while True:
            time.sleep(0.1)
            # 攻击
            # 按房间来走配置 只第一次执行 后续不执行
            if self.param.skill_cnt[self.curl_env] == 0:
                self.bwj_room_logic()
                continue

            result = self.find_result()
            # 刷新命令，主要是识别当前房间的信息
            cmd_array = self.get_cur_order(result)
            # 如果有停止的条件 则不再攻击
            tags = self.find_tag(result, ["opendoor_l", "opendoor_u","opendoor_b","opendoor_r", "arrow", "next_gate", "lion_gate", "card"])
            if len(tags) > 0:
                self.logger.info("找到门，不再攻击")
                self.atack_flag = False
                return

            tags = self.find_tag(result, ["monster", "boss"])
            if len(tags) == 0:

                monster_cnt += 1
                self.ctrl.end_move()
                # 没找到怪物10次 就重新攻击一下
                if (monster_cnt % 30 == 0):
                    self.logger.info(f"没找到怪物,次数:{monster_cnt}，开始攻击一次")
                    # self.move(0.1)
                    self.ctrl.attack(1)
                    self.ctrl.skill_free(self.mainVo.role)
                if monster_cnt >80:
                    self.logger.info(f"没找到怪物,次数:{monster_cnt}，结束攻击")
                continue

            monster_cnt = 0
            # if len(tags) > 0:
            closest_box, min_distance = find_close_point_to_box(tags, self.hero_x, self.hero_y)
            clost_x, clost_y = get_detect_obj_center(closest_box)
            angle = calc_angle(self.hero_x, self.hero_y, clost_x, clost_y)
            # self.logger.info(f'找到怪物:{len(tags)}个,最近怪物距离:{min_distance},x:{clost_x},y:{clost_y},heroX:{self.hero_x},hero_y:{self.hero_y},moveAngle:{angle}')
            # self.move_angle(angle)
            # self.ctrl.attack(1)
            # 不随机放技能 随机放技能会导致重要技能被放完
            # self.ctrl.skill_free(self.mainVo.role)
            #在同一条线上 且距离近 则开始攻击 可能导致不攻击
            atack_distance = 550
            atack_y_distance=100
            if min_distance <atack_distance and abs(clost_y-self.hero_y)<atack_y_distance :
                #面向怪物 怪物在左边 就向左 在右边就向右
                if self.hero_x > clost_x:
                    self.move_angle(180)
                elif self.hero_x < clost_x:
                    self.move_angle(1)
                self.ctrl.attack(1)
                # 不随机放技能 随机放技能会导致重要技能被放完
                self.ctrl.skill_free(self.mainVo.role)
            else:
                self.move_angle(angle)




    def bwj_room_logic(self):
        # 每个角色配置的一套操作，更具体的是要每个图每个角色配置一套 到时候在文件中再区分到每个图
        customer_skills = self.room_skill_util.get_skill_cfg(self.mainVo.role)
        match self.curl_env:
            case Label.bwj_room1.value:
                # 第一个房间
                customer_skills = customer_skills[0][1]
                self.process_customer_cmd(customer_skills)
            case Label.bwj_room2.value:

                # 第二个房间
                customer_skills = customer_skills[1][1]
                self.process_customer_cmd(customer_skills)
            case Label.bwj_room3.value:
                # 第三个房间
                customer_skills = customer_skills[2][1]
                self.process_customer_cmd(customer_skills)
            case Label.bwj_room4.value:

                # 第四个房间
                customer_skills = customer_skills[3][1]
                self.process_customer_cmd(customer_skills)
            case Label.bwj_room5.value:
                # 第五个房间
                customer_skills = customer_skills[4][1]
                self.process_customer_cmd(customer_skills)
            case Label.bwj_room6.value:

                # 第六个房间
                customer_skills = customer_skills[5][1]
                self.process_customer_cmd(customer_skills)
            case Label.bwj_room7.value:
                # 第七个房间
                customer_skills = customer_skills[6][1]
                self.process_customer_cmd(customer_skills)
            case Label.bwj_room8.value:

                # 第八个房间
                customer_skills = customer_skills[7][1]
                self.process_customer_cmd(customer_skills)
            case Label.bwj_room9.value:

                # 第九个房间
                customer_skills = customer_skills[8][1]
                self.process_customer_cmd(customer_skills)
        self.param.skill_cnt[self.curl_env] = 1

    '''
        执行配置的操作
        主要配置在room_skill_util
    '''

    def process_customer_cmd(self, customer_skills):
        for customer_skill in customer_skills:
            if customer_skill[0] == Command.MOVE:
                # 走多少角度 走多少时间
                self.ctrl.move(customer_skill[1], customer_skill[2])
            if customer_skill[0] == Command.SKILL:
                # 如果是技能指令  放哪个技能
                skills = Skill.get_role_skills(self.mainVo.role)
                self.ctrl.touch_skill(skills[customer_skill[1]])
            if customer_skill[0] == Command.SLEEP:
                # 如果是暂停 就要看传入暂停多久
                time.sleep(customer_skill[1])

    def craw_line(self, hx, hy, ax, ay, screen):
        # cv.circle(screen, (hx, hy), 5, (0, 0, 125), 5)
        # 计算需要移动到的的坐标
        cv.circle(screen, (hx, hy), 5, (0, 255, 0), 5)
        cv.circle(screen, (ax, ay), 5, (0, 255, 255), 5)
        cv.arrowedLine(screen, (hx, hy), (ax, ay), (255, 0, 0), 3)
        cv.imshow('screen', screen)
        cv.waitKey(1)

    '''
        拿到图片截取再次挑战地下城的那一部分
        找再次挑战文本
        在指定位置找
        找再次挑战文本的时候会顺便找到
        修理装备的文本
    '''

    def find_again_text(self):
        # 截取图片的一部分，这里的参数是(left, upper, right, lower)
        # 找一次得有1秒钟 找10次就是10秒
        again_cnt = 0
        # 1秒钟处理
        while True:
            time.sleep(0.1)
            self.logger.info(f'找再次挑战，次数：{again_cnt}')
            img_screen = Image.fromarray(self.find_result()[0].plot())
            region = img_screen.crop((1740, 80, 2110, 598))
            again_img_region_path = 'again_img_region_path.jpg'
            region.save(again_img_region_path)
            result = ocr_util.readtext(again_img_region_path)
            if (self.find_text(result, ['装备修理'])):
                #执行修理 先写死吧 不然是要识别好然后重新点击的
                self.logger.info("执行修理装备")
                self.ctrl.repair()
            if (self.find_text(result, ['再次挑战地下城'])):
                return True
            again_cnt+=1

    '''
        find_text(reuslt,['再次挑战地下城','返回城镇'])
    '''

    def find_text(self, result, tag):
        """
        根据标签名称来找到目标
        :param result:
        :param tag:
        :return: list 判断len大于0则是找到了数据
        """

        hero = [x for x in result if x[1] in tag]
        return hero

    '''
        ocr匹配当前角色的疲劳
        没找到疲劳 返回-1，其它返回正常的疲劳
        只找10次

    '''

    def find_role_pl(self):
        # 截取图片的一部分，这里的参数是(left, upper, right, lower)
        # 找一次得有1秒钟 找10次就是10秒 暂时找10次 20次也没意义 识别不了就是识别不了
        again_cnt = 0
        # 1秒钟处理
        while True:
            time.sleep(0.1)
            self.logger.info(f'找当前角色疲劳，次数：{again_cnt}')

            if again_cnt > 10:
                return -1
            img_screen = Image.fromarray(self.find_result()[0].plot())
            region = img_screen.crop((310, 88, 440, 124))
            again_img_region_path = 'pl_img_region_path.jpg'
            region.save(again_img_region_path)
            result = ocr_util.read_red_text(again_img_region_path)
            if len(result) > 0:
                self.logger.info(f'找疲劳结果:{result}')
                (center_x, center_y),pl_txt = result[0]
                if pl_txt.find('/') > 0:
                    pls = pl_txt.split('/')
                    self.logger.info(f'当前疲劳:{pls[0]}')
                    if pls[0] == '@' or pls[0] == 'o':
                        self.mainVo.pl = 0
                        return 0
                    self.mainVo.pl = int(pls[0])
                    return int(pls[0])
            again_cnt += 1
    #打完后找修理 界面 如果出现修理界面 就开始修理 找3次，暂时先不做跟着找再次挑战的逻辑里一起了。
    def find_repair(self):
        # 截取图片的一部分，这里的参数是(left, upper, right, lower)
        # 找一次得有1秒钟 找10次就是10秒
        again_cnt = 0
        # 1秒钟处理
        while True:
            # time.sleep(0.1)
            self.logger.info(f'找当前是否出现修理界面，次数：{again_cnt}')

            if again_cnt > 10:
                return -1
            img_screen = Image.fromarray(self.find_result()[0].plot())
            region = img_screen.crop((278, 88, 420, 124))
            again_img_region_path = 'pl_img_region_path.jpg'
            region.save(again_img_region_path)
            result = self.ocr.readtext(again_img_region_path)
            if len(result) > 0:
                if result[0][1].find('/') > 0:
                    pls = result[0][1].split('/')
                    self.logger.info(f'当前疲劳:{pls[0]}')
                    if pls[0] == '@' or pls[0] == 'o':
                        self.mainVo.pl = 0
                        return 0
                    return int(pls[0])
            again_cnt += 1
    #获取处理的结果
    def find_result(self):
        return self.infer_queue.get()
    def find_ocr_result(self,img):
        return ocr_util.readtext(img)

    '''
        找屏幕上的文字而且点击它
        找到返回True
        未找到返回False
    '''
    def find_screen_text_click(self, text):
        #找寻次数 30次以后 就不找
        find_cnt_limit = 30
        find_select_role_cnt = 0
        while True:
            time.sleep(0.1)
            find_select_role_cnt+=1
            ocr_result =  self.find_ocr_result(self.find_result()[0].plot())
            select_role_list = self.find_text(ocr_result, [text])
            if len(select_role_list) > 0:
                # print(select_role_list)
                (centen_x, centen_y),tmp_txt = select_role_list[0]
                self.ctrl.click(centen_x,centen_y)
                self.logger.info(f'找到:{text},x:{centen_x},y:{centen_y} 点击完了,找寻次数：{find_select_role_cnt}')
                return True

            if find_select_role_cnt>find_cnt_limit:
                self.logger.info(f'未找到{text},结束,找寻次数：{find_select_role_cnt}')
                return False
    def select_game(self):
        game_cfg = GameCfgVo()
        '''
        要在角色选择界面,或者在赛丽亚房间
        1,不管在哪，回到选择角色页面，好识别
        如果是在副本中 则返回城镇
        
        '''
        while True:
            #查看当前的顺序
            if self.mainVo.role_name is None:
                self.logger.info("当前角色为空，需要重新获取攻坚副本配置")
                for fbTmp in game_cfg.fb_cfg :
                    if self.select_role(fbTmp[0]):
                        print(fbTmp[1])
                        time.sleep(20)
                        pass
    '''
    选择指定角色 
    暂时没做找不到的判断
    '''
    def select_role(self, role_name):
        start = time.time()
        if self.find_screen_text_click("选角"):
            if self.find_screen_text_click(role_name):
                self.logger.info(f'找到角色:{role_name},点击完了,耗时{(time.time()-start)*1000}ms')
                return True
            else:
                self.logger.info(f'未找到角色：{role_name},结束,耗时{(time.time()-start)*1000}ms')
                return False
        self.logger.info(f'未找到选角按扭,结束,耗时{(time.time()-start)*1000}ms')
        return False
    '''
    模板匹配最新识别当前地图方法
    '''
    def find_cur_map(self,map_cfg, img, map_name):

        # find_cnt = 0
        # while find_cnt <20:
            #从main方法里进来则 直接是当前路径下了
        template_path = './template/map/'
        for cfg in map_cfg:
            img_search = cv.imread(f'{template_path}{map_name}/{cfg[1]}')
            result = image_match_util.match_template_best(img_search, img, cfg[2])
            if result is not None:
                # print(result)
                # print(cfg)
                # self.logger.info(f"找到当前地图:寻找次数{find_cnt}")
                return cfg[0]
        return None


if __name__ == '__main__':
    ctrl = GameControl(ScrcpyADB())
    action = GameAction(ctrl)

    action.select_game()
    # action.select_game()