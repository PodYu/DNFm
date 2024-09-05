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
from dnfm.utils import room_calutil
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


def distance(x1, y1, x2, y2):
    dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return dist


def calc_angle(x1, y1, x2, y2):
    angle = math.atan2(y1 - y2, x1 - x2)
    return 180 - int(angle * 180 / math.pi)


class GameAction:
    def __init__(self, ctrl: GameControl,infer_queue):

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

        self.last_screen_result = None
        # ocr暂时用的easyocr 会比paddleocr快
        self.ocr = easyocr.Reader(['ch_sim'])

        self.param = GameParamVO()
        self.room_skill_util = RoomSkillUtil()
        #有门的时候的一些操作
        self.find_door = False

        thread = threading.Thread(target=self.start_game,name="action")  # 创建线程，并指定目标函数
        # thread = threading.Thread(target=self.select_game,name="action")  # 创建线程，并指定目标函数
        thread.daemon = True  # 设置为守护线程（可选）
        thread.start()

    def show_result(self, screen):
        # while True:
        try:
            # time.sleep(0.2)
            if screen is not None:
                cv.namedWindow('screen', cv.WINDOW_NORMAL)
                cv.resizeWindow('screen', 896, 414)
                cv.imshow('screen', screen)
                cv.waitKey(1)
        except Exception as e:
            action.param.mov_start = False
            self.logger.info(f'出现异常:{e}')
            traceback.self.logger.info_exc()

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
        self.logger.info("卡死中，随机移动1秒 ")
        moveAngle = random.randint(0, 360)
        self.ctrl.move(moveAngle, 0.7)

    def start_game(self):
        self.logger.info("开始处理布万家逻辑")
        mov_start = False
        # 20张图就重新再过走到中间再过
        sleep_frame = 50
        # 开始游戏 查看当前是在哪地方

        while True:
            # time.sleep(0.1)
            if self.infer_queue.empty():
                time.sleep(0.001)
                continue
            result = self.find_result()

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

                            ada_image = cv.adaptiveThreshold(cv.cvtColor(self.find_result()[0].plot(), cv.COLOR_BGR2GRAY), 255,
                                                             cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 3)
                            # cv.imshow('ada_image', ada_image)
                            # cv.waitKey(1)
                            if np.sum(ada_image) == 0:
                                self.logger.info('过图成功,停止移动')
                                self.last_env = self.curl_env
                                #过图房间 预设
                                match self.curl_env:
                                    case Label.bwj_room1.value:
                                        self.curl_env=Label.bwj_room2.value
                                    case Label.bwj_room2.value:
                                        self.curl_env=Label.bwj_room3.value
                                    case Label.bwj_room3.value:
                                        self.curl_env=Label.bwj_room4.value
                                    case Label.bwj_room4.value:
                                        self.curl_env=Label.bwj_room5.value
                                    case Label.bwj_room5.value:
                                        if self.param.is_succ_sztroom:
                                            self.curl_env = Label.bwj_room7.value
                                        #不设置狮子头，狮子头会自动放觉醒，走错了会导致出问题，狮子头用识别的逻辑
                                        # else:
                                        #     self.curl_env=Label.bwj_room6.value
                                    case Label.bwj_room6.value:
                                        self.curl_env=Label.bwj_room5.value
                                    case Label.bwj_room7.value:
                                        self.curl_env=Label.bwj_room8.value
                                    case Label.bwj_room8.value:
                                        self.curl_env=Label.bwj_room9.value
                                    case _:
                                        pass
                                self.logger.info(f'当前房间:{self.curl_env} 上一个房间:{self.last_env}')
                                self.ctrl.end_move()
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
            if item_detect_cnt > 10:
                self.logger.info(f'翻牌后物品检测大于:{item_detect_cnt}次，后续不处理。准备处理打完副本后逻辑')
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

            if result is None:
                continue

            # self.last_screen_result = result[0].plot()
            # self.show_result(self.last_screen_result)
        self.ctrl.end_move()
        # 找再次挑战按扭 如果找到则返回为 True 找不到则返回False 暂时先一直找吧
        if self.find_again_text():
            cur_pl = self.find_role_pl()
            #现在基本上是疲劳为0时匹配不到 返回-1 所以把-1的也当成是疲劳为0
            if  cur_pl == 0 or cur_pl == -1 :
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

    # 0.8移动时算比较流畅 试试0.6
    def move(self, t: float = 0.7):
        self.move_flag = True
        moveAngle = calc_angle(self.hero_x, self.hero_y, self.moveto_x, self.moveto_y)
        cur_hx_hy_sum = self.hero_x + self.hero_y

        # self.craw_line(self.hero_x, self.hero_y, self.moveto_x, self.moveto_y, self.last_screen_result)
        # 防止卡死随机移动
        if abs(self.param.last_hx_hy_sum - cur_hx_hy_sum) < 100:
            self.param.not_move_cnt += 1

        if self.param.not_move_cnt > 30:
            self.logger.info(f'未移动次数：{self.param.not_move_cnt}，随机移动1秒')
            self.param.not_move_cnt = 0
            self.ctrl.end_move()
            self.random_move()

        self.param.last_hx_hy_sum = cur_hx_hy_sum
        # self.logger.info("hero x,y:", self.hero_x, self.hero_y)
        # self.logger.info("move to:", self.moveto_x, self.moveto_y)
        self.logger.info(f"calc move angle:{moveAngle}" )

        self.ctrl.move(moveAngle, t)

        self.move_flag = False

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
                                self.param.sztroom_move_cnt+=1
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
                                self.moveto_y += 60
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
                        # 当前在的地图
                        self.curl_env = itemTmp
                    elif Label.bwj_room6.value == itemTmp or Label.bwj_room7.value == itemTmp or Label.bwj_room8.value == itemTmp or Label.bwj_room9.value == itemTmp:
                        # 当前在的地图
                        self.curl_env = itemTmp

                        if Label.bwj_room6.value == itemTmp:
                            # self.logger.info("on bwj_room 6 狮子头")
                            # 在狮子头房间 进去后 改为已经进入过狮子头  并放觉醒
                            if not self.param.is_succ_sztroom:
                                self.param.is_succ_sztroom = True
                                self.ctrl.touchSkillJx()
                                self.param.jx_cnt += 1
                        # Boss 房 把已经 进过狮子头重置
                        elif Label.bwj_room9.value == itemTmp:

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
            # time.sleep(0.1)
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
                # 没找到怪物10次 就重新攻击一下
                if (monster_cnt % 30 == 0):
                    self.logger.info(f"没找到怪物,次数:{monster_cnt}，开始攻击一次")
                    self.move(0.1)
                    self.ctrl.attack(1)
                    self.ctrl.randonSkill()

                continue

            monster_cnt = 0
            sleep_time = 0.1
            # self.logger.info("开始攻击，先攻击一秒，再放个技能")
            self.move(sleep_time)
            self.ctrl.attack(1)
            self.ctrl.randonSkill()


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
            result = self.ocr.readtext(again_img_region_path)
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
        # 找一次得有1秒钟 找10次就是10秒
        again_cnt = 0
        # 1秒钟处理
        while True:
            time.sleep(0.1)
            self.logger.info(f'找当前角色疲劳，次数：{again_cnt}')

            if again_cnt > 20:
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
        return self.ocr.readtext(img)

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
                print(select_role_list)
                #0是坐标 4个角的 [([[161, 141], [219, 141], [219, 177], [161, 177]], '选角', 0.7816692970423834)]
                #取正中心点
                x = (select_role_list[0][0][0][0]+select_role_list[0][0][0][1])/2
                y = (select_role_list[0][0][0][1]+select_role_list[0][0][3][1])/2
                self.ctrl.click(x,y)
                self.logger.info(f'找到:{text},点击完了,找寻次数：{find_select_role_cnt}')
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
        #找寻次数 30次以后 就不找
        find_cnt_limit = 30
        find_select_role_cnt = 0
        while True:
            time.sleep(0.1)
            find_select_role_cnt+=1
            ocr_result =  self.find_ocr_result(self.find_result()[0].plot())
            select_role_list = self.find_text(ocr_result, ["选角"])
            if len(select_role_list) > 0:
                print(select_role_list)
                #0是坐标 4个角的 [([[161, 141], [219, 141], [219, 177], [161, 177]], '选角', 0.7816692970423834)]
                #取正中心点
                x = (select_role_list[0][0][0][0]+select_role_list[0][0][0][1])/2
                y = (select_role_list[0][0][0][1]+select_role_list[0][0][3][1])/2
                self.ctrl.click(x,y)
                self.logger.info(f'找到选角按扭,点击完了,找寻次数：{find_select_role_cnt}')
                find_role_cnt = 0

                while True:
                    find_role_cnt +=1
                    time.sleep(0.3)
                    ocr_result = self.find_ocr_result(self.find_result()[0].plot())
                    print(ocr_result)
                    select_role_list = self.find_text(ocr_result, [role_name])
                    if len(select_role_list) > 0:
                        print(select_role_list)
                        x = (select_role_list[0][0][0][0] + select_role_list[0][0][0][1]) / 2
                        y = (select_role_list[0][0][0][1] + select_role_list[0][0][3][1]) / 2
                        time.sleep(1)
                        self.ctrl.click(x, y)
                        self.logger.info(f'找到角色:{role_name},点击完了,找寻次数：{find_role_cnt}')
                        return True
                    if find_role_cnt > find_cnt_limit:
                        self.ctrl.click(0,0)
                        self.logger.info(f'未找到角色:{role_name},结束,找寻次数：{find_role_cnt}')
                        return False
            if find_select_role_cnt>find_cnt_limit:
                self.logger.info(f'未找到选角按扭,结束,找寻次数：{find_select_role_cnt}')
                return False
if __name__ == '__main__':
    ctrl = GameControl(ScrcpyADB())
    action = GameAction(ctrl)

    action.select_game()
    # action.select_game()