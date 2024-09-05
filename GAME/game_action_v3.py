import logging
import threading
import traceback
from typing import Tuple

from dnfm.game.game_control  import GameControl
from dnfm.adb.scrcpy_adb import ScrcpyADB
from dnfm.game.game_command import  Command
from dnfm.game.label import Label
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
        #有些日志不想看到 暂时和无屏蔽掉
        logging.basicConfig(level=logging.DEBUG)

        self.ctrl = ctrl
        # self.yolo = YOLOv10("D:\\repository\\git\\yolov10-main\\runs\detect\\train_v1034\\weights\\best.pt")
        self.yolo = YOLOv10("..\\..\\runs\\detect\\train_v1039\\weights\\best.pt")

        self.adb = self.ctrl.adb
        self.sleep_count = 0

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

        self.last_screen_result = None

        self.param = GameParamVO()
    def show_result(self,screen):
        # while True:
            try:
                # time.sleep(0.2)
                if screen is not None:
                    cv.namedWindow('screen', cv.WINDOW_NORMAL)
                    cv.resizeWindow('screen', 896, 414)
                    cv.imshow('screen',screen)
                    cv.waitKey(1)
            except Exception as e:
                action.param.mov_start = False
                print(f'出现异常:{e}')
                traceback.print_exc()

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
    def random_move(self):
        print("卡死中，随机移动1秒 ")
        # hero_x = 10
        # hero_y = 10
        # moveto_x = random.randint(1, 20)
        # moveto_y = random.randint(1, 20)
        # moveAngle = calc_angle(hero_x, hero_y, moveto_x, moveto_y)
        moveAngle = random.randint(0, 360)
        self.ctrl.move(moveAngle, 0.7)
    def start_game(self):

        mov_start = False
        # 20张图就重新再过走到中间再过
        sleep_frame = 300
        #开始游戏 查看当前是在哪地方

        # thread = threading.Thread(target=self.start_next_game)
        # thread.start()

        #新开线程来展示结果
        # thread_show = threading.Thread(target=self.show_result)
        # thread_show.start()

        while True:
            time.sleep(0.1)
            self.last_screen_result = None
            screen = self.ctrl.adb.last_screen
            if screen is None:
                continue
            # if self.atack_flag or self.move_flag:
            #     print("攻击或移动中不处理图像")
            # else:
            result = self.yolo.predict(screen)


            # print(result)


            # 处理图片结果 并做出相应的反应
            # 1、获取当前角色的位置
            # 2、有怪物就先移动 并攻击怪物
            # 3、如果没怪物有物品就先去捡物品
            # 4、如果没有物品 有箭头和门后 就移动到箭头和门旁边。如果在狮子头旁边就先移动狮子头房间，并攻击狮子头
            if result is None:
                continue

            self.last_screen_result =  result[0].plot()
            self.show_result(self.last_screen_result)
            # 防止卡死
            if self.sleep_count == sleep_frame:
                self.random_move()

                self.sleep_count = 0
                continue


            cmd_arry = self.get_cur_order(result)
            print("cmd order", cmd_arry)
            if len(cmd_arry) == 0:
                self.sleep_count +=1
                continue
            self.sleep_count = 0
            #只执行指令集中按枚举排序 匹配的第一个指令
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

                            if not self.param.skill_buff_start:
                                #只有在英雄时 才加buff
                                if len(self.find_tag(result,['hero']))>0:
                                    # print("开始加buff")
                                    # time.sleep(5)
                                    self.ctrl.skillBuff()
                                    self.param.skill_buff_start = True

                            self.atack()
                            time.sleep(0.2)
                          case Command.MOVE:
                            if (self.hero_x == 0 and self.hero_y == 0):
                                # 找不到英雄时要随机移动加判断
                                self.sleep_count += 1
                                continue
                            self.move()
                            # 移动的时候才判断是否过完图
                            # self.mov_to_next_room(screen)
                          case Command.AGAIN:
                              # self.start_next_game_v1()
                              thread = threading.Thread(target=self.start_next_game_v1)
                              thread.start()

                          case _:
                            self.atack()
                            self.sleep_count += 1
                        #执行完后跳出循环
                        break





    def start_next_game(self):
        # #匹配重新挑战
        while True:
            # 点击重新挑战 ,指定区域进行模版匹配
            crop = (1856, 108, 304, 304)
            crop = tuple(int(value * room_calutil.zoom_ratio) for value in crop)
            template_img = cv.imread('../../../Desktop/dd-help/dnfm/template/again.jpg')
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

    def start_next_game_v1(self):
        time.sleep(0.2)
        # 翻牌
        self.ctrl.adb.touch_start(10, 10)
        time.sleep(0.1)
        self.ctrl.adb.touch_end(0, 0)
        time.sleep(0.2)
        self.ctrl.adb.touch_start(0, 0)
        time.sleep(0.1)
        self.ctrl.adb.touch_end(11, 11)

        time.sleep(10)
        self.ctrl.startNextGame()
        time.sleep(5)
        self.param = GameParamVO()
    #0.8移动时算比较流畅 试试0.6
    def move(self,t: float=0.6):
        self.move_flag = True
        moveAngle = calc_angle(self.hero_x, self.hero_y, self.moveto_x, self.moveto_y)
        cur_hx_hy_sum = self.hero_x+self.hero_y

        self.craw_line(self.hero_x,self.hero_y,self.moveto_x,self.moveto_y,self.last_screen_result)
        #防止卡死随机移动
        if abs(self.param.last_hx_hy_sum-cur_hx_hy_sum) < 30:
            self.param.not_move_cnt+=1

        if self.param.not_move_cnt > 10:
           print(f'未移动次数：{self.param.not_move_cnt}，随机移动1秒')
           self.param.not_move_cnt = 0
           self.random_move()

        self.param.last_hx_hy_sum = cur_hx_hy_sum
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
                            # self.hero_x = box.xywh[0][0].item()
                            # self.hero_y = box.xywh[0][1].item()
                            #获取中心点
                            self.hero_x,self.hero_y = get_detect_obj_center(box)
                        continue
                    elif Label.bwj_room5.value == itemTmp or Label.arrow.value == itemTmp or Label.lion_gate.value == itemTmp or Label.item.value == itemTmp or Label.next_gate.value == itemTmp:
                        if Label.bwj_room5.value == itemTmp:
                            print("on bwj_room 5")
                            # 狮子头优先级在过门之前 如果这个移动没有用 是没找到狮子头房间在布万家5房间再做次判断
                            if lion_gate_flag:
                                continue
                            # 只有在未进入过狮子头时地和进入  不然会死循环，
                            # 在未进入狮子头时 向左侧移动 坐标写死
                            if not self.param.is_succ_sztroom:
                                rtn_arry.append(
                                    Command.MOVE
                                )
                                lion_gate_flag = True
                                self.moveto_x = 282
                                self.moveto_y = 516
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
                                #获得中心点
                                self.moveto_x,self.moveto_y = get_detect_obj_center(box)
                                # 修正物品标的名字 y+60
                                self.moveto_y+=60
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

                        #箭头移动优先级最后面
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
                    #狮子头 如果没进过，则优先进狮子头，判断逻辑 判断lion_flag =  True.
                    elif Label.monster.value == itemTmp or Label.boss.value == itemTmp:
                        if (len(box.xywh[0]) > 0):
                            # self.moveto_x = box.xywh[0][0].item()
                            # self.moveto_y = box.xywh[0][1].item()
                            # 获得中心点
                            self.moveto_x, self.moveto_y = get_detect_obj_center(box)
                        rtn_arry.append(Command.ATACK)
                    #有卡片的时候做再次挑战的处理
                    elif Label.card.value == itemTmp:

                        rtn_arry.append(Command.AGAIN)
                    #布万家前4图
                    elif Label.bwj_room1.value == itemTmp or Label.bwj_room2.value == itemTmp or Label.bwj_room3.value == itemTmp or Label.bwj_room4.value == itemTmp:
                        #当前在的地图
                        self.curl_env = itemTmp
                    elif   Label.bwj_room6.value == itemTmp or Label.bwj_room7.value == itemTmp or Label.bwj_room8.value == itemTmp or Label.bwj_room9.value == itemTmp :
                        #当前在的地图
                        self.curl_env = itemTmp


                        if Label.bwj_room6.value == itemTmp:
                            print("on bwj_room 6 狮子头")
                           #在狮子头房间 进去后 改为已经进入过狮子头  并放觉醒
                            if not self.param.is_succ_sztroom :
                               self.param.is_succ_sztroom = True
                               self.ctrl.touchSkillJx()
                               self.param.jx_cnt += 1
                        #Boss 房 把已经 进过狮子头重置
                        elif Label.bwj_room9.value == itemTmp:
                            print("on bwj_room 9 boss房间")
        return rtn_arry

    #攻击
    def atack(self):
        #攻击
        if self.atack_flag:
            sleep_time = 0.5
            print("start atack")
            self.move(0.3)
            self.ctrl.randonSkillRole(self.param.role)
            time.sleep(sleep_time)
            self.ctrl.randonSkillRole(self.param.role)
            time.sleep(sleep_time)
            self.ctrl.randonSkillRole(self.param.role)
            time.sleep(sleep_time)
            self.ctrl.attack(2)
            self.atack_flag = False
            #觉醒要按两次的补偿机制
            if self.param.jx_cnt > 1 and self.param.jx_cnt < 8:
                self.ctrl.touchSkillJx()
                self.param.jx_cnt += 1

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


    def craw_line(self, hx, hy, ax, ay, screen):
        # cv.circle(screen, (hx, hy), 5, (0, 0, 125), 5)
        # 计算需要移动到的的坐标
        cv.circle(screen, (hx, hy), 5, (0, 255, 0), 5)
        cv.circle(screen, (ax, ay), 5, (0, 255, 255), 5)
        cv.arrowedLine(screen, (hx, hy), (ax, ay), (255, 0, 0), 3)
        cv.imshow('screen', screen)
        cv.waitKey(1)






if __name__ == '__main__':
    ctrl = GameControl(ScrcpyADB())
    action = GameAction(ctrl)



    action.start_game()