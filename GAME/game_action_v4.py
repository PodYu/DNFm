import logging
import threading
import traceback
from typing import Tuple

from PIL import Image

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

from dnfm.game.skill import Skill
from dnfm.utils import room_calutil
from dnfm.utils.cvmatch import image_match_util
from dnfm.utils.room_skill_util import RoomSkillUtil
from dnfm.vo.game_main_vo import GameMainVo
from dnfm.vo.game_param_vo import GameParamVO
from ultralytics import YOLOv10
import easyocr


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
        self.mainVo = GameMainVo()
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
        self.curl_env = Label.bwj_room1.value
        #上一个场景
        self.last_env = ""

        self.hero_x = 0
        self.hero_y = 0
        self.moveto_x = 0
        self.moveto_y = 0

        self.last_screen_result = None
        #ocr暂时用的easyocr 会比paddleocr快
        self.ocr = easyocr.Reader(['ch_sim'])

        self.param = GameParamVO()
        self.room_skill_util = RoomSkillUtil()
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
        hero = [x for x in result[0].boxes if result[0].names[int(x.cls.item())] in tag]
        return hero
    def random_move(self):
        print("卡死中，随机移动1秒 ")
        moveAngle = random.randint(0, 360)
        self.ctrl.move(moveAngle, 0.7)
    def start_game(self):

        mov_start = False
        # 20张图就重新再过走到中间再过
        sleep_frame = 50
        #开始游戏 查看当前是在哪地方


        while True:
            time.sleep(0.1)
            self.last_screen_result = None
            screen = self.ctrl.adb.last_screen
            if screen is None:
                continue
            # if self.atack_flag or self.move_flag:
            #     print("攻击或移动中不处理图像")
            # else:
            result = self.yolo.predict(source=screen,verbose=False)


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
                #先停止移动 再重新移动 防止卡死
                self.ctrl.end_move()
                self.random_move()

                self.sleep_count = 0
                continue


            cmd_arry = self.get_cur_order(result)

            #指令为空时处理
            # print("cmd order", cmd_arry)
            if len(cmd_arry) == 0:
                self.sleep_count +=1
                continue
            self.sleep_count = 0

            #处理指令
            if  self.process_order(cmd_arry,result):
                return True

    #执行指令，当程序结束时返回 True 否则返回 False
    def process_order(self,cmd_arry,result):
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

                            # if not self.param.skill_buff_start:
                            #     #只有在英雄时 才加buff
                            #     if len(self.find_tag(result,['hero']))>0:
                            #         # print("开始加buff")
                            #         # time.sleep(5)
                            #         self.ctrl.skillBuff()
                            #         self.param.skill_buff_start = True
                            self.ctrl.end_move()
                            self.atack()
                            # time.sleep(0.2)
                          case Command.MOVE:
                            if (self.hero_x == 0 and self.hero_y == 0):
                                # 找不到英雄时要随机移动加判断
                                self.sleep_count += 1
                                continue
                            self.move()
                          case Command.AGAIN:
                            #如果下一步返回结果 True 则结束游戏 说明已经刷完了
                            return self.start_next_game_v1()

                          case _:
                            # self.atack()
                            self.sleep_count += 1
                        #执行完后跳出循环
                        break
            return False



    

    def start_next_game_v1(self):
        
        # 翻牌
        self.ctrl.adb.touch_start(10, 10,3)
        time.sleep(0.1)
        self.ctrl.adb.touch_end(0, 0,3)
        time.sleep(0.2)
        self.ctrl.adb.touch_start(0, 0,3)
        time.sleep(0.1)
        self.ctrl.adb.touch_end(11, 11,3)
        time.sleep(2)

        #物品检测次数 如果大于30次则不检测
        item_detect_cnt = 0
        while True:
            if item_detect_cnt > 30:
                break
            #1秒钟看5下 检测物品
            time.sleep(0.2)
            #游戏翻牌后的处理 主要看有没有装备
            result = self.yolo.predict(source = self.ctrl.adb.last_screen,verbose=False)

            cmd_arry = self.get_cur_order(result)
            move_cmd_arry = []
            #在翻过牌后只处理移动的指令
            for cmdTmp in cmd_arry:
                if cmdTmp == Command.MOVE:
                    move_cmd_arry.append(cmdTmp)
                    break
            # print("cmd move order", move_cmd_arry)
            if len(move_cmd_arry) == 0:
                item_detect_cnt +=1
                continue
            
            # 处理图片结果 并做出相应的反应
            # 1、获取当前角色的位置
            # 2、有怪物就先移动 并攻击怪物
            # 3、如果没怪物有物品就先去捡物品
            # 4、如果没有物品 有箭头和门后 就移动到箭头和门旁边。如果在狮子头旁边就先移动狮子头房间，并攻击狮子头
            self.process_order(move_cmd_arry,result)

            if result is None:
                continue

            self.last_screen_result =  result[0].plot()
            self.show_result(self.last_screen_result)

        #找再次挑战按扭 如果找到则返回为 True 找不到则返回False 暂时先一直找吧
        if self.find_again_text():
            if self.find_role_pl() == 0:
                #角色疲劳为空 结束
                print(f"角色{self.mainVo.role} 疲劳为0 结束当前游戏 返回城镇")
                self.ctrl.returnCity()
                return False
            #重新开始游戏
            self.ctrl.end_move()
            self.ctrl.startNextGame()
            time.sleep(5)
            self.param = GameParamVO()
    #0.8移动时算比较流畅 试试0.6
    def move(self,t: float=0.7):
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
           self.ctrl.end_move()
           self.random_move()

        self.param.last_hx_hy_sum = cur_hx_hy_sum
        # print("hero x,y:", self.hero_x, self.hero_y)
        # print("move to:", self.moveto_x, self.moveto_y)
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

        #一直攻击
        #不能一直攻击
        monster_cnt = 0
        while True:
            time.sleep(0.1)
            #攻击
            #按房间来走配置 只第一次执行 后续不执行
            if self.param.skill_cnt[self.curl_env] == 0:
                self.bwj_room_logic()
                return

            result = self.yolo.predict(source=self.ctrl.adb.last_screen,verbose=False)
            if len(result) > 0:
                self.show_result(result[0].plot())
            #刷新命令，主要是识别当前房间的信息
            cmd_array = self.get_cur_order(result)
            #如果有停止的条件 则不再攻击
            tags = self.find_tag(result,["opendoor_l","opendoor_u","arrow","next_gate","lion_gate","card"])
            if len(tags) > 0:
                print("找到门，不再攻击")
                return

            tags = self.find_tag(result, ["monster", "boss"])
            if len(tags) == 0:
                print(f"没找到怪物,次数:{monster_cnt}")
                monster_cnt+=1
                #没找到怪物10次 就重新攻击一下
                if(monster_cnt % 10 == 0):
                    self.move(0.3)
                    self.ctrl.randonSkill()
                    # time.sleep(sleep_time)
                    # self.ctrl.randonSkill()
                    # time.sleep(sleep_time)
                    # self.ctrl.randonSkill()
                    time.sleep(0.2)
                    self.ctrl.attack(1)
                continue

            monster_cnt = 0
            # if self.atack_flag:
            sleep_time = 0.2
            print("开始攻击")
            self.move(0.3)
            self.ctrl.randonSkill()
            # time.sleep(sleep_time)
            # self.ctrl.randonSkill()
            # time.sleep(sleep_time)
            # self.ctrl.randonSkill()
            time.sleep(sleep_time)
            self.ctrl.attack(1)
            self.atack_flag = False
                #觉醒要按两次的补偿机制
                # if self.param.jx_cnt > 1 and self.param.jx_cnt < 8:
                #     self.ctrl.touchSkillJx()
                #     self.param.jx_cnt += 1

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
        #找一次得有1秒钟 找10次就是10秒
        again_cnt = 0
        #1秒钟处理
        while True:
            time.sleep(0.1)
            print(f'找再次挑战，次数：{again_cnt}')
            img_screen = Image.fromarray(self.ctrl.adb.last_screen)
            region = img_screen.crop((1740, 80, 2110, 494))
            again_img_region_path = 'again_img_region_path.jpg'
            region.save(again_img_region_path)
            result = self.ocr.readtext(again_img_region_path)
            if (self.find_text(result,['再次挑战地下城'])):
                return True

    '''
        find_text(reuslt,['再次挑战地下城','返回城镇'])
    '''
    def find_text(self,result, tag):
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
            print(f'找当前角色疲劳，次数：{again_cnt}')
            if again_cnt >10:
                again_cnt+=1
                return -1
            img_screen = Image.fromarray(self.ctrl.adb.last_screen)
            region = img_screen.crop((278, 88, 420, 124))
            again_img_region_path = 'pl_img_region_path.jpg'
            region.save(again_img_region_path)
            result = self.ocr.readtext(again_img_region_path)
            if len(result) > 0:
                if result[0][1].find('/') > 0:
                    pls = result[0][1].split('/')
                    print(f'当前疲劳:{pls[0]}')
                    if pls[0] == '@' or pls[0] == 'o' :
                        self.mainVo.pl = 0
                        return 0
                    return int(pls[0])


if __name__ == '__main__':
    ctrl = GameControl(ScrcpyADB())
    action = GameAction(ctrl)



    action.start_game()