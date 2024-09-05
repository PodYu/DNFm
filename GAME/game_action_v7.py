import inspect
import threading
import traceback

from PIL import Image

from dnfm.enums.game_map import GameMap
from dnfm.game.game_control import GameControl
from dnfm.adb.scrcpy_adb import ScrcpyADB
from dnfm.game.game_command import Command
from dnfm.game.label import Label

#地图导入 如果不导入则会找不到配置
from dnfm.game.map.map import Map
from dnfm.game.map.bwj import Bwj

from dnfm.game.screen import Screen
from dnfm.utils.detect_obj_util import find_farthest_point_to_box, get_detect_obj_center, find_close_point_to_box, \
    calc_angle, find_tag
from dnfm.vo.game_cfg_vo import GameCfgVo
from dnfm.utils import ocr_util
import time
import cv2 as cv
import math
import random
import numpy as np

from dnfm.game.skill import Skill
from dnfm.utils.room_skill_util import RoomSkillUtil
from dnfm.vo.game_main_vo import GameMainVo
from dnfm.vo.game_param_vo import GameParamVO
from dnfm.utils.log_util import logger


def get_all_subclasses(parent_class):
    subclasses = [subclass for subclass in inspect.getmembers(inspect.getmodule(parent_class), inspect.isclass) if
                  issubclass(subclass[1], parent_class)]
    return [subclass[1] for subclass in subclasses]

class GameAction:
    def __init__(self, ctrl: GameControl,infer_queue,screen:Screen):
        #启动时 是停止状态 要点run才开始
        self.stop_event = True
        #从日志工具类中获取 已经配置好了的
        self.logger = logger

        #屏幕对象
        self.screen = screen

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
        self.last_hx_hy_sum = 0
        self.not_move_cnt = 0

        #用来控制过图时改变地图索引,当为True时 要加地图的判断
        self.is_add_room_num = True


        #当前的地图
        self.map= Bwj(self)

        #有门的时候的一些操作
        self.find_door = False
        self.stop_event = True
        self.thread_run = True
        # thread = threading.Thread(target=self.start_game,name="action")  # 创建线程，并指定目标函数
        thread = threading.Thread(target=self.select_game,name="action")  # 创建线程，并指定目标函数
        thread.daemon = True  # 设置为守护线程（可选）
        thread.start()

    def reset(self):
        #重置的时候 让线程跑完 再起新线程
        self.thread_run = False
        time.sleep(1)

        self.stop_event = True
        # self.param = GameParamVO()
        # 用来控制过图时改变地图索引,当为True时 要加地图的判断
        self.is_add_room_num = True

        self.thread_run = True
        thread = threading.Thread(target=self.start_game, name="action")  # 创建线程，并指定目标函数
        thread.daemon = True  # 设置为守护线程（可选）
        thread.start()



    def random_move(self):

        move_angle_cfg=[91,11,181,358]
        moveAngle = move_angle_cfg[random.randint(0, 3)]
        self.logger.info(f"卡死中，随机移动1秒 角度: {moveAngle}")
        self.move_angle(moveAngle,0.5)

    def start_game(self):
        self.logger.info(f"开始处理游戏逻辑,map:{self.map}")
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
            # self.logger.info(result)
            #定位当前地图
            self.map.get_cur_map(result)
            # 处理图片结果 并做出相应的反应
            # 1、获取当前角色的位置
            # 2、有怪物就先移动 并攻击怪物
            # 3、如果没怪物有物品就先去捡物品
            # 4、如果没有物品 有箭头和门后 就移动到箭头和门旁边。如果在狮子头旁边就先移动狮子头房间，并攻击狮子头
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

                            # if not self.skill_buff_start:
                            #     #只有在英雄时 才加buff
                            #     if len(find_tag(result,['hero']))>0:
                            #         # self.logger.info("开始加buff")
                            #         # time.sleep(5)
                            #         self.ctrl.skillBuff()
                            #         self.skill_buff_start = True
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
        #直接按当前地图的再次挑战逻辑判断
        return self.map.start_next_game()

    '''
     移动指令的执行
     包含 物品 门
    '''
    def move(self, t: float = 0.7):
            result = self.find_result()


            monsters = find_tag(result, ["monster","boss"])
            if len(monsters) > 0:
                self.logger.info("找到怪物，停止移动。")
                self.ctrl.end_move()
                return

            monsters = find_tag(result, ["card"])
            if len(monsters) > 0:
                self.logger.info("找到卡，停止移动。")
                self.ctrl.end_move()
                return


            #更新英雄位置
            heros = find_tag(result, ["hero"])
            if len(heros)>0:
                self.hero_x, self.hero_y = get_detect_obj_center(heros[0])
            #判断英雄是否未移动
            self.is_hero_not_move()
            #物品的处理逻辑还是在
            #如果有地图的特殊移动则先执行，返回为True时 则是移动了 否则则执行通用的箭头移动
            if self.map.map_move(result):
                return

            # 找物品 在箭头之前
            items = find_tag(result, ["item"])
            if len(items) > 0:
                # print(items)
                self.logger.info(f'找到物品:{len(items)}个，英雄位置x:{self.hero_x},y:{self.hero_y}')
                # 获取最近的物品 然后计算角度移动
                closest_box, min_distance = find_close_point_to_box(items, self.hero_x, self.hero_y)
                self.move_to_target(closest_box)
                # 优先级算高的 直接返回
                return True
            arrows = find_tag(result, ["arrow"])
            if len(arrows)>0:
                # print(arrows)

                self.logger.info(f'使用箭头过图开始')
                farthest_box,far_distance =find_farthest_point_to_box(arrows,self.hero_x,self.hero_y)
                self.move_to_target(farthest_box)

    def is_hero_not_move(self):

        cur_hx_hy_sum = self.hero_x + self.hero_y
        if abs(self.last_hx_hy_sum - cur_hx_hy_sum) < 100:
            self.not_move_cnt += 1
        else:
            #只要移动过一次就算移动过 初始化一下
            self.not_move_cnt = 0
        if self.not_move_cnt > 20:
            self.logger.info(f'未移动次数：{self.not_move_cnt}，随机移动')
            self.not_move_cnt = 0
            self.ctrl.end_move()
            self.random_move()

        self.last_hx_hy_sum = cur_hx_hy_sum

    #过图逻辑
    def move_to_next_room(self,result):
        ada_image = cv.adaptiveThreshold(cv.cvtColor(result[0].plot(), cv.COLOR_BGR2GRAY), 255,
                                         cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 3)
        if np.sum(ada_image) == 0:
            if not self.is_add_room_num :
                self.ctrl.end_move()
                self.logger.info('过图中...')
                self.is_add_room_num = True
            return True

        if self.is_add_room_num:

            if len(find_tag(result,["hero"])) > 0:
                self.is_add_room_num = False
                self.map.add_room(result)
        return False


    #移动到检测到的box 传入box
    def move_to_target(self,target_box):
        clost_x, clost_y = get_detect_obj_center(target_box)
        angle = calc_angle(self.hero_x, self.hero_y, clost_x, clost_y)
        self.move_angle(angle)


    '''
     单纯的移动
    '''
    def move_angle(self, angle:int,t: float = 0):
        #需要更新英雄位置 检测到时需要更新
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
        for box in result[0].boxes:
            if box is None:
                continue

            itemTmp = box.cls.item()
            for labelTmp in Label:
                if labelTmp.value == itemTmp:
                    if Label.hero.value == itemTmp:
                        if (len(box.xywh[0]) > 0):
                            self.hero_x, self.hero_y = get_detect_obj_center(box)
                        continue
                    elif Label.arrow.value == itemTmp or Label.lion_gate.value == itemTmp or Label.item.value == itemTmp or Label.next_gate.value == itemTmp \
                            or Label.opendoor_b.value == itemTmp or Label.opendoor_u.value == itemTmp or Label.opendoor_l.value == itemTmp or Label.opendoor_r.value == itemTmp :

                        rtn_arry.append(
                            Command.MOVE
                        )

                    # 狮子头 如果没进过，则优先进狮子头，判断逻辑 判断lion_flag =  True.
                    elif Label.monster.value == itemTmp or Label.boss.value == itemTmp:
                        rtn_arry.append(Command.ATACK)
                    # 有卡片的时候做再次挑战的处理
                    elif Label.card.value == itemTmp:

                        rtn_arry.append(Command.AGAIN)

                    #当前地图的特殊指令
                    self.map.get_map_cmd(rtn_arry,itemTmp)

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
            self.map.map_skill_defi()
            # if self.skill_cnt[self.curl_env] == 0:
            #     self.bwj_room_logic()
            #     continue

            result = self.find_result()
            #攻击的时候 也要刷新地图，不然一直卡住了。
            self.map.get_cur_map(result)
            # 刷新命令，主要是识别当前房间的信息
            cmd_array = self.get_cur_order(result)
            # 如果有停止的条件 则不再攻击
            tags = find_tag(result, ["opendoor_l", "opendoor_u","opendoor_b","opendoor_r", "arrow", "next_gate", "lion_gate", "card"])
            if len(tags) > 0:
                self.logger.info("找到门，不再攻击")
                self.atack_flag = False
                return

            tags = find_tag(result, ["monster", "boss"])
            if len(tags) == 0:
                self.ctrl.end_move()
                monster_cnt += 1
                # 没找到怪物10次 就重新攻击一下
                if (monster_cnt % 30 == 0):
                    self.logger.info(f"没找到怪物,次数:{monster_cnt}，开始攻击一次")
                    # self.move(0.1)
                    self.ctrl.attack(1)
                    self.ctrl.skill_free(self.mainVo.role)

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
                # 执行修理 先写死吧 不然是要识别好然后重新点击的
                self.logger.info("执行修理装备")
                self.ctrl.repair()
            if (self.find_text(result, ['再次挑战地下城'])):
                return True
            again_cnt += 1

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
            try:
                time.sleep(0.1)
                self.logger.info(f'找当前角色疲劳，次数：{again_cnt}')

                if again_cnt > 20:
                    return -1
                img_screen = Image.fromarray(self.find_result()[0].plot())
                region = img_screen.crop((278, 80, 440, 124))
                again_img_region_path = 'pl_img_region_path.jpg'
                region.save(again_img_region_path)
                result = ocr_util.read_red_text(again_img_region_path)
                if result is not None and len(result) > 0:
                    self.logger.info(f'找疲劳结果:{result}')
                    (center_x, center_y), pl_txt = result[0]
                    if pl_txt.find('/') > 0:
                        pls = pl_txt.split('/')
                        self.logger.info(f'当前疲劳:{pls[0]}')
                        if pls[0] == '@' or pls[0] == 'o':
                            self.mainVo.pl = 0
                            return 0
                        self.mainVo.pl = int(pls[0])
                        return int(pls[0])
                again_cnt += 1
            except Exception as e:
              print(f'出现异常:{e}')
            traceback.print_exc()

    def find_result(self):
        return self.infer_queue.get()


    '''
        找屏幕上的文字而且点击它
        找到返回True
        未找到返回False
    '''

    def select_game(self):
        game_cfg = GameCfgVo()
        '''
        要在角色选择界面,或者在赛丽亚房间
        先点开个人角色面板 识别当前角色和疲劳，然后看配置，进行每日清理
        1,不管在哪，回到选择角色页面，好识别
        如果是在副本中 则返回城镇
        
        '''
        # while True:
            #查看当前的顺序
            # if self.mainVo.role_name is None:
            #     self.logger.info("当前角色为空，需要重新获取攻坚副本配置")
        self.logger.info("开始按配置选择角色 进行副本")
        for roleCfg in game_cfg.fb_cfg :
            time.sleep(10)
            if self.select_role(roleCfg[0]):
                self.logger.info(f'当前角色:{roleCfg[0]},职业:{roleCfg[1]},挑战的副本配置:{roleCfg[2]}')

                for fbCfgTmp in roleCfg[2] :
                    self.logger.info(f"开始挑战副本：{fbCfgTmp[0]},挑战次数:{fbCfgTmp[1]}")
                    self.mainVo = GameMainVo()
                    self.mainVo.role = roleCfg[1]
                    #当前准备挑战的副本
                    self.mainVo.cur_map = fbCfgTmp[0]
                    #当前副本的配置挑战次数
                    self.mainVo.fb_cfg_cnt = fbCfgTmp[1]

                    # 获取当前地图副本
                    # subclasses = get_all_subclasses(Map)
                    # tmpMap = None
                    # for subc in subclasses:
                    #     #初始化当前Map
                    #     tmpMap = subc(self)
                    #     if tmpMap.get_map() == fbCfgTmp[0]:
                    #         self.logger.info(f"找到 挑战副本：{fbCfgTmp[0]},的配置。")
                    #         break
                    #     tmpMap = None
                    # if tmpMap is not None:
                    if fbCfgTmp[0] == GameMap.BWJ:
                        self.map = Bwj(self)
                    #选择角色很慢
                    time.sleep(20)
                    #进入副本 并开始游戏
                    if self.map.find_and_entry():
                        self.start_game()
                    else:
                        self.logger.info(f"挑战副本：{fbCfgTmp[0]},未找到副本的配置。")



    '''
    选择指定角色 
    暂时没做找不到的判断
    '''
    def select_role(self, role_name):
        start = time.time()
        #先判断当前角色是不是 点击头像看是不是如果
        # result = self.screen.find_screen_text("选角")
        # if result is not None:
        #     self.ctrl.click(result)
        if self.screen.find_screen_text_click("选角"):
            if self.screen.find_screen_text_click(role_name):
                self.logger.info(f'找到角色:{role_name},点击完了,耗时{(time.time() - start) * 1000}ms')
                return True
            else:
                #没找到角色可能是左边没滑，左边选角界面滑一下
                self.ctrl.adb.touch_swipe(486,532,482,160)
                if self.screen.find_screen_text_click(role_name):
                    self.logger.info(f'找到角色:{role_name},点击完了,耗时{(time.time() - start) * 1000}ms')
                    return True
                self.logger.info(f'未找到角色：{role_name},结束,耗时{(time.time() - start) * 1000}ms')
                self.ctrl.click(0,0)
                return False
        self.logger.info(f'未找到选角按扭,结束,耗时{(time.time() - start) * 1000}ms')
        return False

    '''
    选择指定角色 
    暂时没做找不到的判断
    '''
    def select_role1(self, role_name):
        start = time.time()
        #先判断当前角色是不是 点击头像看是不是如果
        # result = self.screen.find_screen_text("选角")
        # if result is not None:
        #     self.ctrl.click(result)
        if self.screen.find_screen_text_click("选角"):
            if self.screen.find_screen_text_click(role_name):
                self.logger.info(f'找到角色:{role_name},点击完了,耗时{(time.time() - start) * 1000}ms')
                return True
            else:
                #没找到角色可能是左边没滑，左边选角界面滑一下
                self.ctrl.adb.touch_swipe(486,532,482,160)
                if self.screen.find_screen_text_click(role_name):
                    self.logger.info(f'找到角色:{role_name},点击完了,耗时{(time.time() - start) * 1000}ms')
                    return True
                self.logger.info(f'未找到角色：{role_name},结束,耗时{(time.time() - start) * 1000}ms')
                self.ctrl.click(0,0)
                return False
        self.logger.info(f'未找到选角按扭,结束,耗时{(time.time() - start) * 1000}ms')
        return False

    '''
    直接不选地图那些操作 直接开始布万家 用来做中断后的操作 手动介入
    新开一个线程
    '''
    def start_bwj(self):
        self.map = Bwj(self)
        self.map.param = GameParamVO()

        thread = threading.Thread(target=self.start_game, name="action")  # 创建线程，并指定目标函数
        thread.daemon = True  # 设置为守护线程（可选）
        thread.start()

if __name__ == '__main__':
    ctrl = GameControl(ScrcpyADB())
    action = GameAction(ctrl)

    action.select_game()
    # action.select_game()