import time

from GAME.game_control import GameControl
from dnfm.utils import ocr_util
from dnfm.utils.log_util import logger


class Screen:
    def __init__(self, image_que, ctrl: GameControl):
        self.logger = logger
        self.ctrl = ctrl
        self.image_que = image_que

    def find_screen_text_click(self, text):
        #找寻次数 20次以后 就不找
        find_cnt_limit = 20
        find_select_role_cnt = 0
        while True:
            time.sleep(0.2)
            find_select_role_cnt+=1
            ocr_result = self.find_ocr_result(self.find_result())
            select_role_list = self.find_text(ocr_result, [text])
            if len(select_role_list) > 0:
                (centen_x, centen_y), tmp_txt = select_role_list[0]
                self.ctrl.click(centen_x, centen_y)
                self.logger.info(f'找到:{text},点击完了,找寻次数：{find_select_role_cnt}')
                return True

            if find_select_role_cnt > find_cnt_limit:
                self.logger.info(f'未找到{text},结束,找寻次数：{find_select_role_cnt}')
                return False
    '''
        找到并返回坐标
    '''
    def find_screen_text(self, text):
        #找寻次数 20次以后 就不找
        find_cnt_limit = 20
        find_select_role_cnt = 0
        while True:
            time.sleep(0.4)
            find_select_role_cnt+=1
            ocr_result = self.find_ocr_result(self.find_result())
            select_role_list = self.find_text(ocr_result, [text])
            if len(select_role_list) > 0:
                self.logger.info(f'找到:{text},找寻次数：{find_select_role_cnt}')
                (centen_x, centen_y), tmp_txt = select_role_list[0]
                return (centen_x, centen_y)

            if find_select_role_cnt > find_cnt_limit:
                self.logger.info(f'未找到{text},结束,找寻次数：{find_select_role_cnt}')
                return False

    def find_ocr_result(self,img):
        return ocr_util.readtext(img)

    '''
        返回当前屏幕的图片
    '''
    def find_result(self):
        if self.image_que.empty():
            time.sleep(0.001)
        return self.image_que.get()

    def find_text(self, result, tag):
        """
        根据标签名称来找到目标
        :param result:
        :param tag:
        :return: list 判断len大于0则是找到了数据
        """

        hero = [x for x in result if x[1] in tag]
        return hero