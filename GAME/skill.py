from dnfm.game.role import Role


#技能实体类
class Skill:


    #buff技能 暂时都定为两个 在上面四排的右边1，2个进图后释放
    @staticmethod
    def getBuffSkills():
        skillBuff = Skill(1094, 960, 0.1,False,"buff1")
        skillBuff2 = Skill(988, 960, 0.1,False,'buff2')
        return skillBuff,skillBuff2


    @staticmethod
    def getSkillJX():
        skillJx = Skill(776, 960, 0.1,False,'觉醒')
        return skillJx

    @staticmethod
    def getSkills():
        # 技能1 排列在最右侧时 圆形
        # skillBuff= Skill(1920, 343, 0.1)
        # skill1 = Skill(1920, 473, 0.1, True)
        # skill2 = Skill(1920, 639, 0.1)
        # skill3 = Skill(1758, 671, 0.1)
        # skill4 = Skill(1629, 800, 0.1)
        #
        # skill5 = Skill(1295, 967, 0.1)
        # skill6 = Skill(1447, 967, 0.1)
        # skill7 = Skill(1589, 967, 0.1)
        #
        #
        # skillHT = Skill(1742, 980, 0.1,True)
        # skillTY = Skill(2100, 730, 0.1)
        #
        # skillJx = Skill(1611, 341, 0.1)

        # 技能1 排列在最下面时 基础排列阵形

        #第一排右到左，
        skill1 = Skill(1818, 834, 0.1,False,'技能1')
        skill2 = Skill(1664, 835, 0.1,False,'技能2')
        skill3 = Skill(1553, 836, 1.5,False,'技能3')
        skill4 = Skill(1380, 854, 0.1,False,'技能4')

        #第二排左到右
        skill5 = Skill(1370, 952, 0.1,False,'技能5')
        skill6 = Skill(1444, 954, 0.1,False,'技能6')
        skill7 = Skill(1602, 948, 0.1,False,'技能7')

        #觉醒旁边的技能
        skill8 = Skill(876, 952, 0.1,False,'技能8')


        skillHT = Skill(1746, 966, 0.1,True,'后跳')
        skillTY = Skill(1982, 836, 0.1,False,'跳跃')



        return skill1, skill2,skill3,skill4,skill5,skill6,skill7,skill8,skill3

    '''
    根据角色返回技能,但是角色的技能排列要对
    '''
    #根据角色返回技能,但是角色的技能排列要对
    @staticmethod
    def get_role_skills(role):
        match role:
            case Role.NAIMA:
                return Skill.getSkillsNaima()
            case Role.QIGONG:
                return Skill.getSkillsQigong()
            case Role.HONGYAN:
                return Skill.getSkillsHongyan()
            case Role.Jianshen:
                return Skill.getSkillsJianshen()
            case _:
                return Skill.getSkillsNaima()




    #最常用的一个技能
    @staticmethod
    def getSkill3():
        print('释放技能3')
        skill3 = Skill(1553, 836, 0.5,False,'技能3')
        return skill3

    #奶妈的技能定义
    @staticmethod
    def getSkillsNaima():
        # 技能1 排列在最右侧时 圆形
        # skillBuff= Skill(1920, 343, 0.1)
        # skill1 = Skill(1920, 473, 0.1, True)
        # skill2 = Skill(1920, 639, 0.1)
        # skill3 = Skill(1758, 671, 0.1)
        # skill4 = Skill(1629, 800, 0.1)
        #
        # skill5 = Skill(1295, 967, 0.1)
        # skill6 = Skill(1447, 967, 0.1)
        # skill7 = Skill(1589, 967, 0.1)
        #
        #
        # skillHT = Skill(1742, 980, 0.1,True)
        # skillTY = Skill(2100, 730, 0.1)
        #
        # skillJx = Skill(1611, 341, 0.1)

        # 技能1 排列在最下面时 基础排列阵形

        # 第一排右到左，
        skill1 = Skill(1818, 834, 0.1, False, '光芒烬盾')
        skill2 = Skill(1664, 835, 0.1, False, '洁净之光')
        skill3 = Skill(1553, 836, 0.1, False, '光明惩戒')
        skill4 = Skill(1380, 854, 0.1, False, '领悟之雷')

        # 第二排左到右
        skill5 = Skill(1300, 952, 0.1, False, '勇气颂歌')
        skill6 = Skill(1444, 954, 0.1, False, '光明之杖')
        skill7 = Skill(1602, 948, 0.1, False, '沐天之光')

        # 觉醒旁边的技能
        skill8 = Skill(876, 952, 0.1, False, '胜利之矛')

        skillHT = Skill(1746, 966, 0.1, True, '后跳')

        #10 是buff
        skillBuff = Skill(1094, 960, 0.1, False, "勇气祝福")

        #11 是副buff
        skillBuff2 = Skill(988, 960, 0.1, False, '光之护盾')
        #12 是觉醒
        skillJx = Skill(776, 960, 0.1, False, '觉醒')

        skillTY = Skill(1982, 836, 0.1, False, '跳跃')

        return skill1, skill2, skill3, skill4, skill5, skill6, skill7, skill8, skillHT,skillBuff,skillBuff2,skillJx
    #气功的技能定义
    @staticmethod
    def getSkillsQigong():
        # 技能1 排列在最右侧时 圆形
        # skillBuff= Skill(1920, 343, 0.1)
        # skill1 = Skill(1920, 473, 0.1, True)
        # skill2 = Skill(1920, 639, 0.1)
        # skill3 = Skill(1758, 671, 0.1)
        # skill4 = Skill(1629, 800, 0.1)
        #
        # skill5 = Skill(1295, 967, 0.1)
        # skill6 = Skill(1447, 967, 0.1)
        # skill7 = Skill(1589, 967, 0.1)
        #
        #
        # skillHT = Skill(1742, 980, 0.1,True)
        # skillTY = Skill(2100, 730, 0.1)
        #
        # skillJx = Skill(1611, 341, 0.1)

        # 技能1 排列在最下面时 基础排列阵形

        # 第一排右到左，
        skill1 = Skill(1818, 834, 0.1, False, '气玉弹')
        #2是幻影爆碎需要双击
        skill2 = Skill(1664, 835, 0.1, True, '幻影爆碎')
        #技能3放气功波需要长按
        skill3 = Skill(1553, 836, 1.5, False, '念气波')
        #狮子吼需要长按
        skill4 = Skill(1380, 854, 1.5, False, '狮子吼')

        # 第二排左到右
        skill5 = Skill(1300, 952, 0.1, False, '幻影进击')
        #念气罩 双击可以引爆 但是可以不引爆 先不引爆吧 可以做防御
        skill6 = Skill(1444, 954, 0.1, False, '念气罩')
        skill7 = Skill(1602, 948, 0.1, False, '螺旋念气场')

        # 觉醒旁边的技能
        skill8 = Skill(876, 952, 0.1, False, '念兽：雷龙出海')

        skillHT = Skill(1746, 966, 0.1, True, '后跳')
        # 10 是buff
        skillBuff = Skill(1094, 960, 0.1, False, "念气环绕")

        # 11 是副buff
        skillBuff2 = Skill(988, 960, 0.1, False, '光之兵刃')
        # 12 是觉醒
        skillJx = Skill(776, 960, 0.1, False, '觉醒')

        skillTY = Skill(1982, 836, 0.1, False, '跳跃')

        return skill1, skill2, skill3, skill4, skill5, skill6, skill7, skill8, skillHT, skillBuff, skillBuff2, skillJx

    #红眼技能
    @staticmethod
    def getSkillsHongyan():
        # 技能1 排列在最右侧时 圆形
        # skillBuff= Skill(1920, 343, 0.1)
        # skill1 = Skill(1920, 473, 0.1, True)
        # skill2 = Skill(1920, 639, 0.1)
        # skill3 = Skill(1758, 671, 0.1)
        # skill4 = Skill(1629, 800, 0.1)
        #
        # skill5 = Skill(1295, 967, 0.1)
        # skill6 = Skill(1447, 967, 0.1)
        # skill7 = Skill(1589, 967, 0.1)
        #
        #
        # skillHT = Skill(1742, 980, 0.1,True)
        # skillTY = Skill(2100, 730, 0.1)
        #
        # skillJx = Skill(1611, 341, 0.1)

        # 技能1 排列在最下面时 基础排列阵形

        # 第一排右到左，
        skill1 = Skill(1818, 834, 0.1, False, '噬魂之手')
        #2是幻影爆碎需要双击
        skill2 = Skill(1664, 835, 1.5, True, '绝念除恶击 大吸')
        #技能3放气功波需要长按
        skill3 = Skill(1553, 836, 0.1, False, '崩山击')
        #狮子吼需要长按
        skill4 = Skill(1380, 854, 0.1, False, '爆发之刃')

        # 第二排左到右
        skill5 = Skill(1300, 952, 0.1, False, '愤怒狂刃')
        #念气罩 双击可以引爆 但是可以不引爆 先不引爆吧 可以做防御
        skill6 = Skill(1444, 954, 0.1, False, '怒气爆发')
        skill7 = Skill(1602, 948, 0.1, False, '崩山裂地')

        # 觉醒旁边的技能
        skill8 = Skill(876, 952, 0.1, False, '鬼影剑-鬼斩')

        skillHT = Skill(1746, 966, 0.1, True, '后跳')
        # 10 是buff
        skillBuff = Skill(1094, 960, 0.1, False, "暴走")

        # 11 是副buff
        skillBuff2 = Skill(988, 960, 0.1, False, '压血')
        # 12 是觉醒
        skillJx = Skill(776, 960, 0.1, False, '觉醒')

        skillTY = Skill(1982, 836, 0.1, False, '跳跃')

        return skill1, skill2, skill3, skill4, skill5, skill6, skill7, skill8, skillHT, skillBuff, skillBuff2, skillJx

    #剑神技能
    @staticmethod
    def getSkillsJianshen():
        # 技能1 排列在最右侧时 圆形
        # skillBuff= Skill(1920, 343, 0.1)
        # skill1 = Skill(1920, 473, 0.1, True)
        # skill2 = Skill(1920, 639, 0.1)
        # skill3 = Skill(1758, 671, 0.1)
        # skill4 = Skill(1629, 800, 0.1)
        #
        # skill5 = Skill(1295, 967, 0.1)
        # skill6 = Skill(1447, 967, 0.1)
        # skill7 = Skill(1589, 967, 0.1)
        #
        #
        # skillHT = Skill(1742, 980, 0.1,True)
        # skillTY = Skill(2100, 730, 0.1)
        #
        # skillJx = Skill(1611, 341, 0.1)

        # 技能1 排列在最下面时 基础排列阵形

        # 第一排右到左，
        skill1 = Skill(1818, 834, 0.1, False, '拔刀斩')
        skill2 = Skill(1664, 835, 0.1, False, '流心：绝')
        skill3 = Skill(1553, 836, 0.1, False, '连闪')
        #猛龙需要双击
        skill4 = Skill(1380, 854, 0.1, True, '猛龙断空斩')

        # 第二排左到右
        skill5 = Skill(1300, 952, 0.1, False, '幻影剑舞')
        #念气罩 双击可以引爆 但是可以不引爆 先不引爆吧 可以做防御
        skill6 = Skill(1444, 954, 0.1, False, '流心：连')
        skill7 = Skill(1602, 948, 0.1, False, '破军升龙击')

        # 觉醒旁边的技能
        skill8 = Skill(876, 952, 0.1, False, '破军斩龙击')

        skillHT = Skill(1746, 966, 0.1, True, '后跳')
        # 10 是buff
        skillBuff = Skill(1094, 960, 0.1, False, "破极兵刃")

        # 11 是副buff
        skillBuff2 = Skill(988, 960, 0.1, False, '崩山击')
        # 12 是觉醒
        skillJx = Skill(776, 960, 0.1, False, '觉醒')

        skillTY = Skill(1982, 836, 0.1, False, '跳跃')

        return skill1, skill2, skill3, skill4, skill5, skill6, skill7, skill8, skillHT, skillBuff, skillBuff2, skillJx

    #技能的x坐标  y坐标  按键空隙，是否要双击释放。
    def __init__(self, x: int,y:int,t: float=0.1,doubleClick: bool=False,detail:str='技能'):
        self.x = x
        self.y = y
        self.t = t
        self.doubleClick = doubleClick
        self.detail = detail




