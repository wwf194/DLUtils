x+)JMU�46a040031Q�K�,�L��/JeX}Y�aE@���d��"��ݩ}/L�@A��89?%�a~�$�I��'w�ܨ�r��/�C�	�/I����K-�+�`�����"�د��4�J����m�*�����,���+�dX��H�ڮ������d�?��b\YnbfD��O-FK���o(8���p���a�%E�y�i�E�g�m>{�޾���P�E�{�����ļĜ���b��z�Ϝ����~��"S���z�� J���S�R)<_hy���/�,�+�/59uPr~nnb^J1���o�	��^{'���'5-b��BJI,IdX������}�.�z�<Aj7�~ԔԲĜ�ĒT�)fK֙���M��s���#.BLI�H�-�Ie�P)4��J΃����޹�Ϸ�P��S�sSK�*�\@F]��?-��m�+�/��YO��:K�0'��8��R������D�,�Gǖ�Z`��C�������̼����̒�������'��EY�ʹYe:����ȵ�!���Og�2�ߊ��Qk��(?�Qǩ���L���!E���a/w�s����]��0ӅD#�tn~J)�o�7?�V���ii��/�ۭ�p��(�K-)�/�f��9e�ǰ�O�Tr���=i��˚P���d�fV���I�K���{6�9��UF�h��3��`�^�N��+;]��C��a��^t~��T���l���Z��7V�d,N������c�)%���V��XC_'�v*�'U�M�<i�C*]L�J{,�p��=�H4#[Ϛ�J͚pyh
�Q�r�g�$���˪��O?��-DMiIfN1�s�P�{E��z�	ڿ/3�m� ���                                                                                                                                                                                         ��moe��Y�DÕ`�J�~+����-�
[c.UB���
�}��ak���3��mv��.�bʇ>ܕc!��Z1f8�p��4Eq�аZ��8��~Y��pK'XP�������&��u!�D�'�8Ky�Bn��Zv�������B�<�ϡ�|��N 5L�=]kj��<�+��\j=��I�`���Duxpx"����w�mr�6U����R)���w=*ui��K˨Gث��Q0Z��Ra�I01a�  # ^����������'4Lo���ۜ�9�@�}�)taw_F �R�/�ԭ�h�Vo�:[��p{��p���z�n�k���,9�ωv�-���^ݪ�H�����]4�vUA�\�1���u�nYӀ�]�#୷���V��(;��O�qSu�q�҈���� >�TO0�'����N`���76����Q��l�
��Ԓ:�:�ށ�L���<�wU?I���Q*��mm!�Gvcg"50����      VALUES (?)
    '''
    Operator.execute(Query, [(datetime.datetime.now())])
    Session.commit()

def Str2DataTimeObject(Str, Format):
    datetime.strptime(Str, Format)

InsertCurrentTime(Operator, Session)
Result = Operator.execute(
    '''
    SELECT time_stamp FROM heart_beat
    ORDER BY time_stamp DESC
    '''
    # later time will come first
)
TimeStampList = Result.fetchall()
LatestTime = Str2DataTimeObject(TimeStampList[0][0], Format="YYYY-MM-DD HH:MM:SS")


print(TimeStampList)

Session.close()

from apscheduler.schedulers.background import BackgroundScheduler
Scheduler = BackgroundScheduler()

# interval：固定间隔重复执行
Scheduler.add_job(
    id="1", name="tick",
    func=InsertCurrentTime, trigger="interval",
    min=1
)




import time
def Test():
    print("Running Startup Task")
    print("Time: ", DLUtils.system.CurrentTimeStr())
    print("PID: ", DLUtils.system.CurrentProcessPID())

    for Str in PrintList:
        print(Str, end='')
        time.sleep(1.0)

if __name__ == "__main__":    
    CurrentDirPath = DLUtils.file.CurrentDirPath(__file__)
    CurrentFileName = DLUtils.file.CurrentFileName(__file__)

    Name, Suffix = DLUtils.file.SeparateFileNameSuffix(CurrentFileName)

    with open(CurrentDirPath + Name + "-out.txt", 'w') as sys.stdout:
        Test()                                                                                                                                                                     tils.image.TextOnImageCenter(Image, Str, Color=(0, 255, 0))
        import matplotlib.pyplot as plt
        plt.imshow(Image)
        Image2File(Image, DLUtils.file.AppendSuffix2FileName(ImageFilePath, "-text-%s"%Str))

if __name__ == '__main__':
    Test()

def CompressImageAtFolder(DirPath):
    ListFiles = DLUtils.file.ListAllFileNames(DirPath)
    return
