# f = open("err_log.txt", "r")
# print(f.readline())

# f.close()

import os

with open('err_log.txt') as f:
    line = f.readline()
    while line:
        line = f.readline()
        directory = "D:/Book_English/TaiLieu_XuLyAnh_ThiGiacMay/NhanDangKhuonMat/" + \
            line.replace("\\", "/").replace("\n", "")
        print(directory)
        if os.path.isfile(directory):
            os.remove(directory)
        print("Error: %s file not found" % directory)
        # os.remove(r("BanBao\\" + line.replace('\n', '')))
        print(line)
