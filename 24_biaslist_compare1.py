import os
import pathlib
import csv
import numpy as np
def get_root(file_path, file_search):  # get_root 抓取路徑底下所有的txt檔
    pathProg = os.getcwd()
    pathProg = pathProg + file_path
    data_path = []
    for root, dirs, files in os.walk(pathProg, topdown=False):
        for f in files:
            if f.endswith(file_search):
                data_path.append(root + '/' + f)
    data_path.sort()
    return data_path

def get_file_path(file_path, n):
    # 使用 os 模組的方法來獲取目錄名稱
    normalized_path  = os.path.dirname(file_path)

    # 使用 os.path.split 來分割路徑，得到 (head, tail) 的元組
    head, tail = os.path.split(normalized_path)

    # 將 tail 使用斜線和反斜線分割
    segments = head.split(os.path.sep)[-2]

    # 再次分割相對路徑
    head, tail = os.path.split(segments)
    segments = tail.split(os.path.sep)

    # 取得倒數第四段字串
    # for segment in reversed(segments[-2]):
    #     if segment:
    return segments[0]

    # return result

def sum_columns_from_txt(filename):
    # 讀取文本文件，將字符串轉換為整數
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 將字符串轉換為整數列表，並過濾掉換行
    data = [list(map(lambda x: int(x) if x.strip() else 0, line.strip('[]\n').split(','))) for line in lines]

    # 確保每行的長度相同，如果太短，用零填充
    max_length = max(len(row) for row in data)
    data = [row + [0] * (max_length - len(row)) for row in data]

    print(np.shape(data))
    # 計算每一列的總和
    sum_column1 = sum(data[1])
    sum_column2 = sum(data[3])

    # print(sum_column1, sum_column2)

    return sum_column1, sum_column2


#############################################################################################

##### [程式用途：計算雜訊比例] ##### 論文 p.14
# 輸入  ：antibias_list.txt    / 資料夾：folder_A_path
# 輸出01：biaslist_2A_ERD.csv  / 資料夾：output_file_path


# 收集 antibias_list.txt，比較 2a 2b 版本輸出差異
set = '2A'
# 資料夾 A 的路徑
folder_A_path = f'./Step2/{set}_ERD_plot_30trials/'
# 要寫入的合併檔案的路徑
output_file_path = f'./Step4/_雜訊紀錄/biaslist_{set}_ERD.csv'

npy_data_path_All = get_root(folder_A_path, 'antibias_list.txt')

with open(output_file_path, 'w') as output_file:
    output_file.write('')

# with open(output_file_path, 'w') as output_file:
with open(output_file_path, 'a', newline='') as file:
    for npy_data_path in npy_data_path_All:
        name = str(pathlib.Path(npy_data_path).parent.parent.resolve())[-2:]
        team = get_file_path(npy_data_path, -3)
        left_n, right_n = sum_columns_from_txt(npy_data_path)
        
        writer = csv.writer(file)
        writer.writerow([team, name, left_n, right_n])
        print(team, name, left_n, right_n)

        # 
