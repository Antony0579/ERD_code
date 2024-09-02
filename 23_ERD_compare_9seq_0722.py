# 將每個人的 ERD 指標進行整理
# 包含 (峰值時間、峰值強度、總能量)
# 預定輸出 shape = 9,1,8,3 (seq, side, ch, 指標)
# 1seq = 約30trials

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定支援中文的字型

#%% Function #

def progressBar(title, temp, skip, total):
    if temp % skip == 0 or temp + skip >= total:
        print('\r' + '['+title+']:[%s%s] %s(%.2f%%)' % ('█' * int(temp/total*20), ' ' * (20-int(temp/total*20)), str(temp)+'/'+str(total), float(temp/total*100)), end='')
        if temp == total:
            print ('')

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

def ERD_index_02(data, ch): # 指標計算

    # 1-2 取 0-6 秒內小於 0 的點個數
    data_subset = data[ch, 4000:4000+TIME_reigon] # 取 0-6 秒

    first_index = 0
    last_index  = 0
    current_count = 0

    for t in range(len(data_subset)):
        if data_subset[t] < 0:
            current_count += 1

            if first_index is 0:
                first_index = t
            last_index = t

    # 計算0-6秒 <0的值
    ERD_delay =  current_count
    if current_count == 0:
        print(f"ERROR！持續時間 = {current_count}")

    # 2. 計算峰值
    ERD_max = -min(data_subset)
    if ERD_max < 0:
        print(f"ERROR！計算峰值 {ERD_max}")
        print(f"強制歸零")
        ERD_max = 0

    # 3. 找出峰值瞬間
    ERD_maxT = np.argmin(data_subset)

    # print(first_index,last_index)
    # 4. 計算小於0總能量
    ERD_sum = -np.sum([value for value in data_subset[
        first_index:last_index+1] if value < 0])
    # print(f"計算總能量 {ERD_sum}")

    return ERD_delay, ERD_max, ERD_maxT, ERD_sum
    
def params_shift(par1, par2):
    # 確保資料不被覆蓋
    return par2.copy(), par1.copy()
    # X = par1
    # par1 = par2
    # par2 = X
    # return par1, par2

def LR_shift(ERD_output_L, ERD_output_R): # 左右撇子互換
    # print('01',ERD_output_L[:,:,2])
    # print('01',ERD_output_R[:,:,2])
    # print(ERD_output_R[5])

    # 左右互換 + C34互換
    ERD_output_L, ERD_output_R = params_shift(ERD_output_L, ERD_output_R)
    # print('02',ERD_output_L[:,:,2])
    # print('02',ERD_output_R[:,:,2]) 

    # Left-Right and C34 swaps
    columns_to_swap = [(0, 1), (2, 4), (5, 7)]
    # Apply swaps to ERD_output_L and ERD_output_R
    for col1, col2 in columns_to_swap:
        ERD_output_L[:, col1], ERD_output_L[:, col2] = params_shift(ERD_output_L[:, col1], ERD_output_L[:, col2])
        ERD_output_R[:, col1], ERD_output_R[:, col2] = params_shift(ERD_output_R[:, col1], ERD_output_R[:, col2])
    # print('03',ERD_output_L[:,:,2])
    # print('03',ERD_output_R[:,:,2]) 

    return ERD_output_L, ERD_output_R      

    

#%% Function #

##### [程式用途：ERD指標分析計算] ##### 論文 p.15
# 輸入  ：ERDS_data_left.npy    / 資料夾：FILE_PATH
# 輸出01：ERDS_9seq_left.npy    / 資料夾：SAVE_PATH_0

# 1207 新增變數 峰值瞬間 
# 0305 聖佶左撇子，左右手互換
# 0308 取0-4s /取0-2s
    
# 全通道顯示
# SUBJECT_NAMES_NONE = ['泓誌','建誌']
SUBJECT_NAMES_NF = ['又諄','文豪','柏瑋','皓嚴','羅喨']
SUBJECT_NAMES_FB = ['嘉浚','浩軒','柏勛','松育','聖佶']
SUBJECT_NAMES_AD = ['建誌','柏崴','郁芹','璿瑋','寶心']
SUBJECT_NAMES    = SUBJECT_NAMES_AD #['柏崴']

SIDE_SET_LIST = ['left' ,'right']  #,'right'
INDEX_LIST = ['持續時間','峰值強度','峰值瞬間','總能量']
len_index = len(INDEX_LIST)

TIME_LIST = [6]
# TIME = 6
# TIME_reigon = TIME*1000 # 0~4s
#################################################
from _Run_set import FRERUENCY_LIST
# SUBJECT_NAMES = ['又諄','柏瑋','羅喨','嘉浚','浩軒','松育']
# SUBJECT_NAMES = SUBJECT_NAMES_AD

for TIME in TIME_LIST:
    TIME_reigon = TIME*1000 # 0~4s
    
    for freq_n, freq in enumerate(FRERUENCY_LIST):
        print("-"*20)
        for name in SUBJECT_NAMES: 

            # if name in SUBJECT_NAMES_NONE: TEAM = '備案'
            if   name in SUBJECT_NAMES_FB: TEAM = '震動想像組'; 
            elif name in SUBJECT_NAMES_NF: TEAM = '純想像組';   
            elif name in SUBJECT_NAMES_AD: TEAM = '純震動組';   
            
            if freq == 'A1': name_set = 'A'
            if freq == 'B1': name_set = 'B'
            FILE_PATH   = f'./Step2/2{name_set}_ERD_plot_30trials/{TEAM}/{name}/'
            SAVE_PATH_0 = f'./Step3_2/3{name_set}比較_ERD_plot_9seq/{TEAM}/{name}/'
            if not os.path.isdir(SAVE_PATH_0):os.makedirs(SAVE_PATH_0)
            print(name)
            
            for side in range(len(SIDE_SET_LIST)):   
                ERD_output = np.zeros((16, 8, len_index)) # (side, seq, ch, 指標)   
                npy_data_path = get_root(FILE_PATH, f'ERDS_data_{SIDE_SET_LIST[side]}.npy')
                for i_npy in range(len(npy_data_path)):
                    data = np.load(npy_data_path[i_npy]) # (8,9000)
                    # print(data.shape)
                    for ch in range(len(data)):
                        # ERD_delay, ERD_max, ERD_sum = ERD_index(data, ch)
                        ERD_delay, ERD_max, ERD_maxT, ERD_sum = ERD_index_02(data, ch)
                        ERD_output[i_npy, ch] = [ERD_delay, ERD_max, ERD_maxT, ERD_sum]
                        # (峰值時間、峰值強度、總能量)
                if SIDE_SET_LIST[side]=='left': ERD_output_L = ERD_output[:len(npy_data_path)]
                if SIDE_SET_LIST[side]=='right':ERD_output_R = ERD_output[:len(npy_data_path)]
            
            if name == '聖佶':
                ERD_output_L, ERD_output_R =  LR_shift(ERD_output_L, ERD_output_R)   
                # print('03',ERD_output_L[:,:,2])
                # print('03',ERD_output_R[:,:,2])     
                
            print(ERD_output_L.shape) #(9, 8, 4)
            print(ERD_output_R.shape)
            np.save(f'{SAVE_PATH_0}/ERDS_9seq_left',  ERD_output_L)
            np.save(f'{SAVE_PATH_0}/ERDS_9seq_right', ERD_output_R)
            # ERD_seq_compare(ERD_output_L, name+'_left', SAVE_PATH_0)
            # ERD_seq_compare(ERD_output_R, name+'_right',SAVE_PATH_0)
            # ERD_seq_compare_All(ERD_output_L, ERD_output_R, SAVE_PATH_0, name)
            # ERD_seq_compare_C3C4(ERD_output_L, ERD_output_R, SAVE_PATH_0, name)
            





            

                