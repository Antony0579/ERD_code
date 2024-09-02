import os
import numpy as np
import matplotlib.pyplot as plt
import statistics
import scipy.stats as stats
import csv
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定支援中文的字型

plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定支援中文的字型
plt.rcParams['axes.unicode_minus'] = False # 顯示負號
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

def mean_std_2D(Array): # 計算平均值標準差
    # print(Array.shape)

    mean_list  = []
    stdev_list = []
    list_array = Array.tolist()

    for col in range(Array.shape[1]): # 5
        col_data = [row[col] for row in list_array]
        # print(col_data) # 3
        mean  = round(statistics.mean( col_data), 2)
        stdev = round(statistics.stdev(col_data), 2)
        mean_list .append(mean)
        stdev_list.append(stdev)
    
    return mean_list, stdev_list

def bias_filter(data_inf, mean, std): #雜訊濾除
    # F_data = []
    
    F_means = []
    F_stds  = []
    F_point = 0
    
    F_len_1 = 0
    F_len_2 = 0
    F_len_3 = 0

    # print('/n 01', np.shape(data_inf), np.shape(mean), np.shape(std))
    # (15, 11801) (11801,) (11801,)
    for i in range(len(mean)):
        
        
        # 0628
        data_inf_I = data_inf[:,i]
        std_lim = 3
        
        # 計算上下界
        lower_bound = mean[i] - std_lim * std[i]
        upper_bound = mean[i] + std_lim * std[i]

        # 過濾掉超出上下界的數據點
        filtered_data = data_inf_I[(data_inf_I >= lower_bound) & (data_inf_I <= upper_bound)]
        
        # print(np.shape(filtered_data)) #(14,)
        filtered_means = np.mean(filtered_data, axis=0)
        filtered_stds = np.std(filtered_data, axis=0)
        
        F_len = len(data_inf) - len(filtered_data)
        F_point +=F_len
        
        # 找到最長的列長度以保證數據的一致性 (15)    
        # 用0填充每列，使其長度相同
        # padded_data = np.pad(filtered_data, (0, 15 - len(filtered_data)), 'constant').T
        # F_data.append(padded_data)
        
        # if F_len == 1: F_len_1+=1
        # if F_len == 2: F_len_2+=1
        # if F_len == 3: F_len_3+=1

        # 用平均值填充每列，使其長度相同
        # fill_value = np.mean(filtered_data)
        # padded_data = np.pad(filtered_data, (0, len(data_inf) - len(filtered_data)), 'constant', constant_values=(fill_value))
        # F_data.append(padded_data)
        # print(padded_data) #(14,)
        
        F_means.append(filtered_means)
        F_stds.append(filtered_stds)
        
        progressBar("plot...", i+1, 1, len(mean)); i+=1
        
    # print(F_len_1, F_len_2, F_len_3)
    # (11801, 15)    
    # F_data = np.array(F_data).T    
        
    # # 平滑處理2
    SMOOTH = 100
    new_F_means = np.zeros((11801-SMOOTH+1))
    new_F_means = np.convolve(F_means, np.ones(SMOOTH), 'valid') / SMOOTH
    new_F_stds = np.zeros((11801-SMOOTH+1))
    new_F_stds = np.convolve(F_stds, np.ones(SMOOTH), 'valid') / SMOOTH
    # F_means = new_F_means
    
    # 計算過濾後的平均值和標準差
    # filtered_means = np.mean(F_data, axis=0)
    # filtered_stds = np.std(F_data, axis=0)
    # print(np.shape(F_data), np.shape(filtered_means), np.shape(filtered_stds))
    
    return new_F_means, new_F_stds, F_point

### 3比較_ERD_plot_team ###

##### [程式用途：畫出ERD趨勢圖] ##### 論文 p.17、p.29~33
# 輸入  ：ERDS_data_left.npy    / 資料夾：FILE_PATH
# 輸出01：A_left_持續時間       / 資料夾：SAVE_PATH_0

# 0要完整資料才能用

#%% Function #

# SUBJECT_NAMES_NONE = ['泓誌','建誌']
SUBJECT_NAMES_NF = ['又諄','文豪','柏瑋','皓嚴','羅喨']
SUBJECT_NAMES_FB = ['嘉浚','浩軒','柏勛','松育','聖佶']
SUBJECT_NAMES_AD = ['建誌','柏崴','郁芹','璿瑋','寶心']
SIDE_SET_LIST = ['left' ,'right']  #,'right'
# ch_list       = ['C3', 'C4']

TIME_LEN_REST   = 4000
TIME_LEN_ACTION = 8000

import itertools
from _Run_set import SUBJECT_NAMES
FRERUENCY_LIST  = ['A1','B1']
# SUBJECT_NAMES   = SUBJECT_NAMES_AD

# TIME_LIST       = ['6s'] #['2s','4s','6s']
SIDE_SET_LIST   = ['left' ,'right'];                                   LEN_SIDE  = len(SIDE_SET_LIST)
INDEX_LIST      = ['持續時間','峰值強度','峰值瞬間','總能量'];          LEN_INDEX = len(INDEX_LIST) 
CH_LIST         = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'];    LEN_CH    = len(CH_LIST) 

len_freq = 2 # A、B band
len_team = 3 # 2組
len_name = 5 # 1組5人
len_seq  = 9 # 9次實驗
len_data = 11801
# 限制不可超過 9 筆
# 差值為有反饋的減去無反饋的

Data_ALL_0 = np.zeros((len_freq, LEN_SIDE, len_team, len_name, len_seq, LEN_CH, len_data))
# 2, 2, 2, 5, 9, 8, 11801

for freq_n, side_n in itertools.product( # 資料統整
    range(len_freq),
    range(LEN_SIDE),
    ):
    freq = FRERUENCY_LIST[freq_n]
    side = SIDE_SET_LIST[side_n] 
    
    name_n0 = 0
    name_n1 = 0
    name_n2 = 0
    for name_n, name in enumerate(SUBJECT_NAMES): 
        # if name in SUBJECT_NAMES_NONE: TEAM = '備案'
        if   name in SUBJECT_NAMES_FB: TEAM = '震動想像組'; team_n = 0; 
        elif name in SUBJECT_NAMES_NF: TEAM = '純想像組';   team_n = 1; 
        elif name in SUBJECT_NAMES_AD: TEAM = '純震動組';   team_n = 2; 
        
        if freq == 'A1': freq_set = 'A'
        if freq == 'B1': freq_set = 'B'
        
        FILE_PATH   = f'./Step2/2{freq_set}_ERD_plot_30trials/{TEAM}/{name}/'
        SAVE_PATH_0 = f'./Step5_6/5AB_比較_ERD_plot_team_3t2/'
        if not os.path.isdir(SAVE_PATH_0):os.makedirs(SAVE_PATH_0)
       
        npy_data_path = get_root(FILE_PATH, f'ERDS_data_{side}.npy')
        if len(npy_data_path) == 0: continue
        # print('data_subject_NF',len(data_subject_NF))
        data_9seq = []
        for i_npy in range(len(npy_data_path)):
            data = np.load(npy_data_path[i_npy]) # (8,11801)
            data_9seq.append(data)
        
        # print(np.array(data_9seq).shape, freq_n, side_n, team_n, name_n)
        
        if team_n == 0:   Data_ALL_0[freq_n, side_n, 0, name_n0] = np.array(data_9seq);name_n0+=1
        elif team_n == 1: Data_ALL_0[freq_n, side_n, 1, name_n1] = np.array(data_9seq);name_n1+=1
        elif team_n == 2: Data_ALL_0[freq_n, side_n, 2, name_n2] = np.array(data_9seq);name_n2+=1
        # Data_ALL_0[freq_n, side_n, 2, :] = Data_ALL_0[freq_n, side_n, 1, :]

print('Data_ALL_0 Done')
#######################

# fig2, ax2 = plt.subplots(3, 3, figsize=(18,9))
    

#len_freq, LEN_SIDE, len_team, len_name, len_seq, LEN_CH, len_data
#Data_ALL_0 2, 2, 3, 5, 9, 8, 11801
reshaped  = np.reshape( Data_ALL_0, (2, 2, 3, 5, 3, 3, 8, 11801))
reshaped2 = np.swapaxes(reshaped, 4, 5) # 用後面的那個維度# [123-2. 345-5. 678-8.]  
Data_IN   = np.reshape( reshaped2, (2, 2, 3, -1, 3, 8, 11801))
# Data_IN 2, 2, 3, 15, 3, 8, 11801
# len_seq = data_NF_in.shape[1] 
   
# lenq3 = int(len_seq/3)
lenq2 = 3


# 紀錄+畫圖
with open(SAVE_PATH_0+'output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['freq', 'side', 'seq_n', 'ch_n', 'FB_F_point', 'NF_F_point', 'AD_F_point'] )  # 寫入標題行
    
    for_loop01 = 0
    for freq_n, side_n in itertools.product(
        range(len_freq),
        range(LEN_SIDE),
        ):
        freq = FRERUENCY_LIST[freq_n]
        side = SIDE_SET_LIST[side_n] 
        progressBar("plot...", for_loop01+1, 1, lenq2*LEN_CH); for_loop01+=1
        
        # 畫圖
        fig, ax = plt.subplots(8, lenq2, figsize=(24,36))
        LIMIT = 75
        
        # if for_loop01 == 2: break
        
        for_loop02 = 0
        seq_n_skip = 0
        for seq_n, ch_n in itertools.product(
            range(lenq2),
            range(LEN_CH),
            ):
            ch = CH_LIST[ch_n] 
                
            # if seq_n == 1: seq_n_skip = 1
            # else: seq_n_skip = 0
            
            # if for_loop02 == 1: break
            # if for_loop02 == 10: break
            
            progressBar("plot...", for_loop02+1, 1, lenq2*LEN_CH)
            for_loop02+=1
            # data_re3 = data_All[: ,i_npy ,row ,col ,:]
            # print(data_NF_in[:,seq,:].shape)

            # 計算平均
            # Data_IN 2, 2, 3, 15, 3, 8, 11801
            FB_Data_IN = Data_IN[freq_n, side_n, 0, :, seq_n+seq_n_skip, ch_n, :]
            NF_Data_IN = Data_IN[freq_n, side_n, 1, :, seq_n+seq_n_skip, ch_n, :]
            AD_Data_IN = Data_IN[freq_n, side_n, 2, :, seq_n+seq_n_skip, ch_n, :]
            
            FB_mean, FB_std = mean_std_2D(FB_Data_IN)
            NF_mean, NF_std = mean_std_2D(NF_Data_IN)        
            AD_mean, AD_std = mean_std_2D(AD_Data_IN)
                    
            print('FB')
            FB_mean, FB_std, FB_F_point = bias_filter(FB_Data_IN, FB_mean, FB_std)
            print('NF')
            NF_mean, NF_std, NF_F_point = bias_filter(NF_Data_IN, NF_mean, NF_std)
            print('AD')
            AD_mean, AD_std, AD_F_point = bias_filter(AD_Data_IN, AD_mean, AD_std)
            
            writer.writerow([freq, side, seq_n, ch, FB_F_point, NF_F_point, AD_F_point])
            
            print('Drawing...01')
            txt_size = 21
            
            len_NF_mean = len(NF_mean)
            x  = np.linspace(0, len_NF_mean, len_NF_mean)
            y1 = np.zeros      (len_NF_mean)
            
            second_x = (x-TIME_LEN_REST)

            if for_loop02 <= 8 :
                # 添加文字
                ax[ch_n, seq_n].text(-0.3, 0.5, f'{ch}', transform=ax[ch_n, seq_n].transAxes, va='center', fontsize=txt_size*2)

            # if seq_n == 0: ax[ch_n, seq_n].set_title( f'week 1', fontsize=txt_size+4) #{ch}, 
            # else: ax[ch_n, seq_n].set_title( f'week 3', fontsize=txt_size+4) #{ch}, 
            ax[ch_n, seq_n].set_xticks(range(-4000, len_NF_mean, 2000)) 
            
            ax[ch_n, seq_n].set_title( f'week {seq_n+1}', fontsize=txt_size) #{ch}, 
            
            # ax[ch_n, seq_n].axvline(-1000,-20,20,color="red") 
            ax[ch_n, seq_n].axvline(0,-1,1,color="green")        # 畫時間起始線
            # ax[ch_n, seq_n].axvline(4000,-20,20,color="green")     # 畫時間結束線
            ax[ch_n, seq_n].plot(second_x,y1,'--',color="black")
            
            ax[ch_n, seq_n].plot(second_x,AD_mean,label='Vibration only',       color="blue")
            ax[ch_n, seq_n].plot(second_x,NF_mean,label='AOMI only',            color="red")
            ax[ch_n, seq_n].plot(second_x,FB_mean,label='AOMI with Vibration',  color="green")
            
            
            ax[ch_n, seq_n].tick_params(axis='x', labelsize=txt_size)
            ax[ch_n, seq_n].tick_params(axis='y', labelsize=txt_size)        
            ax[ch_n, seq_n].set_xlabel(f"(ms)",  x=1.05, fontsize=txt_size) 
            ax[ch_n, seq_n].set_ylabel(f"%", rotation=0, y=1.05, labelpad=-20, fontsize=txt_size) 
            
            ax[ch_n, seq_n].fill_between(second_x, 
                    np.array(AD_mean) - np.array(AD_std)/2, 
                    np.array(AD_mean) + np.array(AD_std)/2, color='lightblue',alpha=0.5)
            ax[ch_n, seq_n].fill_between(second_x, 
                    np.array(NF_mean) - np.array(NF_std)/2, 
                    np.array(NF_mean) + np.array(NF_std)/2, color='lightcoral', alpha=0.5)
            ax[ch_n, seq_n].fill_between(second_x, 
                    np.array(FB_mean) - np.array(FB_std)/2, 
                    np.array(FB_mean) + np.array(FB_std)/2, color='limegreen',alpha=0.5)
            
            
            ###########################
            # ax2[seq_n, ch_n].axis(ymin=-50,ymax=50)
            # ax2[seq_n, ch_n].axvline(-1000,-20,20,color="red")       # 畫時間起始線
            # ax2[seq_n, ch_n].axvline(0,-20,20,color="green")       # 畫時間起始線
            # # ax2[seq_n, ch_n].axvline(4000,-20,20,color="green")    # 畫時間結束線
            # ax2[seq_n, ch_n].plot(second_x,y1,'--',color="black")        
            # ax2[seq_n, ch_n].plot(second_x,mean_dif, label='差值',color="royalblue")
            # print('Drawing...03')
            
            # if LIMIT != 0:
            ax[ch_n, seq_n].axis(xmin=-TIME_LEN_REST,xmax=TIME_LEN_ACTION)
            ax[ch_n, seq_n].axis(ymin=-LIMIT,ymax=LIMIT)

            # if seq_n == 0:
            #     fig.legend(
            #         loc='upper right',
            #         fontsize=10,
            #         shadow=True,
            #         facecolor='#ccc',
            #         edgecolor='#000',
            #     )
            plt.legend(
                loc='upper left',
                fontsize=txt_size,
                shadow=True,
                facecolor='white',
                edgecolor='#000',
                bbox_to_anchor=(0, 11.5),
                )

        # 調整子圖之間的間距
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.suptitle(f'plot_team_{freq}_{side}', fontsize=txt_size*1.5)  # 添加整張圖的標題
        fig.savefig(f'{SAVE_PATH_0}/plot_team_{freq}_{side}.png', dpi=300)

        # fig2.suptitle(f'plot_team_差值{side_set}', fontsize=16)  # 添加整張圖的標題
        # fig2.savefig(f'{SAVE_PATH_0}/plot_team_diff_{side_set}.png', dpi=300) 
    
