import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import itertools

plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定支援中文的字型
plt.rcParams['axes.unicode_minus'] = False # 顯示負號

def progressBar(title, temp, skip, total): # 進度條
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

from scipy import stats
from scipy.stats import mannwhitneyu
# 適用於兩個樣本大小不同並且可能來自不同的母體分佈
# 原理是將兩組樣本的觀測值合併，並將它們從小到大排列，然後將排名相加，得到兩組樣本的秩次總和。
from scipy.stats import ranksums
# 適用於兩個樣本大小相等並且來自相同的母體分佈
from scipy.stats import wilcoxon
# 通常用於檢測一組受試者在兩個不同時間點或條件下的測量值是否有顯著的差異。

class Test(): # 顯著性計算
    def u_test(group1, group2):
        # 進行 Mann-Whitney U 檢定 /rank sum 檢定
        statistic, p_value = mannwhitneyu(group1, group2)
        # statistic, p_value = ranksums(group1, group2)

        # 顯示檢定結果
        # print("Mann-Whitney U 檢定統計量：", statistic)

        # 判斷 p 值是否小於顯著水準（例如 0.05），並進行結果解釋        
        if   p_value < 0.01: txt = '**'
        elif p_value < 0.05: txt = '*'
        else: txt = '  '
        return txt , p_value
        
    def signed_rank_test(before, after):
        # 使用 wilcoxon 函數進行檢定
        statistic, p_value = wilcoxon(before, after)
        if   p_value < 0.01: txt = '**'
        elif p_value < 0.05: txt = '*'
        else: txt = '  '
        return txt , p_value
            
    def linear_regress(data):
        time = np.array(np.arange(0, len_seq)) #[1, 2, 3, 4, 5]
        # print(time)
        # 使用線性回歸模型擬合數據
        slope, intercept, r_value, p_value, std_err = stats.linregress(time, data)

        # 檢查 p 值是否小於顯著性水平（例如 0.05）
        if   p_value < 0.01: txt = '**'
        elif p_value < 0.05: txt = '*'
        else: txt = '  '
        return time, txt , p_value, slope, intercept
   
class Plot_Page():

    def __init__(self, png_path):
        # self.num_boxes = len_seq  # 每張圖中箱子的數量 9 
        # self.avg_path  = avg_path 
        self.png_path = png_path
              
    def plot_average_bar(self, data_box_in0, freq, side, idx, lim_set, idx_n):
        # 畫出正規化的平均值變化圖(整頁)
        
        # data_in(9,8,2) # 將第一個和第二個維度按照順序合併
        # reshaped = np.reshape(data_in0, (3, 3, 8, 2))
        # data_in = np.average(reshaped, axis=1)
        # # [123-2. 345-5. 678-8.] 用後面的那個維度
        # print("合併後形狀:", merged.shape)3 8 2
        
        # data_box_in(5,9,8,3)
        reshaped_box    = np.reshape(data_box_in0, (5, 3, 3, 8, 3))
        reshaped_box2   = np.swapaxes(reshaped_box, 1, 2) # 用後面的那個維度
        data_box_in     = np.reshape(reshaped_box2, (-1, 3, 8, 3)) # [123-2. 345-5. 678-8.]  
        data_in         = np.average(data_box_in, axis=0)
        # print(data_in.shape) # 3 8 3
        
        save_csv(data_in, idx_n)
        
        page_width  = 3
        page_length = 3
        
        txt_size = 14
        title_size = 18

        week = int(len_seq/3)
        # NF/left
        figA, axsA = plt.subplots(page_length, page_width, figsize=(14,10))
        ch_set = 0
        for leg in range(page_length):
            for wid in range(page_width):
                
                if leg==0 and wid ==1: 
                    continue
                    
                # positions = np.arange(self.num_boxes)+1  # 調整盒狀圖位置              
                # axsA[leg][wid].set_ylabel(f"{index}", rotation=90, loc='top', labelpad=-20)

                axsA[leg][wid].set_xlabel(f"week", x=1.05 ,labelpad=-10, fontsize=txt_size)
                axsA[leg][wid].set_xticks(np.arange(week))  # 設定橫軸刻度位置
                axsA[leg][wid].set_xticklabels(np.arange(1, week+1), fontsize=txt_size)  # 設定橫軸刻度標籤 
                
                if   idx=='持續時間': ylabel = 'ms' 
                elif idx=='峰值瞬間': ylabel = 'ms' 
                elif idx=='峰值強度': ylabel = '%'   
                elif idx=='總能量':   ylabel = '%'  
                axsA[leg][wid].set_ylim(lim_set[idx_n,0], lim_set[idx_n,1]) 
                axsA[leg][wid].set_ylabel(f"{ylabel}", rotation=0, y=1.05, labelpad=-20, fontsize=txt_size) 
                axsA[leg][wid].tick_params(axis='y', labelsize=txt_size)
                axsA[leg][wid].set_title(f'  {CH_LIST[ch_set]} ', fontsize=txt_size)              
                
                # 計算標準差
                std_NF_list = []
                std_FB_list = []
                std_AD_list = []
                # print((data_box_in.shape),week)
                for seq in range(week): 
                    std0 = np.std(data_box_in[:,seq,ch_set,0])
                    std1 = np.std(data_box_in[:,seq,ch_set,1])
                    std2 = np.std(data_box_in[:,seq,ch_set,2])
                    # std_NF_list.append(std0*0.5) 
                    std_FB_list.append(std0*0.5)
                    std_NF_list.append(std1*0.5)
                    std_AD_list.append(std2*0.5)
                                    
                # 繪製長條圖
                x_pos = np.arange(week)
                bar_width = 0.25
                axsA[leg][wid].bar(x_pos - bar_width, data_in[:,ch_set,0], bar_width, yerr=[std_FB_list],
                                   facecolor='lightgreen', edgecolor='black',label='AOMI with Vibration')
                axsA[leg][wid].bar(x_pos, data_in[:,ch_set,1], bar_width, yerr=[std_NF_list],
                                   facecolor='lightcoral', edgecolor='black',label='AOMI only')
                axsA[leg][wid].bar(x_pos + bar_width, data_in[:,ch_set,2], bar_width, yerr=[std_AD_list],
                                   facecolor='lightskyblue', edgecolor='black',label='Vibration only')

                # 誤差線加入端點邊界
                # seq = 0
                err_line_wid = 0.1
                for team in range(len_team):
                    if   team==0 :
                        std_list = std_FB_list
                        std_width = bar_width 
                    elif team==1 :
                        std_list = std_NF_list
                        std_width = 0 
                    elif team==2 :
                        std_list = std_AD_list
                        std_width = -bar_width # 轉向
                        
                    for seq in range(week): 
                        xerr_L      = seq - std_width - err_line_wid/2
                        xerr_R      = seq - std_width + err_line_wid/2
                        yerr_top    = data_in[seq,ch_set,team] + std_list[seq]
                        yerr_bottom = data_in[seq,ch_set,team] - std_list[seq]
                        axsA[leg][wid].plot([xerr_L ,xerr_R], [yerr_top, yerr_top], color='black', linestyle='-', linewidth=1)
                        axsA[leg][wid].plot([xerr_L ,xerr_R], [yerr_bottom, yerr_bottom], color='black', linestyle='-', linewidth=1)
                
                # 差異標示：FB_3週皆較高/FB趨勢/NF趨勢
                # NF_incre = data_in[:,ch_set,0]
                # FB_incre = data_in[:,ch_set,1]
                # # incre_TS = np.average(data_in[:,ch_set,0])*0.05
                # # incre_TS = np.average(data_in[:,ch_set,1])*0.05
                # incre_TS = (lim_set[idx_n,1] - lim_set[idx_n,0])*0.03
                # incre_TS_Stay = (lim_set[idx_n,1] - lim_set[idx_n,0])*0.01
                # NF_FB_diff = data_in[:,ch_set,0]-data_in[:,ch_set,1]                
                # all_positive, NFis_increasing, FBis_increasing = self.cal_plot_diff(
                #     NF_incre, FB_incre, incre_TS, incre_TS_Stay, NF_FB_diff)
                axsA[leg][wid].set_title(f'{CH_LIST[ch_set]}', fontsize=txt_size)   
                
                # 利用 boxplot 計算 p_value # 顯著/
                for seq in range(week): 
                    P_title, p_value = Test.u_test(data_box_in[:,seq,ch_set,0], data_box_in[:,seq,ch_set,1])
                    text_set = lim_set[idx_n,0]*0.05+lim_set[idx_n,1]*0.95
                    axsA[leg][wid].text(seq, text_set, f'{P_title}    ', fontsize=12, # p={round(p_value, 3)}
                                        horizontalalignment='center', verticalalignment='top')       
                ch_set+=1  
                
        plt.subplots_adjust(hspace=0.4, wspace=0.3)  # 設置水平和垂直間距      

        plt.legend(
            loc='upper left',
            fontsize=txt_size-2,
            shadow=True,
            facecolor='white',
            edgecolor='#000',
            bbox_to_anchor=(0, 4.4),
            )

        if freq == 'A1': title_set = 'α band'
        if freq == 'B1': title_set = 'β band'        
        figA.suptitle(f'{side} side, {title_set}, {idx}', fontsize=title_size, y=0.05)
        
        
        plt.savefig(self.png_path, dpi=300)
        plt.close()
        # plt.show()
        # plt.clf()


import csv
def save_csv(csv_data_in, idx_n):  
             
    # if freq_n == 0 and side_n == 0 and idx_n == 0: reset_flag = 1
    # else: reset_flag = 0
    
    # reshaped = np.reshape(csv_data_in, (3, 3, 8, 3))        
    # plot_diff_save = np.average(reshaped, axis=1) #(3,8,3) data, ch, team
    
    reshaped_data = csv_data_in.reshape(3, 24)
    # 將第一行數據格式化為小數點後兩位
    if idx_n == 0:  #時間
        reshaped_data = np.around(reshaped_data, decimals=0).astype(int)
    elif idx_n == 1:#強度
        reshaped_data = np.around(reshaped_data, decimals=2)
    elif idx_n == 3:#能量
        reshaped_data = np.around(reshaped_data/1000, decimals=0).astype(int)

    with open(f'{SAVE_PATH_0}/Data_avg_output.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # 寫入每一行數據
        writer.writerow([TEAM_SET[0], TEAM_SET[1], TEAM_SET[2], 
                         TEAM_SET[0], TEAM_SET[1], TEAM_SET[2],
                         TEAM_SET[0], TEAM_SET[1], TEAM_SET[2],])
        writer.writerow(['F3',#'','',
                           '',#'','',
                         'F4'])
        for row in reshaped_data[:,0:6]:
            new_row = row[:3].tolist() + ['','','',] + row[3:].tolist()
            writer.writerow(new_row)
            
        writer.writerow(['C3',#'','',
                         'Cz',#'','',
                         'C4'])
        for row in reshaped_data[:,6:15]:
            writer.writerow(row) 
            
        writer.writerow(['P3',#'','',
                         'Pz',#'','',
                         'P4'])
        for row in reshaped_data[:,15:]:
            writer.writerow(row) 
        writer.writerow([f'{freq}, {side}, {idx}'])

        progressBar(PLOT_SET, testX, 1, 12)

def save_txt(csv_file,reset=0, *args): #接受任意数量的参数
    if reset==1:
        with open(csv_file, 'w') as file:
            writer = csv.writer(file)
    try:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for txt in args:
                writer.writerow(txt)
    except FileNotFoundError:
        with open(csv_file, 'w') as file:
            writer = csv.writer(file)
            
################################################

##### [程式用途：畫出指標分析圖] ##### 論文 p.15、p.19~28
# 輸入  ：ERDS_9seq_left.npy    / 資料夾：FILE_PATH
# 輸出01：plot_team_A1_left     / 資料夾：SAVE_PATH_0

# Npy_file = f'./Step4/Step4_6s/4A比較_ERD_change_8ch/left/ERDS_change_leftC3_NFaFB_avg.npy'
# NFaFB_avg = np.load(Npy_file)
# print(NFaFB_avg.shape) # [NF/FB, seq, index]

from _Run_set import FRERUENCY_LIST, SUBJECT_NAMES, SUBJECT_NAMES_NONE

SUBJECT_NAMES_NF = ['又諄','文豪','柏瑋','皓嚴','羅喨']
SUBJECT_NAMES_FB = ['嘉浚','浩軒','柏勛','松育','聖佶']
SUBJECT_NAMES_AD = ['建誌','柏崴','郁芹','璿瑋','寶心']

# TIME_LIST       = ['6s'] #['2s','4s','6s']
SIDE_SET_LIST   = ['left' ,'right'];                                   LEN_SIDE  = len(SIDE_SET_LIST)
INDEX_LIST      = ['持續時間','峰值強度','峰值瞬間','總能量'];          LEN_INDEX = len(INDEX_LIST) 
CH_LIST         = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'];    LEN_CH    = len(CH_LIST) 
TEAM_SET        = ['震動想像組','純想像組','純震動組']; 

TIME_reigon = '6s'
len_freq = 2 # A、B band
len_team = 3 # 3組
len_name = 5 # 1組5人
len_seq  = 9 # 9次實驗

# 0402 將9次分成三週重新畫
# 0403 要求繪製為柱狀圖
# 0417 確認趨勢
# PLOT_SET = ['diff', 'average','boxplot','startend']
PLOT_SET_LIST = ['average']
################################################
for PLOT_SET in PLOT_SET_LIST:
    Data_ALL_0 = np.zeros((len_freq, LEN_SIDE, len_team, len_name, len_seq, LEN_CH, LEN_INDEX)) 
    # (2,2,2,5,  9,8,4) 
    # (freq, side, team, name, seq, ch, idx)

    for freq_n, side_n in itertools.product(
        # range(len(TIME_LIST)),
        range(len_freq),
        range(LEN_SIDE),
        # range(LEN_CH)
        ):
        # range(len(SUBJECT_NAMES)
        # TIME_reigon = TIME_LIST[TIME_reigon_n]
        freq        = FRERUENCY_LIST[freq_n]
        side        = SIDE_SET_LIST[side_n]
        # ch          = CH_LIST[ch_n]
        # name        = SUBJECT_NAMES[name_n]
        
        name_n0 = 0
        name_n1 = 0
        name_n2 = 0
        if freq == 'A1': freq_set = 'A'
        if freq == 'B1': freq_set = 'B'
        
        for name_n, name in enumerate(SUBJECT_NAMES):        
            if name in SUBJECT_NAMES_NONE: TEAM = '備案'
            elif name in SUBJECT_NAMES_FB: 
                TEAM = TEAM_SET[0]; team_n = 0; 
            elif name in SUBJECT_NAMES_NF: 
                TEAM = TEAM_SET[1]; team_n = 1; 
            elif name in SUBJECT_NAMES_AD: 
                TEAM = TEAM_SET[2]; team_n = 2; 
            
            FILE_PATH   = f'./Step3_2/3{freq_set}比較_ERD_plot_9seq/{TEAM}/{name}'
            data_path = get_root(FILE_PATH, f'ERDS_9seq_{side}.npy')
            DATA = np.load(data_path[0]) # 9,8,4
            print(name, DATA.shape)

            if   team_n == 0: Data_ALL_0[freq_n, side_n, 0, name_n0] = DATA; name_n0+=1
            elif team_n == 1: Data_ALL_0[freq_n, side_n, 1, name_n1] = DATA; name_n1+=1
            elif team_n == 2: Data_ALL_0[freq_n, side_n, 2, name_n2] = DATA; name_n2+=1
            # Data_ALL_0[freq_n, side_n, 2, :] = Data_ALL_0[freq_n, side_n, 1, :]
            
    SAVE_PATH_0 = f'./Step5_6/比較_ERD_{PLOT_SET}_8ch/'
    # SAVE_PATH_csv = f'{SAVE_PATH_0}/output.csv'
    if not os.path.isdir(SAVE_PATH_0):os.makedirs(SAVE_PATH_0)
    np.save(f'{SAVE_PATH_0}/Data_all.npy', Data_ALL_0) 
    # print(Data_ALL_0) # 2 2 2 5 9 8 4
#--------------------------------------------------------------------------------------------------------
    Data_avg = np.average(Data_ALL_0, axis=3) # 2,2,2, 9,8,4
    # (freq, side, team, seq, ch, idx) # 取每組平均
    Data_box = np.swapaxes(Data_ALL_0, 2, 6) # 2,2,4, 5,9,8,2
    Data_avg = np.swapaxes(Data_avg,   2, 5) # 2,2,4, 9,8,2
    # (freq, side, idx, seq, ch, team) # 對 2 5 維進行交換
    
    lim_set = np.zeros((4,2))
    # 統一所有圖表上下限
    # for idx_n in range(LEN_INDEX):
    #     # print(np.min(Data_avg[:,:,idx_n]), np.percentile(Data_avg[:,:,idx_n], 5))
    #     lim_set[idx_n,0] = np.min(Data_avg[:,:,idx_n])*2 - np.percentile(Data_avg[:,:,idx_n], 1)  # 下百分位數
    #     lim_set[idx_n,1] = np.max(Data_avg[:,:,idx_n])*2 - np.percentile(Data_avg[:,:,idx_n], 99)  # 上百分位數

    with open(SAVE_PATH_0+'/Data_avg_output.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow([''])   
#--------------------------------------------------------------------------------------------------------
    testX = 1
    for freq_n, side_n, idx_n in itertools.product(
        range(len_freq),
        range(LEN_SIDE),
        range(LEN_INDEX)):
        freq = FRERUENCY_LIST[freq_n]
        side = SIDE_SET_LIST[side_n]
        idx  = INDEX_LIST[idx_n]
        
        if freq == 'A1': freq_set = 'A'
        if freq == 'B1': freq_set = 'B'  
        
        if idx == '峰值瞬間': continue # 不使用
        
        # print(Data_avg.shape)        
        Data_in     = Data_avg[freq_n, side_n, idx_n] # Data_in(9, 8, 2) 
        Data_box_in = Data_box[freq_n, side_n, idx_n] # Data_box_in(5, 9, 8, 2) 
        # Data_diff_in= Data_in[:,:,1]-Data_in[:,:,0] # Data_in(9, 8) 
        # print(Data_box_in.shape)      
                
        avg_path = f'{SAVE_PATH_0}/{freq_set}_{side}_{idx}.png'# 平均變化圖
        # if PLOT_SET == 'boxplot': 
        #     lim_set[idx_n,0] = np.percentile(Data_box[freq_n, :,idx_n], 0)*1.1-np.percentile(Data_box[freq_n, :,idx_n], 100)*0.1
        #     lim_set[idx_n,1] = np.percentile(Data_box[freq_n, :,idx_n], 100)              
        #     Plot_Page(avg_path).plot_boxplot(Data_box_in, freq, side, idx, lim_set, idx_n)
        # el
        if PLOT_SET == 'average': 
            lim_set[idx_n,0] = np.percentile(Data_avg[freq_n, :,idx_n], 0)
            lim_set[idx_n,1] = np.percentile(Data_avg[freq_n, :,idx_n], 100)  
            Plot_Page(avg_path).plot_average_bar(Data_box_in, freq, side, idx, lim_set, idx_n)

           
        # if freq_n == 0 and side_n == 0 and idx_n == 0: reset_flag = 1
        # else: reset_flag = 0
        # reshaped = np.reshape(Data_in, (3, 3, 8, 3))        
        # plot_diff_save = np.average(reshaped, axis=1) #(3,8,2) data, ch, team
        # plot_diff_save = np.transpose(plot_diff_save, (1, 2, 0))# 转置维度 ch, team, data
        # save_txt(f'{SAVE_PATH_0}/average_data.txt',reset_flag, [freq], [side], [idx], [plot_diff_save])
                
        progressBar(PLOT_SET, testX, 1, 12)
        
        testX+=1
        # if testX == 2: break
