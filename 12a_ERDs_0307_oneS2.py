import os
import time
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, lfilter, savgol_filter

# print ("time:", time.process_time())

# %% # ===========================[濾波器+]============================ # 
class Decoder():
    def __init__(self):
        self.comp = 8388607  # complement (7FFFFF)??why

    def HEX_to_DEC(self, hexdata):  # 16轉10進制
        length = len(hexdata)
        num = 0
        for i in range(0, length, 2):
            data_short = hexdata[i] + hexdata[i + 1]

            data_dec = int(data_short, 16)
            num = num + data_dec * (256 ** (((length - i) / 2) - 1))
        return num
    def get_BCI(self, raw_data):
        # 8ch + framid + trigger
        datalength = len(raw_data)
        n = (datalength // 64)  # 確認幾組資料，每個封包的大小是64，所以n為封包數目
        output = np.zeros([n, 10])  # 返回用0填充的數組 10*n的陣列，9+trigger
        ch1 = np.zeros([n])
        ch2 = np.zeros([n])
        ch3 = np.zeros([n])
        ch4 = np.zeros([n])
        ch5 = np.zeros([n])
        ch6 = np.zeros([n])
        ch7 = np.zeros([n])
        ch8 = np.zeros([n])
        frameID = np.zeros([n]) 
        trigger = np.zeros([n])
        
        i = 0
        j = 0
        while i < len(raw_data) - 62:
            # 確認頭尾封包(頭2bytes:5170 ; 尾1byte: a1)
            if raw_data[i:i + 4] == "5170" and raw_data[i + 60:i + 62] == "a1":
                # 擷取每筆資料的 frame_id

                frameID[j] = self.HEX_to_DEC(raw_data[i + 4:i + 6])  # 2
                output[j, 8] = frameID[j]

                # 擷取每筆資料的 ch1~ch8
                ch1[j] = self.HEX_to_DEC(raw_data[i + 10:i + 16])  # 6
                if ch1[j] > self.comp:  # 讓他有正有負
                    ch1[j] = ch1[j]-2*self.comp
                output[j, 0] = ch1[j]

                ch2[j] = self.HEX_to_DEC(raw_data[i + 16:i + 22])  # 6
                if ch2[j] > self.comp:
                    ch2[j] = ch2[j]-2*self.comp
                output[j, 1] = ch2[j]

                ch3[j] = self.HEX_to_DEC(raw_data[i + 22:i + 28])  # 6
                if ch3[j] > self.comp:
                    ch3[j] = ch3[j]-2*self.comp
                output[j, 2] = ch3[j]

                ch4[j] = self.HEX_to_DEC(raw_data[i + 28:i + 34])  # 6
                if ch4[j] > self.comp:
                    ch4[j] = ch4[j]-2*self.comp
                output[j, 3] = ch4[j]

                ch5[j] = self.HEX_to_DEC(raw_data[i + 34:i + 40])  # 6
                if ch5[j] > self.comp:
                    ch5[j] = ch5[j]-2*self.comp
                output[j, 4] = ch5[j]

                ch6[j] = self.HEX_to_DEC(raw_data[i + 40:i + 46])  # 6
                if ch6[j] > self.comp:
                    ch6[j] = ch6[j]-2*self.comp
                output[j, 5] = ch6[j]

                ch7[j] = self.HEX_to_DEC(raw_data[i + 46:i + 52])  # 6
                if ch7[j] > self.comp:
                    ch7[j] = ch7[j]-2*self.comp
                output[j, 6] = ch7[j]

                ch8[j] = self.HEX_to_DEC(raw_data[i + 52:i + 58])  # 6
                if ch8[j] > self.comp:
                    ch8[j] = ch8[j]-2*self.comp
                output[j, 7] = ch8[j]

                trigger[j] = self.HEX_to_DEC(raw_data[i + 58:i + 60])  # trigger
                if trigger[j] > self.comp:
                    trigger[j] = trigger[j]-2*self.comp
                output[j, 9] = trigger[j]

                i += 64  # 一組資料32bytes(每讀完一組平移32bytes)
                j += 1  # 每組整理好後的資料
            else:
                i += 2  # 若沒有讀到頭尾封包，往後找1byte
            progressBar("解碼進度", j,1000, n-1)

        output[:, :8] = output[:, :8] * 2.5 / (2**23 - 1)  # 將收到的data換成實際電壓
        return output  # 八通道腦波資料(每個通道以十進制存), shape = (n, 10)  
    def decode(self, eeg_txt_path):
        # decode EEG.txt and filter
        # input
        #   eeg_txt_path : file path of raw HEX eeg data, e.g. ./exp/2022_10_20_XXXX/1/EEG.txt
        # return 
        #   eeg_filtered : filtered eeg data with shape = (n, 10)
        
        f = open(eeg_txt_path, 'r')
        raw = f.read()
        f.close()

        eeg_raw = self.get_BCI(raw)
        F = Filter(4, 40)
        eeg_filtered = F.filter(eeg_raw)
  
        return eeg_filtered

    def decode_to_txt_npy(self, eeg_txt_path, return_data = False):
        # decode, filtering EEG.txt and then write into 1.txt
        # if return_data == True : return filtered eeg data with shape = (n, 10) 
        print("\n Process data \n{}\n".format(eeg_txt_path))
        print("Decoding...")

        start_t = time.time()
        eeg_filtered = self.decode(eeg_txt_path)

        f = open('/'.join(eeg_txt_path.split('/')[0:-1]) + '/1.txt', 'w')  # 打開1.txt檔案
        for i in range(np.size(eeg_filtered, 0)):
            for j in range(np.size(eeg_filtered, 1)):
                f.write(str(eeg_filtered[i][j]))
                f.write('\t')
            f.write('\n')
        f.close() 
    
        #存成.npy
        np.save('/'.join(eeg_txt_path.split('/')[0:-1]) + '/1.npy', eeg_filtered)
        print('Cost: {0:.1f} s'.format(int((time.time() - start_t) * 10) * 0.1))
        print("Shape of decoded EEG data: ({}, {})".format(np.size(eeg_filtered, 0), np.size(eeg_filtered, 1)))
        print('-'*40)
        # self.plot_eeg(eeg_filtered)

        if return_data:
            return eeg_filtered   

class Filter():
    def __init__(self, hp_freq_in, lp_freq_in):
        self.fs = 1000
        # self.hp_freq = 4
        # self.lp_freq = 40

        self.hp_freq = hp_freq_in
        self.lp_freq = lp_freq_in
    def filter(self, eeg_raw):
        # 濾波 : 60Hz市電，0.5 ~ 50 Hz 帶通濾波取出腦波頻率範圍
        # input 
        #   eeg_raw : shape = (n, 10)，8通道+frameID+Trigger
        # output 
        #   eeg_raw : shape = (n, 10) 
  
        for i in range(8): # ch1 ~ ch8
            # 60 Hz notch
            eeg_raw[:, i] = self.butter_bandstop_filter(eeg_raw[:, i], 55, 65, self.fs)
            # 0.5 & 50 Hz filter
            eeg_raw[:, i] = self.butter_bandpass_filter(eeg_raw[:, i], self.hp_freq, self.lp_freq, self.fs) 

        return eeg_raw 
            
    def butter_bandpass(self, lowcut, highcut, fs, order=3):  # fs & order??#EEG:3，EMG:6
        nyq = 0.5 * fs
        # nyquist frequency(fs is the sampling rate, and 0.5 fs is the corresponding Nyquist frequency.)
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandstop(self, lowcut, highcut, fs, order=3):  # 55~65Hz
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='stop')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_bandstop_filter(self, data, lowcut, highcut, fs, order=3):
        b, a = self.butter_bandstop(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

# import matplotlib.patches as patches
class FFT():            
    def fft(eeg_data, Fs=1000.0, Ts=1.0/1000.0):
        # eeg_data : with shape = (n, ) which n = number of sample points
        # Fs : sampling rate取樣率
        # Ts : sampling interval 取樣區間 (1/Fs)

        y = eeg_data

        t = np.arange(0, len(y)/Fs, Ts)  # time vector,這裡Ts也是步長

        n = len(y)     # length of the signal
        k = np.arange(n)
        T = n/Fs
        freq = k/T     # two sides frequency range
        freq1 = freq[range(int(n/20))]  # one side frequency range

        #YY = np.fft.fft(y)   # 未歸一化
        Y = np.fft.fft(y)/n   # fft computing and normalization 歸一化
        Y1 = Y[range(int(n/20))]
        return t, freq1, Y1

    # def normal_distribution(self, x,mu,sig):
    #     sig = sig*sig
    #     ans = 1 / (sig* (2*math.pi)**0.5) * math.exp(- (x-mu)**2 / (2*sig**2))
    #     return ans         

    # def plot_fft(self, eeg_data):
    #     # eeg_data : eeg data with shape = (8, n) which
    #     #            8 is the number of channel,
    #     #            n is the number of sample points
    #     # num_channel : number of channels 
    #     num_channel=8

    #     label_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']

    #     fig, ax = plt.subplots(num_channel, 2, figsize=(15,num_channel*2))

    #     for channel in range(num_channel):
    #         # y = eeg_data[channel,:]
    #         y = eeg_data[channel]
    #         t, freq, Y = self.fft(y)

    #         # plot raw data
    #         ax[channel, 0].plot(t, y, linewidth=0.5, color='steelblue')
    #         ax[channel, 0].set_ylabel(label_list[channel], fontsize=14, rotation=0) # 'Amplitude'
    #         ax[channel, 0].yaxis.set_label_coords(-.2, .4)
    #         # plot spectrum        
    #         ax[channel, 1].axis(xmin=0,xmax=30)   # 頻譜上下限
    #         ax[channel, 1].plot(freq, abs(Y), linewidth=0.5, color='steelblue')        
    #         ax[channel, 1].set_ylabel(label_list[channel], fontsize=14, rotation=0) # '|Y(freq)|'
    #         ax[channel, 1].yaxis.set_label_coords(1.1, .4)

    #         # remove x label except the bottom plot
    #         if (channel + 1 != num_channel):
    #             ax[channel, 0].axes.xaxis.set_ticklabels([])
    #             ax[channel, 1].axes.xaxis.set_ticklabels([])
    #         else:
    #             ax[channel, 0].set_xlabel('Time')
    #             ax[channel, 1].set_xlabel('Freq (Hz)')
            
    #         # mark the range of alpha band (8 - 12 Hz)
    #         rect = patches.Rectangle((8,0),
    #                         width=4,
    #                         height=np.max(abs(Y)),
    #                         facecolor = 'orange',
    #                         fill = True,
    #                         alpha = 0.2)             
    #         ax[channel, 1].add_patch(rect)

    #         # mark the range of alpha band (8 - 12 Hz)
    #         # rect = patches.Rectangle((0,0),
    #         #                 width=4,
    #         #                 height=np.max(abs(Y)),
    #         #                 facecolor = 'red',
    #         #                 fill = True,
    #         #                 alpha = 0.2)             
    #         # ax[channel, 1].add_patch(rect)

    #     ax[0, 0].set_title('Raw EEG', fontsize=14)
    #     ax[0, 1].set_title('FFT', fontsize=14)

        
    #     # data_date = get_latest_date('./exp')
    #     png_path1 = DATA_PATH +'/FFT.png'
    #     plt.savefig(png_path1, dpi=300)
  
# %% # ===========================[Get_Data]============================ # 

def data_point_loss(data_point_check):
    for i in range(len(data_point_check)-1):
        if ((data_point_check[i+1] - data_point_check[i]) != 1) and (
            data_point_check[i] != 255 ):            
            # print(data_point_check[i-1:i+2])    
            return False
    return True

# EEG_antibias_filter(act, npy_data, 1, 4,   3, 4)
import statistics
class EEG_antibias_filter():

    def __init__(self, act, hp_freq, lp_freq):
        self.act = act
        self.hp_freq = hp_freq
        self.lp_freq = lp_freq

        # self.std_treh = std_treh
        # self.cont_treh = cont_treh

    def anti_bias(self, data):
        # print("004",np.shape(data)) # (15, 8, 9000)
        data = Filter(hp_freq, lp_freq).filter(data)
        data = self.get_action(data)
                
        data_trials_set = len(data)
        data_ch_set     = len(data[0])
        time_00         = TIME_LEN_REST
        
        # 計算休息資料
        Y_sum = np.zeros((data_trials_set, data_ch_set))
        for trials in range(data_trials_set):
            for channel in range(data_ch_set):

                # y = eeg_data[channel,:]
                y = data[trials][channel][time_00 - rest_on:time_00 - rest_off]
                # FFT
                t, freq, Y = FFT.fft(y)
                # 平方轉能量
                Y = Y**2
                # 濾波區間(已濾波) + 加總
                Y_sum[trials][channel] = np.sum(Y[self.hp_freq:self.lp_freq]).real

        # 計算標準差，所有在休息時段超出此閾值的資料都被丟棄
        # Y_sum_stdev = np.zeros((data_trials_set, data_ch_set))
        # bias_counter = 0
        bias_list = []
        for trials in range(data_trials_set):
            ch_check = 0
            for channel in range(data_ch_set):
                Y_sum_avg   = np.average(Y_sum[:,channel])
                Y_sum_stdev = statistics.stdev(Y_sum[:,channel])

                # 閥值設定為3倍標準差
                if (abs(Y_sum[trials][channel] - Y_sum_avg) > 3 * Y_sum_stdev):
                 #and (
                    # channel==3 or channel==5):    
                    # 指定通道                
                    # print('BIAS',trials, channel)
                    # bias_counter+=1
                    ch_check += 1

            bias_list.append(ch_check)

        # print(Y_sum.shape)
        # ####################################################################

        
        # # data_trials_set_act = data_trials_set - bias_counter
        # Y_sum_all  = np.zeros((data_trials_set, data_ch_set))
        # Y_sum_rest = np.zeros((data_trials_set, data_ch_set))
        # for trials in range(data_trials_set):

        #     # 重新計算剩餘休息資料
        #     if bias_list[trials] == 0: 
        #         for channel in range(data_ch_set):
        #             # y = eeg_data[channel,:]
        #             y_rest = data[trials][channel][time_00 - rest_on:time_00 - rest_off]
        #             # FFT
        #             t, freq, YR = FFT.fft(y_rest)
        #             # 平方轉能量
        #             YR = YR**2
        #             # 濾波區間(已濾波) + 加總
        #             Y_sum_rest[trials][channel] = np.sum(YR[self.hp_freq:self.lp_freq]).real

        #         # 重新計算所有資料
        #         for channel in range(data_ch_set):
        #             # y = eeg_data[channel,:]
        #             y = data[trials][channel]
        #             # FFT
        #             t, freq, Y = FFT.fft(y)
        #             # 平方轉能量
        #             Y = Y**2
        #             # 濾波區間(已濾波) + 加總
        #             Y_sum_all[trials][channel]  = np.sum(Y[self.hp_freq:self.lp_freq]).real
        
        # # print(Y_sum_act.shape)

        # # 計算標準差，所有時段超出此閾值的資料都被丟棄
        # # Y_sum_stdev = np.zeros((data_trials_set, data_ch_set))
        # # bias_counter_act = 0
        # bias_list_act    = []
        # for trials in range(data_trials_set):
        #     ch_check = 0
        #     if bias_list[trials] == 0: 

        #         for channel in range(data_ch_set):
        #             Y_sum_avg   = np.average(Y_sum_all[:,channel])
        #             Y_sum_stdev = statistics.stdev(Y_sum_rest[:,channel])

        #             # 閥值設定為3倍標準差
        #             if (abs(Y_sum_all[trials][channel] - Y_sum_avg) > 3 * Y_sum_stdev):

        #                 # bias_counter_act+=1
        #                 ch_check += 1

        #     bias_list_act.append(ch_check)

        # # print(bias_list)
        # # print(bias_list_act)

        # # 合併
        # for i in range(len(bias_list)):
        #     bias_list[i] = bias_list[i] + bias_list_act[i]

        return bias_list

    def get_action(self, data):
        
        # INPUT:  data = (N,10)
        data_trig = np.zeros((0, 8, TIME_LEN_ALL))

        act_start = 0
        count_bad = 0
        counter   = 0
        loss_counter = 0
        for sample in range(len(data)):
            progressBar("Get_Action", sample+1, 1000, len(data))
            data_of_trigger = data[sample][9]
        
            if (self.trigger_condition(data_of_trigger)) and (
                sample + TIME_LEN_ACTION < len(data)) and (
                sample - TIME_LEN_ACTION > 0) and ( # 1024 add
                sample > act_start+5) :                                       
                act_start = sample

                rest_start  = act_start - TIME_LEN_REST
                act_end     = act_start + TIME_LEN_ACTION

                # data_cut = data.T[ 0:8 , rest_start : act_end ]

                data_cut0 = data[rest_start : act_end , :]
                data_cut = data_cut0.T[ 0:8,:]

                if (data_point_loss(data_cut0[:, 8])) != True:
                    loss_counter+=1
                else:
                    counter+=1
                    
                    data_cut    = np.expand_dims(data_cut, axis=0)
                    data_trig   = np.concatenate([data_trig, data_cut], axis=0)                    

        return data_trig
    
    def trigger_condition(self, data_of_trigger):

        # if  TRIGGER_SET == 'new04': 
        if data_of_trigger>= 249:
            if   self.act == 'AO':
                if   (self.trigger_condition_if(data_of_trigger, 'left',  249)): return True #$#
                elif (self.trigger_condition_if(data_of_trigger, 'right', 252)): return True
            elif self.act == 'MI':
                if   (self.trigger_condition_if(data_of_trigger, 'left',  251)): return True
                elif (self.trigger_condition_if(data_of_trigger, 'right', 249)): return True
        else: return False

    def trigger_condition_if(self, data_of_trigger, sideset, trigset):
        if (SIDE_SET == sideset) and (data_of_trigger == trigset):
            return True
        return False

class Get_Data(): 

    def get_data(self, act, npy_data_path):
        print(f'---------{act}---------')

        if MAX_EEG != 0:
            self.act = act
            act_data = np.zeros((0, 8, TIME_LEN_ALL))
            npy_data = np.load(npy_data_path)

            bias_list_M = []
            # print(np.shape(npy_data))  
            npy_data = Filter(hp_freq, lp_freq).filter(npy_data) 

            npy_data, bias_list_M = self.get_action(npy_data, bias_list_M)
            act_data = np.concatenate([act_data, npy_data], axis=0)

        ############################

        elif MAX_EEG == 0:
            self.act = act
            act_data = np.zeros((0, 8, TIME_LEN_ALL))
            npy_data = np.load(npy_data_path)

            # 尋找雜訊
            bias_list_H = EEG_antibias_filter(act, 30, 48).anti_bias(npy_data)
            bias_list_L = EEG_antibias_filter(act, 1, 4).anti_bias(npy_data)
            bias_list_M = []
                
            # 分割資料
            npy_data = Filter(hp_freq, lp_freq).filter(npy_data) 
            npy_data, bias_list_M = self.get_action(npy_data, bias_list_M)
            # print(bias_list_H, bias_list_L, bias_list_M)  

            # 使用布尔索引删除 npy_data 对应位置的元素
            bias_list_bool = []
            for i in range(len(bias_list_H)):
                if bias_list_H[i] > 0 or bias_list_L[i] > 0 or bias_list_M[i] > 0:
                    bias_list_bool.append(1)
                else:
                    bias_list_bool.append(0)
            print(bias_list_bool)  

            boolean_mask = np.array(bias_list_bool, dtype=bool)
            new_npy_data = npy_data[~boolean_mask]

            # print(np.shape(new_npy_data)) 
            act_data = np.concatenate([act_data, new_npy_data], axis=0)
 
        return act_data, bias_list_bool

    def get_action(self, data , bias_list_M):
        
        # INPUT:  data = (N,10)
        data_trig = np.zeros((0, 8, TIME_LEN_ALL))

        act_start = 0
        count_bad = 0
        counter   = 0
        loss_counter = 0
        for sample in range(len(data)):
            progressBar("Get_Action", sample+1, 1000, len(data))
            data_of_trigger = data[sample][9]
        
            if (self.trigger_condition(data_of_trigger)) and (
                self.MI_FB_condition(data, sample)) and (
                sample + TIME_LEN_ACTION < len(data)) and (
                sample - TIME_LEN_ACTION > 0) and ( # 1024 add
                sample > act_start+5) :                                       
                act_start = sample

                rest_start  = act_start - TIME_LEN_REST
                act_end     = act_start + TIME_LEN_ACTION

                # data_cut = data.T[ 0:8 , rest_start : act_end ]

                data_cut0 = data[rest_start : act_end , :]
                data_cut = data_cut0.T[ 0:8,:]

                if (data_point_loss(data_cut0[:, 8])) != True:
                    loss_counter+=1
                else:
                    counter+=1
                    # print(data_cut.shape)
                    if self.cut_bad_2(data_cut) == False:
                        count_bad += 1
                        bias_list_M.append(1)
                    else:
                        bias_list_M.append(0)
                    data_cut    = np.expand_dims(data_cut, axis=0)
                    data_trig   = np.concatenate([data_trig, data_cut], axis=0)   
             
        return data_trig, bias_list_M
    
    def MI_FB_condition(self, data, sample):
        # 避免 MI、FB 混淆：
        # 因為 MI 取後2+0.5秒結束，FB 取前2秒開始 共4.5s->5s
        
        if (self.act == 'MI'):
            data_FBMI_conf  = data[sample + 2000 : sample + 5000, 9]
            data_FBMI_conf2 = data[sample - 2520 : sample - 1500, 9]
            # print(data_FBMI_conf.shape)
            if   254 in data_FBMI_conf  :return 0
            elif 254 in data_FBMI_conf2 :return 0
            elif 253 in data_FBMI_conf  :return 0    
            elif 253 in data_FBMI_conf2 :return 0  

        return True       

    def trigger_condition(self, data_of_trigger):

        # if  TRIGGER_SET == 'new04': 
        if data_of_trigger>= 249:
            if   self.act == 'AO':
                if   (self.trigger_condition_if(data_of_trigger, 'left',  249)): return True
                elif (self.trigger_condition_if(data_of_trigger, 'right', 252)): return True
            elif self.act == 'MI':
                if   (self.trigger_condition_if(data_of_trigger, 'left',  251)): return True
                elif (self.trigger_condition_if(data_of_trigger, 'right', 249)): return True
        else: return False

    def trigger_condition_if(self, data_of_trigger, sideset, trigset):
        if (SIDE_SET == sideset) and (data_of_trigger == trigset):
            return True
        return False

    def cut_bad_2(self, data_cut):

        if MAX_EEG == 0:
            return True

        # 移除過大資料
        for samp in range(len(data_cut[0])):
            for ch in range(len(data_cut)):
                if (abs(data_cut[ch][samp]) >= MAX_EEG):                    
                    return False
        
        return True

    def smooth_func(self, a, WSZ):
        # a: NumPy 1-D array containing the data to be SMOOTHed
        # WSZ: SMOOTHing window size needs, which must be odd number,
        # as in the original MATLAB implementation
        out0 = np. convolve(a, np. ones(WSZ, dtype=int), 'valid') / WSZ
        r = np. arange(1, WSZ - 1, 2)
        start = np. cumsum(a[:WSZ - 1])[:: 2] / r
        stop = (np. cumsum(a[:-WSZ:- 1])[:: 2] / r)[::- 1]
        return np. concatenate((start, out0, stop))

class plotERD(): 
    def get_ERD(self, data, hp_freq, lp_freq):

        # 取平方
        for i in range(len(data)):
            for ch in range(len(data[0])):
                for j in range(len(data[0][0])):
                    data[i, ch, j] = math.pow(data[i, ch, j], 2)

        # 對試驗取平均
        data = np.average(data,axis=0)
        # print("005",np.shape(data))

        # 平滑處理1
        # if SMOOTH > 1:
        #     # for i in range(len(data)):
        #     for ch in range(len(data)):            
        #         data[ch,:] = savgol_filter(data[ch,:], SMOOTH, 5)

        # # 平滑處理2
        # SMOOTH = 200
        new_cut_data = np.zeros((8,TIME_LEN_ALL-SMOOTH+1))
        for i in range(8):
            new_cut_data[i] = np.convolve(data[i], np.ones(SMOOTH), 'valid') / SMOOTH
        data = new_cut_data

        # # # 平滑處理3
        # for i in range(len(data)):
        #     data[i,:] = self.smooth_func(data[i,:], 5)

        # 取休息的平均，取資料動作開始前 3 - 1 秒當作休息
        data_rest = data[:, TIME_LEN_REST - rest_on : TIME_LEN_REST - rest_off] 

        # 休息的平均        
        data_rest = np.average(data_rest,axis = 1)
        # print("002",np.shape(data_rest)) 
        # C3=((C3-C3m)/C3m)*1
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i,j] = (data[i,j] - data_rest[i])/data_rest[i]

        # 休息時的平均能量 = 0%
        # 休息時的平均能量 * 2 = 100%
        return data

    def Plot_ERDS(self, ACT, new_data, png_path):

        # print('01', self.new_data[0][10])
        # print("004",np.shape(new_data))                
        ch_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
        fig, ax = plt.subplots(3, 3, figsize=(18,9))
        plt.xlim(-TIME_LEN_REST,TIME_LEN_ACTION)
        
        # values01 = np.zeros((TIME_LEN_ALL-100+1))
        # for i in range(len(values01)):
        #     print(range(10))
        col = 0
        for ch in range(len(ch_list)):
            
            ax[0, 1].set_title(ACT + ' data', fontsize=20)            
            x  = np.linspace(0, len(new_data[0]), len(new_data[0]))    
            y1 = np.zeros(len(new_data[0]))
            y2 = new_data[ch]
            # print('02', y1.shape)

            for i in range(len(y2)):
                y2[i] = y2[i]*100
            # print('03', new_data[0][10])


            if (ch == 0):
                ax[0, 0].plot(x-TIME_LEN_REST,y1,'--',color="black")
                ax[0, 0].plot(x-TIME_LEN_REST,y2,label=ch_list[ch])
                ax[0, 0].legend(loc='lower right')
                if LIMIT != 0:
                    ax[0, 0].axis(ymin=-LIMIT,ymax=LIMIT)
            elif (ch == 1):
                ax[0, 2].plot(x-TIME_LEN_REST,y1,'--',color="black")
                ax[0, 2].plot(x-TIME_LEN_REST,y2,label=ch_list[ch])
                ax[0, 2].legend(loc='lower right')
                if LIMIT != 0:
                    ax[0, 2].axis(ymin=-LIMIT,ymax=LIMIT)
            else:
                raw = int((ch+1)/3)
                col = (ch+1) % 3
                # print(ch,raw)
                ax[raw, col].plot(x-TIME_LEN_REST,y1,'--',color="black")
                ax[raw, col].plot(x-TIME_LEN_REST,y2,label=ch_list[ch])
                ax[raw, col].legend(loc='lower right')
                if LIMIT != 0:
                    ax[raw, col].axis(ymin=-LIMIT,ymax=LIMIT)            

            # 找出大於/小於均值的點，用於上色
            rS = np.zeros((len(y1)))
            for i in range(len(y1)):
                if(y2[i]>y1[i] and i>0):
                    rS[i] = True
                else:
                    rS[i] = False

            rD = np.zeros((len(y1)))
            for i in range(len(y1)):
                if(y2[i]<=y1[i] and i>0):
                    rD[i] = True
                else:
                    rD[i] = False

            if (ch == 0):
                ax[0, 0].axvline(0,-20,20,color="green") # 畫時間起始線
                ax[0, 0].fill_between(x-TIME_LEN_REST,y1,y2,where=rD,facecolor='blue',interpolate=True) #y1基準，y2_ERDs
                ax[0, 0].fill_between(x-TIME_LEN_REST,y1,y2,where=rS,facecolor='red',interpolate=True)                        
                ax[0, 0].set_xlabel('time(ms)')
                ax[0, 0].set_ylabel('ERD %')  
            elif (ch == 1):
                ax[0, 2].axvline(0,-20,20,color="green") # 畫時間起始線
                ax[0, 2].fill_between(x-TIME_LEN_REST,y1,y2,where=rD,facecolor='blue',interpolate=True) #y1基準，y2_ERDs
                ax[0, 2].fill_between(x-TIME_LEN_REST,y1,y2,where=rS,facecolor='red',interpolate=True)                        
                ax[0, 2].set_xlabel('time(ms)')
                ax[0, 2].set_ylabel('ERD %')  
            else:
                raw = int((ch+1)/3)
                col = (ch+1) % 3
                # print(ch,raw)                
                ax[raw, col].axvline(0,-20,20,color="green") # 畫時間起始線
                ax[raw, col].fill_between(x-TIME_LEN_REST,y1,y2,where=rD,facecolor='blue',interpolate=True) #y1基準，y2_ERDs
                ax[raw, col].fill_between(x-TIME_LEN_REST,y1,y2,where=rS,facecolor='red',interpolate=True)                        
                ax[raw, col].set_xlabel('time(ms)')
                ax[raw, col].set_ylabel('ERD %')     
        
        plt.savefig(png_path, dpi=300)
        # plt.show()
        plt.close()
        # print(np.shape(new_data))
        # print('011', new_data[0][10])

        return new_data
     
# %% # ===========================[其他+]============================ # 

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

def progressBar(title, temp, skip, total):
    if temp % skip == 0 or temp + skip >= total:
        print('\r' + '['+title+']:[%s%s] %s(%.2f%%)' % ('█' * int(temp/total*20), ' ' * (20-int(temp/total*20)), str(temp)+'/'+str(total), float(temp/total*100)), end='')
        if temp == total:
            print ('')

def get_latest_date(exp_path):
    # exp_path : path of exp directory. e.g. './exp'
    # return : latest date in exp_path. e.g. '2022_01_01_0000'

    exp_date = [item for item in os.listdir(exp_path) if os.path.isdir('{}/{}'.format(exp_path, item))]
    exp_date_int = [int("".join(item.split("_"))) for item in exp_date]
    data_date = exp_date[np.argmax(exp_date_int)]

    return data_date

def check_decoder():
    eeg_data_path = get_root(FILE_PATH, 'EEG.txt')  # 找出每個腦波txt檔 
    npy_data_path = get_root(FILE_PATH, '.npy')  # 找出每個腦波npy檔  
    for i in range(len(eeg_data_path)):
        len_decode = 0
        if len(eeg_data_path) != len(npy_data_path):
            len_decode = True        
        if (DECODE_SWITCH == 1) or (len_decode):
            # 解碼16進制EEG.txt資料至10進制1.txt資料，並存成.npy
            decode = Decoder().decode_to_txt_npy(eeg_data_path[i], True)

def save_path_set(act_str, npy_data_path, hp_freq, lp_freq):
    date  = str(pathlib.Path(npy_data_path).parent.parent.resolve())+'/'  
    path1 = f'{SAVE_PATH_0}/{date[-16:]}'
    # print('save_path_set', path1)

    tr = int(TIME_LEN_REST/1000) ;    ta = int(TIME_LEN_ACTION/1000)

    if not os.path.isdir(path1):os.makedirs(path1)
    save_path = f'{path1}/ERDs_{SIDE_SET}_T{tr}{ta}_Hz{hp_freq}{lp_freq}.png'

    return path1, save_path

# 打開檔案並將新列表寫入
def save_txt(existing_file_path, new_list):
    # 檢查檔案是否存在，若不存在則新增空白檔案
    if not os.path.isfile(existing_file_path):
        with open(existing_file_path, 'w'):  
            pass 
    with open(existing_file_path, 'a') as file:
        file.write('\n')  # 在舊資料的末尾添加新行
        # for item in new_list:
        file.write(f"{new_list}\n")

def clear_file(existing_file_path, txtset):
    if os.path.isfile(existing_file_path) and txtset == 1:
        with open(existing_file_path, 'w') as file:
            file.truncate(0)  # 清空檔案內容


# %% # ===========================[MAIN+]============================ #  

# 解碼完濾出 ERD、ERS (174) 
# 動作時間 4秒 
# cut bad data (179)
# 0102 加入0秒線，刺激前上色
# 0122 trigger 更換
# 0202 trigger 設定
# 0203 擴增頻譜、平滑化
# 0313 *319 修正去雜訊後幅度縮小問題
# 0412-A 正規化再平均       # 子謙資料
# 0412-B 平均在正規化(OK) * # 子謙資料
# 0425 更改為新實驗 trigger
# 0519 合併多筆資料
# 0524 MI FB 混淆修正
# 0606 max=5e
# 0608 data_point_loss 掉點監測
# 0801 取休息的平均，取資料動作開始前 3 - 1 秒當作休息

# 1024 SMOOTH 曲線更改2
# 1114 增加抗雜訊程式1 (2.4.2.2. 抑制運動偽影和肌肉偽影) (要設定MAXEEG = 0)
# https://www.sciencedirect.com/science/article/pii/S2213158218303024#bb0185
# 1122 用在 trials 足夠的，否則會去掉太多trials導致雜訊更明顯

##### 此版本針對 皓嚴2024_01_23_1456 #####

import pathlib
from os.path import exists
if __name__ == '__main__':
    
    TIME_LEN_REST = 4000 # max 4000 +1 
    TIME_LEN_ACTION = 8000 # max 5000
    TIME_LEN_ALL = TIME_LEN_REST + TIME_LEN_ACTION

    # 取資料動作開始前 3 - 1 秒當作休息
    rest_on  = 3000
    rest_off = 1000

    DECODE_SWITCH = 0
    SMOOTH = 200
    LIMIT  = 100  # 畫圖上下限 0為浮動
    FRERUENCY_LIST = ['A1','B1'] # 'A1','B1', 
    SIDE_SET_LIST  = ['right','left'] # right left

    SUBJECT_NAMES_NONE = ['柏勛','建誌','半糖']
    SUBJECT_NAMES_NF = ['又諄','文豪','柏瑋','皓嚴','羅喨']
    SUBJECT_NAMES_FB = ['嘉浚','浩軒','泓誌','松育','聖佶']
    # SUBJECT_NAMES = ['又諄','羅喨','松育','浩軒']

    # SUBJECT_NAMES = ['又諄']
    # DATE_LIST = ['2023_11_27_1623']

    SUBJECT_NAMES = ['皓嚴']
    DATE_LIST = ['2024_01_23_1456']

    # NEW trigger    
    ACT_list = ['AO']
    # ACT_list = ['AO', 'FB', 'MIFB']
    TRIGGER_SET = 'new04'     
    # FILE_PATH   = './Raw_data_New/Raw_data_new_FB/' #子謙_FILE_PATH\cut_data\B\re2\left.npy
    DAY      = 'day1'

    MAX_EEG     = 0  #2.5e-5  0
    # FB_shift    = 2520

    # cut bad 2.5e-5 # 超過則為max
    # FB 不用 cutbad

# %% #  刪除 txt 
    # clear_error_txt()

# %% #  
    for AO_MI in ACT_list:
        for name in SUBJECT_NAMES:
            # if name in SUBJECT_NAMES_NONE: TEAM = '備案'
            # if name in SUBJECT_NAMES_NF: 
            TEAM = '純想像組'
            # if name in SUBJECT_NAMES_FB: TEAM = '震動反饋組'

            # FILE_PATH   = f'./Raw_data_All/{name}/{AO_MI}_{DAY}/' 
            # SAVE_PATH_0 = f'./ERD_plot/ERD_1004/{name}/{AO_MI}_{DAY}' 

            FILE_PATH   = f'./1_AO_data/{TEAM}/{name}/'            

            check_decoder()         

            for side in range(len(SIDE_SET_LIST)):
                SIDE_SET = SIDE_SET_LIST[side]
                print(f"-----------{SIDE_SET} hand ---------")

                for band in range(len(FRERUENCY_LIST)): # len(FRERUENCY_LIST)
                    freq_band = FRERUENCY_LIST[band]
                    if      freq_band == 'A1':  SAVE_PATH_0 = f'./Step2/2A_ERD_plot_30trials/{TEAM}/{name}/'
                    elif    freq_band == 'B1':  SAVE_PATH_0 = f'./Step2/2B_ERD_plot_30trials/{TEAM}/{name}/'

                    if   freq_band == 'A1': hp_freq, lp_freq = 8  ,14 # alpha1    
                    elif freq_band == 'B1': hp_freq, lp_freq = 16 ,24 # beta1
                    elif freq_band == 'theta': hp_freq, lp_freq = 4 ,7 
                    elif freq_band == 'gamma': hp_freq, lp_freq = 30 ,80 
                    else :hp_freq, lp_freq = 1 ,99
    #######################

                    npy_data_path_All = get_root(FILE_PATH, '.npy')  # 找出每個腦波npy檔  

                    for npy_data_path in npy_data_path_All:
                        path1, save_path = save_path_set(AO_MI, npy_data_path, hp_freq, lp_freq)

                        plt.style.use("seaborn")
                        # print(os.path.split(os.path.dirname(npy_data_path))[-2][-15:])
                        now_date = os.path.split(os.path.dirname(npy_data_path))[-2][-15:]

                        if now_date in DATE_LIST:
                        ###
                            # # 確認有哪些 alpha band 已經畫好的
                            # file_check_R = f'{SAVE_PATH_0}/{now_date}/ERDs_right_T48_Hz814.png'
                            # file_check_L = f'{SAVE_PATH_0}/{now_date}/ERDs_left_T48_Hz814.png'
                            # if os.path.exists(file_check_R) and os.path.exists(file_check_L):
                            #     print(f"A已完成"); continue
                            
                            # # 確認有哪些 beta band 已經畫好的
                            # file_check_BR = f'{SAVE_PATH_0}/{now_date}/ERDs_right_T48_Hz1624.png'
                            # file_check_BL = f'{SAVE_PATH_0}/{now_date}/ERDs_left_T48_Hz1624.png'
                            # if os.path.exists(file_check_BR) and os.path.exists(file_check_BL):
                            #     print(f"B已完成"); continue
                        ###                        
                            new_data_concat = np.zeros((0, 8, TIME_LEN_ALL))
                            new_data, boolean_mask = Get_Data().get_data(AO_MI, npy_data_path)

                            if side == 0: clear_file(f'{path1}/antibias_list.txt',1)
                            save_txt(f'{path1}/antibias_list.txt', boolean_mask)
                            new_data_concat = np.concatenate([new_data_concat, new_data], axis=0)
                            # print(np.shape(new_data_concat))
            
                            plot_data = plotERD().get_ERD( new_data_concat, hp_freq, lp_freq)
                            plot_data = plotERD().Plot_ERDS(AO_MI, plot_data, save_path)             
                            np.save(f'{path1}/ERDS_data_{SIDE_SET}', plot_data)

# print ("time:", time.process_time())y
