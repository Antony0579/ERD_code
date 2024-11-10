# 修改訊號品質計分方式

# 解碼收到的腦波資料 EEG.txt，並濾波
# 畫 FFT 
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from scipy.signal import butter, lfilter
from colorama import Fore, Back, Style

def get_latest_date(exp_path):
    # exp_path : path of exp directory. e.g. './exp'
    # return : latest date in exp_path. e.g. '2022_01_01_0000'

    exp_date = [item for item in os.listdir(exp_path) if os.path.isdir('{}/{}'.format(exp_path, item))]
    exp_date_int = [int("".join(item.split("_"))) for item in exp_date]
    data_date = exp_date[np.argmax(exp_date_int)]

    return data_date

#########################################################
show_fft = 1
show_eeg = 0    #   Decode時才畫
exp_dir = './exp'
# data_date = '2022_10_26_1557'
data_date = get_latest_date(exp_dir) # 在exp資料夾中找出最後一筆資料的時間  
######################################################### 

def progressBar(title, temp, total):
    print('\r' + '['+title+']:[%s%s] %s(%.2f%%)' % ('█' * int(temp/total*20), ' ' * (20-int(temp/total*20)), str(temp)+'/'+str(total), float(temp/total*100)), end='')
    if temp == total:
        print ('')

def show_leadOff(eeg_data):
    # 每個channel中的leadOff只要出現一個1就視為斷線
    lead_off = [str(int(x)) for x in eeg_data[9]]

    # 補0，leadOff長度有可能不是8
    # e.g. 100, 1110, 0, ...
    for i, x in enumerate(lead_off):
        if len(x) < 8:
            num_padding = 8 - len(x)
            lead_off[i] = '0'*num_padding + lead_off[i]

    # string turn into list
    # e.g. 1100010 -> [1, 1, 0, 0, 0, 1, 0]
    lead_off = np.array([list(x) for x in lead_off], dtype=float).T

    # if there more than 1 consider as disconnected
    connect_state = np.sum(lead_off, axis=1)
    color = [91 if x else 92 for x in connect_state]
    # 91 : red, 92 green

    # print layout
    # ╭─ EEG Electrode Connection ─╮
    # │   F3                  F4   │
    # │   C3        Cz        C4   │
    # │   P3        Pz        P4   │
    # ╰─ connected ─ disconnected ─╯    
    # https://en.wikipedia.org/wiki/Box-drawing_character
    # u'\u2502' = "│" , u'\u256D' = "╭", u'\u2500' = "─", u'\u256E' = "╮", u'\u2570' = "╰", u'\u256F' = "	╯"
    print('\n')
    print(u'\u256D' + u'\u2500' + ' EEG Electrode Connection ' + u'\u2500' +  u'\u256E')
    print(u'\u2502' + f"\033[{color[0]}m{'F3':^9}\033[0m {'':^9} \033[{color[1]}m{'F4':^8}\033[0m" + u'\u2502')
    print(u'\u2502' + f"\033[{color[2]}m{'C3':^9}\033[0m \033[{color[3]}m{'Cz':^9}\033[0m \033[{color[4]}m{'C4':^8}\033[0m" + u'\u2502') 
    print(u'\u2502' + f"\033[{color[5]}m{'P3':^9}\033[0m \033[{color[6]}m{'Pz':^9}\033[0m \033[{color[7]}m{'P4':^8}\033[0m" + u'\u2502')
    print(u'\u2570' + u'\u2500' + "\033[92m connected \033[0m" + u'\u2500' + "\033[91m disconnected \033[0m" + u'\u2500' + u'\u256F')

class Decoder(): # 已變更
    
    def __init__(self):
        self.comp = 8388607  # complement (7FFFFF)??why
        # self.eeg_date = eeg_date
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

        #### Filter ####
        F = Filter()
        eeg_filtered = F.filter(eeg_raw)
        ####        ####
  
        return eeg_filtered

    def decode_to_txt__polt_eeg(self, eeg_txt_path, plotq, return_data, eeg_date):
        # decode, filtering EEG.txt and then write into 1.txt
        # if return_data == True : return filtered eeg data with shape = (n, 10) 
        print("Process data \n\n{}\n".format(eeg_txt_path))
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

        print('Cost: {0:.1f} s'.format(int((time.time() - start_t) * 10) * 0.1))
        print("Shape of decoded EEG data: ({}, {})".format(np.size(eeg_filtered, 0), np.size(eeg_filtered, 1)))
        print('-'*40)

        if plotq == 1:
            self.plot_eeg(eeg_filtered, eeg_date)

        if return_data:
            return eeg_filtered   

    def read_decoded(self, file_path):
        # read decoded eeg data file. e.g. 1.txt
        # input 
        #   file_path : path of eeg data file. e.g. ./exp/2022_10_20_XXXX/1/1.txt
        # output 
        #   data : eeg data with shape = (n, 10)
        print("Process data \n\n{}\n".format(eeg_txt_path))
        data = []
        f = open(file_path)
        for line in f.readlines():
            content = line.split()  # 指定空格作為分隔符對line進行切片
            content = list(map(float, content)) # string轉成float
            data.append(content)

        return data
    
    def plot_eeg(self, eeg_data, eeg_date):
        # eeg_data : eeg data with shape = (n, 10) which
        #            8 is the number of channel,
        #            n is the number of sample points
        # num_channel : number of channels 
        n = len(eeg_data) # number of samples 
        num_channel = 8
        label_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'FrameID', 'Trigger']

        fig = plt.subplots(4, 3, figsize=(7, 5))

        # check the continuity of frameID
        frame_id = np.zeros([n - 1])
        for i in range(n - 1):
            frame_id[i] = eeg_data[i + 1][8] - eeg_data[i][8]
            if frame_id[i] == -255:
                frame_id[i] = 1
        
        # trigger
        s = 0
        triggers = []
        for i in range(n):
            if s > 0:
                s -= 1
                continue
            else:
                if eeg_data[i][9] != 248:
                    triggers.append([i, eeg_data[i][9]])
                    s = 5

        eeg_data = np.array(eeg_data).T # shape (n, 10) -> (10, n)

        x_axis = np.arange(1, n + 1)

        # set x_tick and xticklabels
        sec = n // 1000
        if sec < 10:
            x_tick = np.arange(1, n, 1000)
            xticklabels = [str(int(time/1000)) for time in range(0, n, 1000)] # name of x_ticks (time)
        elif (10 <= sec) and (sec < 100):
            x_tick = np.arange(1, n, 10000)
            xticklabels = [str(int(time/1000)) for time in range(0, n, 10000)] # name of x_ticks (time)            
        elif (100 <= sec) and (sec < 500):
            x_tick = np.arange(1, n, 100000)
            xticklabels = [str(int(time/1000)) for time in range(0, n, 100000)] # name of x_ticks (time)
        elif (500 <= sec) and (sec < 1000):
            x_tick = np.arange(1, n, 200000)
            xticklabels = [str(int(time/1000)) for time in range(0, n, 200000)] # name of x_ticks (time)
                
        x_tick = np.append(x_tick, n)
        xticklabels.append(str(int(n/1000)))

        color = '#17395C'
        linewidth = 0.85                              
        plt.style.use('ggplot')
        for i in range(num_channel + 2):
            row = i // 3
            column = i % 3

            if i < 8: # ch0 - ch7
                y = eeg_data[i]
                plt.subplot2grid((4, 3), (row, column))
                plt.plot(x_axis, y, label=label_list[i], c=color, lw=linewidth)
                plt.ylim([-0.002, 0.002])
                
            elif i == 8: # frameID
                y = frame_id
                x_axis = np.arange(1, n)
                plt.subplot2grid((4, 3), (row, column))
                plt.plot(x_axis, y, label=label_list[i], c=color, lw=linewidth)                
                plt.ylim([0, 2])
                plt.ylim([-0.002, 0.002])

            elif i == 9: # trigger
                y = eeg_data[i]
                trigger = [[i, 1] if x != 241 else [0, 0] for i, x in enumerate(y)]
                x_axis = np.arange(1, n + 1)
                plt.subplot2grid((4, 3), (3, 0), colspan = 3)
                plt.plot(x_axis, [x[1] for x in trigger], label=label_list[i], c=color, lw=linewidth)                
                for trigger in triggers:
                    plt.text(trigger[0], 1.5, int(trigger[1]), c=color, ha='center', va='center')
                plt.ylim([-0.2, 2])
            plt.xticks(x_tick, xticklabels)
            plt.legend(labelcolor='darkorange')
        plt.suptitle("EEG (sec)")                        
        plt.tight_layout()

        # data_date = get_latest_date('./exp')
        png_path = './exp/{date}/EEG.png'.format(date=eeg_date)
        plt.savefig(png_path, dpi=300)  
        # plt.show()

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
        progressBar("解碼進度", j, n-1)

        output[:, :8] = output[:, :8] * 2.5 / (2**23 - 1)  # 將收到的data換成實際電壓
        return output  # 八通道腦波資料(每個通道以十進制存), shape = (n, 10)                        

class Filter():
    def __init__(self):
        self.fs = 1000
        self.hp_freq = 4
        self.lp_freq = 40
    
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

class FFT():            
    def fft(self, eeg_data, Fs=1000.0, Ts=1.0/1000.0):
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

    def normal_distribution(self, x,mu,sig):
        sig = sig*sig
        ans = 1 / (sig* (2*math.pi)**0.5) * math.exp(- (x-mu)**2 / (2*sig**2))
        return ans

    def data_check_2(self, eeg_data, save_path):
        # label_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
        num_channel = 8
        check_ch = check_p =  0

        path_check = save_path + '/data_check.txt'
        check_write = open(path_check,"w")

        print("channel  |   E_score")
        check_write.write("channel  |   E_score"+ "\n")

        for channel in range(num_channel):
            # y = eeg_data[channel,:]
            y = eeg_data[channel]
            t, freq, Y = self.fft(y)

            E_good_data = E_con_data = E_bad_data = ExF = ExF_02 = 0
            eg = eb_8 = Y_num = F_standard = 0        
            max_i = 0
            Y = abs(Y)        

            # 計算8-12Hz的訊號能量占比
            for i in range(len(freq)):         
                if (freq[i] > 8 and freq[i] < 12):  
                    # print(Y[i])              
                    E_good_data = E_good_data + Y[i]**2
                    ExF = ExF + freq[i] * (Y[i]**2)
                    # if (Y[max_i]**2 < Y[i]**2):
                    #     max_i = i
                    eg = eg+1
                else:
                    E_bad_data = E_bad_data + Y[i]**2 
                    if freq[i] < 8:
                        eb_8 = eb_8+1 
            Er_out = E_good_data/(E_good_data + E_bad_data) 

            # 計算均值頻率(能量=個數)
            F_avg = ExF / E_good_data
            # 計算8-12Hz的訊號標準差
            for i in range(len(freq)):         
                if (freq[i] > 8 and freq[i] < 12): 
                    ExF_02 = ExF_02 + ((freq[i] - F_avg)**2) * (Y[i]**2)
                    Y_num = Y_num + Y[i]**2
            F_standard = (ExF_02 / Y_num) * 0.5
            Er_in = 1 - F_standard           
            
            # # 計算跟10Hz的差值
            # Er_shift = 10 / (10 + abs(10-F_avg))
            Emu = 6.4
            E_score =  Er_in * Er_out**2 * 100 *Emu
            # print(Er_in, Er_out, E_score)

            if E_score > 60:
                print(Fore.GREEN + "  ",channel+1, "    |   %.2f "%E_score+"%")
                check_ch+=1
            elif E_score > 30:
                print(Fore.YELLOW+ "  ",channel+1, "    |   %.2f "%E_score+"%")
                check_p+=1
            else:
                print(Fore.RED   + "  ",channel+1, "    |   %.2f "%E_score+"%")

            check_write.write("    " + str(channel+1) + "    |   %.2f "%E_score + "%" + "\n")
        # print(Fore.WHITE)

        if check_ch == 8:
            print(Back.GREEN + "  Data Check Successful！     ")
        elif (check_p + check_ch) == 8:
            print(Back.YELLOW + "  Data Check Pass  ")
        print(Fore.WHITE + Back.BLACK)
            

    def plot_fft(self, eeg_data, fft_date):
        # eeg_data : eeg data with shape = (8, n) which
        #            8 is the number of channel,
        #            n is the number of sample points
        # num_channel : number of channels 
        num_channel=8

        label_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']

        fig, ax = plt.subplots(num_channel, 2, figsize=(15,num_channel*2))

        for channel in range(num_channel):
            # y = eeg_data[channel,:]
            y = eeg_data[channel]
            t, freq, Y = self.fft(y)

            # plot raw data
            ax[channel, 0].plot(t, y, linewidth=0.5, color='steelblue')
            ax[channel, 0].set_ylabel(label_list[channel], fontsize=14, rotation=0) # 'Amplitude'
            ax[channel, 0].yaxis.set_label_coords(-.2, .4)
            # plot spectrum        
            ax[channel, 1].axis(xmin=0,xmax=30)   # 頻譜上下限
            ax[channel, 1].plot(freq, abs(Y), linewidth=0.5, color='steelblue')        
            ax[channel, 1].set_ylabel(label_list[channel], fontsize=14, rotation=0) # '|Y(freq)|'
            ax[channel, 1].yaxis.set_label_coords(1.1, .4)

            # remove x label except the bottom plot
            if (channel + 1 != num_channel):
                ax[channel, 0].axes.xaxis.set_ticklabels([])
                ax[channel, 1].axes.xaxis.set_ticklabels([])
            else:
                ax[channel, 0].set_xlabel('Time')
                ax[channel, 1].set_xlabel('Freq (Hz)')
            
            # mark the range of alpha band (8 - 12 Hz)
            rect = patches.Rectangle((8,0),
                            width=4,
                            height=np.max(abs(Y)),
                            facecolor = 'orange',
                            fill = True,
                            alpha = 0.2)             
            ax[channel, 1].add_patch(rect)

            # mark the range of alpha band (8 - 12 Hz)
            # rect = patches.Rectangle((0,0),
            #                 width=4,
            #                 height=np.max(abs(Y)),
            #                 facecolor = 'red',
            #                 fill = True,
            #                 alpha = 0.2)             
            # ax[channel, 1].add_patch(rect)

        ax[0, 0].set_title('Raw EEG', fontsize=14)
        ax[0, 1].set_title('FFT', fontsize=14)

        
        # data_date = get_latest_date('./exp')
        png_path1 = './exp/{date}/FFT.png'.format(date=fft_date)
        plt.savefig(png_path1, dpi=300)
        
    

if __name__ == '__main__':
    
    # data_date = '2022_11_16_1537'
    # data_date = '2022_11_09_1721'

    date_path = '{exp}/{date}'.format(exp=exp_dir, date=data_date)
    eeg_txt_path = date_path + '/1/EEG.txt'
    decoded_file_path = date_path + '/1/1.txt'
    # file_exists = os.path.exists(decoded_file_path)
    file_exists = 0
    if file_exists:
        # 讀取已解碼後的EEG資料，1.txt
        decode = np.array(Decoder().read_decoded(decoded_file_path)) 
    else:
        # 解碼16進制EEG.txt資料至10進制1.txt資料
        # EEG需要重新decode
        decode = Decoder().decode_to_txt__polt_eeg(eeg_txt_path = eeg_txt_path, plotq = show_eeg, return_data = 1, eeg_date = data_date) 
        
    start = 5000  # 捨棄前一秒的資料以及最後一秒的資料 
    end = decode.shape[0]-1000
    decode = decode[start:end].T

    show_leadOff(decode)
    FFT().data_check_2(eeg_data = decode, save_path = date_path)

    if show_fft:
        print("Plot FFT")
        FFT().plot_fft(decode, data_date)        
    plt.show()