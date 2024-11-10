# %% # ===========================[import+]============================ # 
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from PyQt5 import QtWidgets
from PyQt5.QtGui import *
# from Ui_NewGUI import Ui_MainWindow
from datetime import datetime

import shutil
import sys
import multiprocessing
import serial
import time
import numpy as np
import os

# %% # ===========================[import-]============================ # 

# 2021/02/04 EEG_button
# 2023 01/14 新增 trigger plot
#  https://stackoverflow.com/a/6981055/6622587
# 2023 01/13    realtime plot, made by chenfu 
#  0329 整理code  

MAIN_PATH  = '.'
CHECK_FILE = MAIN_PATH + '/check_save.txt'
SAVE_FILE  = MAIN_PATH + '/exp'
FILE_NAME  = 'EEG.txt'

EEG_COM_PORT = 20
TRIGGER_SET  = '負邏輯'

class My_function():
    def get_latest_date(exp_path):
        # exp_path : path of exp directory. e.g. './exp'
        # return : latest date in exp_path. e.g. '2022_01_01_0000'
        exp_date = [item for item in os.listdir(exp_path) if os.path.isdir('{}/{}'.format(exp_path, item))]
        exp_date_int = [int("".join(item.split("_"))) for item in exp_date]
        data_date = exp_date[np.argmax(exp_date_int)]

        return data_date
        
    def progress_bar(title, temp, total):
        if temp >= total:
            temp = total
        # print('\r' + '['+title+']:[%s%s] %s(%.2f%%)' % ('█' * int(temp/total*20), ' ' * (20-int(temp/total*20)), str(temp)+'/'+str(total), float(temp/total*100)), end='')
        print('\r{}: |{}{}| {}/{} [{:.2f}%]'.format(title, 
            '█' * int(temp/total*25), ' ' * (25-int(temp/total*25)), 
            str(temp), str(total), 
            float(temp/total*100)),   
            end='')
    
    def open_read_write(open_file, do, txt='0'): # 預設值參數 txt='0'
        if do == "w" or do == "a":
            with open(open_file, do) as f:
                f.write(txt)
                f.close()
        elif do == "r":
            with open(open_file, do) as f:
                file_read = f.read()
                f.close()
            return file_read

# %% # ===========================[RealtimePlot+]============================ # 
# from eeg_decoder import Decoder, Filter
import matplotlib.animation as animation
from multiprocessing import Queue
from matplotlib.ticker import StrMethodFormatter

class RealtimePlot():
    def __init__(self, fileDir, queue, max_t=2000):
        # self.ax = ax
        self.max_t = max_t # 視窗顯示秒數長度
        self.tdata = np.empty(0) 
        self.ydata = np.empty((10, 1))
        self.t_total = 0    
        self.fileDir = fileDir   
        self.raw_total = '' # for test
        self.cut_point = 1000 # remove filtered data points before cut_point
        self.decoder = Decoder()
        self.queue = queue
        self.eeg_total = np.empty((1, 10))
        self.filter = Filter()

        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax_trig) = plt.subplots(9, 1, figsize=(10, 6))
        self.ax = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax_trig)

        # plot parameter
        self.color = '#17395C' #17395C # steelblue
        self.linewidth = 0.8
        self.channel_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'Trigger']
 
        ani = animation.FuncAnimation(fig=fig, 
                                      init_func=self.init_func,
                                      func=self.update, 
                                      interval=30, 
                                      frames = self.data_gen, 
                                      blit=True,
                                      repeat=False,
                                      save_count=200)
        plt.show()

    def init_func(self):
        # 要分開寫，用for產生會有問題
        line1, = self.ax[0].plot([], [], c=self.color, lw=self.linewidth)
        line2, = self.ax[1].plot([], [], c=self.color, lw=self.linewidth)
        line3, = self.ax[2].plot([], [], c=self.color, lw=self.linewidth)
        line4, = self.ax[3].plot([], [], c=self.color, lw=self.linewidth)
        line5, = self.ax[4].plot([], [], c=self.color, lw=self.linewidth)
        line6, = self.ax[5].plot([], [], c=self.color, lw=self.linewidth)
        line7, = self.ax[6].plot([], [], c=self.color, lw=self.linewidth)
        line8, = self.ax[7].plot([], [], c=self.color, lw=self.linewidth)
        trigger, = self.ax[8].plot([], [], c=self.color, lw=self.linewidth)
        self.line = [line1, line2, line3, line4, line5, line6, line7, line8, trigger]
        xticks = [x for x in range(0, self.max_t+1, 1000)]
        xticklabels = [str(int(time/1000)) for time in range(0,  self.max_t+1, 1000)]        
        for i in range(9):
            self.ax[i].set_xticks(xticks)            
            self.ax[i].set_xlim(0, self.max_t)
            # remove x label except the bottom plot
            if i == 8: # trigger
            # if i == 7: # ax8
                self.ax[i].set_xticklabels(xticklabels)
            else: # ax1 ~ ax7
                self.ax[i].axes.xaxis.set_ticklabels([])
            self.ax[i].set_ylabel(self.channel_list[i], fontsize=14, rotation=0) # channel name    
            self.ax[i].yaxis.set_label_coords(-0.1, .35) # set the label position  

        return self.line

    def update(self, y):     
        # input 
        #   y : raw data from EEG.txt

        self.raw_total += y
        eeg_raw = np.array(self.decoder.decode(self.raw_total)) # shape = (n, 10)
        # print(eeg_raw.shape)
        
        # rest the parameter
        if len(eeg_raw)-self.cut_point-1 >= self.max_t: # 長度超過顯示秒數就重畫x座標軸
            self.t_total += len(eeg_raw[0])
            self.ydata =np.empty((10, 1))
            
            self.raw_total = ''
            xticklabels = [str(int(time/1000)-1) for time in range(self.t_total, self.t_total + self.max_t+1, 1000)] # name of x_ticks (time)
            self.ax[7].set_xticklabels(xticklabels)
            self.ax[7].figure.canvas.draw() # redraw everything, would made animation slow        
        
        # 捨棄前一秒及最後一筆濾波後的資料，因為有問題
        self.tdata = np.arange(len(eeg_raw)-self.cut_point-1)
        self.ydata = eeg_raw[self.cut_point:-1].T # shape = (n, 10)
        for i in range(8):
            self.line[i].set_data(self.tdata, self.ydata[i])
            self.ax[i].set_ylim([-4e-5, 4e-5]) # 腦波電壓振幅範圍
            # self.ax[i].relim() 
            # self.ax[i].autoscale_view()
        #　trigger 設定
        if TRIGGER_SET == '正邏輯':
            self.line[8].set_data(self.tdata, self.ydata[9])
            self.ax[8].set_ylim([0, 8]) # 振幅範圍
        elif TRIGGER_SET == '負邏輯':
            self.line[8].set_data(self.tdata, 255-self.ydata[9])
            self.ax[8].set_ylim([5, 7]) # 振幅範圍

        return self.line
    
    def data_gen(self):        
        # retrun raw data each interval time        
        while os.path.exists('{}/1/{}'.format(self.fileDir, 'EEG.txt')):
            try:                               
                raw = self.queue.get()     
                yield raw 
            except Exception:
                pass        
 
class Decoder():
    def __init__(self):
        self.comp = 8388607  # complement (7FFFFF)??why

    def decode(self, raw, show_progress = False):
        """
        decode EEG.txt and filtering

        Parameters
        ----------------------------------------
        `raw` : raw HEX eeg data, e.g. content of './exp/2022_10_20_XXXX/1/EEG.txt'
        `show_progress` : print progress bar if true, not print if false


        Return
        ----------------------------------------
        `eeg_filtered` : filtered eeg data with shape = (n, 10)   


        Examples
        ----------------------------------------
        >>> decoder = Decoder()
        >>> eeg_filtered = decoder.decode(raw, show_progress = True)
        """
        
        eeg_raw = self.get_BCI(raw, show_progress)
        F = Filter()
        eeg_filtered = F.filter(eeg_raw)
  
        return eeg_filtered

    def decode_to_txt(self, eeg_txt_path, return_data = False, not_triggered = 0):
        """
        decode, filtering EEG.txt and then write into 1.txt

        Parameters
        ----------------------------------------
        `eeg_txt_path` : raw HEX eeg data, e.g. content of './exp/2022_10_20_XXXX/1/EEG.txt'
        `return_data` : True : return filtered eeg data with shape = (n, 10)
                        False : just write txt and plot eeg
        `not_triggered` : trigger value when not trigger, e.g. 0 or 255


        Return
        ----------------------------------------
        `eeg_filtered` : filtered eeg data with shape = (n, 10)   


        Examples
        ----------------------------------------
        >>> decoder = Decoder()
        >>> eeg_filtered = decoder.decode_to_txt(eeg_txt_path = eeg_txt_path, return_data=True, not_triggered = 0) # 解碼16進制EEG.txt資料至10進制1.txt資料
        """


        print("Process file >>", end='')
        print('\033[32m {} \033[0m\n'.format(eeg_txt_path))

        f = open(eeg_txt_path)
        raw = f.read()
        f.close()

        start_t = time.time()
        eeg_filtered = self.decode(raw, show_progress = True)

        print("")
        f = open('/'.join(eeg_txt_path.split('/')[0:-1]) + '/1.txt', 'w')  # 打開1.txt檔案
        for i in range(np.size(eeg_filtered, 0)):
            for j in range(np.size(eeg_filtered, 1)):
                f.write(str(eeg_filtered[i][j]))
                f.write('\t')
            f.write('\n')
            if i % 1000 == 0 or i + 1000 >= np.size(eeg_filtered, 0):
                My_function.progress_bar("Saving  ", i, np.size(eeg_filtered, 0)-1)            
        f.close() 

        print('\nCost: {0:.1f} s'.format(int((time.time() - start_t) * 10) * 0.1))
        print("Shape of decoded EEG data: ({}, {})".format(np.size(eeg_filtered, 0), np.size(eeg_filtered, 1)))
        print("Ploting...")
        print('-'*40)
        
        self.plot_eeg(eeg_filtered, png_path='/'.join(eeg_txt_path.split('/')[0:-2]) + '/EEG.png', not_triggered = not_triggered)

        if return_data:
            return eeg_filtered   

    def read_decoded(self, file_path):
        """
        read decoded eeg data file. e.g. 1.txt

        Parameters
        ----------------------------------------
        `file_path` : path of eeg data file. e.g. ./exp/2022_10_20_XXXX/1/1.txt


        Return
        ----------------------------------------
        `data` : eeg data with shape = (n, 10)


        Examples
        ----------------------------------------
        >>> decoder = Decoder()
        >>> eeg_data = decoder.read_decoded(file_path) # 讀取解碼後的EEG資料，1.txt
        """

        print("Read file >>", end='')
        print('\033[32m {} \033[0m'.format(file_path))
        print("")

        data = []
        f = open(file_path)
        num_lines = sum([1 for line in open(file_path)])
        for i, line in enumerate(f.readlines()):
            content = line.split()  # 指定空格作為分隔符對line進行切片
            content = list(map(float, content)) # string轉成float
            data.append(content)
            if i % 1000 == 0 or i + 1000 >= num_lines-1:
                My_function.progress_bar("Reading", i, num_lines-1)
        return np.array(data)

    def plot_eeg(self, eeg_data, png_path, not_triggered = 0):
        """
        remove first 1 second and and last 1 second

        Parameters
        ----------------------------------------
        `eeg_data` : eeg data with shape = (n, 10) which
                     8 is the number of channel,
                     n is the number of sample points
        
        `png_path` : path for save figure
        
        `not_triggered` : trigger value when not triggered. 
                        e.g. 0 when not trigger, or 255 when not trigger

        Examples
        ----------------------------------------
        >>> decoder = Decoder()
        >>> decoder.plot_eeg(eeg_filtered, png_path='/'.join(eeg_txt_path.split('/')[0:-2]) + '/EEG.png', not_triggered = 0)
        """
        
        eeg_data = eeg_data[1000:-1000] # remove first 1 second and last 1 second

        n = len(eeg_data) # number of samples 
        num_channel = 8
        label_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'FrameID', 'Trigger']

        fig = plt.subplots(4, 3, figsize=(14, 6))

        # check the continuity of frameID
        frame_id = np.zeros([n - 1])
        for i in range(n - 1):
            frame_id[i] = eeg_data[i + 1][8] - eeg_data[i][8]
            if frame_id[i] == -255:
                frame_id[i] = 1
        
        # trigger
        # 預期會收到連續五個一樣的trigger，找出第一個trigger的位置和值

        s = 0
        triggers = []
        for i in range(n):
            if s > 0:
                s -= 1
                continue
            else:
                
                if eeg_data[i][9] != not_triggered:
                    triggers.append([i, eeg_data[i][9]])
                    s = 5

        eeg_data = np.array(eeg_data).T # shape (n, 10) -> (10, n)

        # set x_tick and xticklabels, 不同秒數設定不同的時間跨度
        sec = n // 1000
        if sec < 10: point_gap = 1000
        elif (10 <= sec) and (sec < 50): point_gap = 5000 
        elif (50 <= sec) and (sec < 100): point_gap = 10000
        elif (100 <= sec) and (sec < 200): point_gap = 50000 
        elif (200 <= sec) and (sec < 500): point_gap = 100000     
        elif (500 <= sec) and (sec < 1000): point_gap = 200000

        x_tick = np.arange(1, n, point_gap)
        x_tick = np.append(x_tick, n) # 最後一筆資料秒數的位置
        xticklabels = [str(int(time/1000)+1) for time in range(0, n, point_gap)] # name of x_ticks (time)                                
        xticklabels[0] = 1 # 從第一秒開始
        xticklabels.append(str(round(n/1000, 1)+1)) # 顯示最後一筆資料的秒數
        
        # plot
        color = '#17395C'
        linewidth = 0.5
        fontsize = 8                       
        plt.style.use('ggplot')

        x_axis = np.arange(1, n + 1)
        for i in range(num_channel + 2):
            row = i // 3
            column = i % 3
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.5f}')) # 4 decimal places
            if i < 8: # ch0 - ch7
                y = eeg_data[i]                
                plt.subplot2grid((4, 3), (row, column))
                plt.plot(x_axis, y, label=label_list[i], c=color, lw=linewidth)
                
            elif i == 8: # frameID
                y = frame_id
                x_axis = np.arange(1, n)
                plt.subplot2grid((4, 3), (row, column))
                plt.plot(x_axis, y, label=label_list[i], c=color, lw=linewidth)                
                plt.ylim([-0.2, 2.2])
                plt.yticks([])

            elif i == 9: # trigger
                y = eeg_data[i]

                trigger = [[i, 1] if x != not_triggered else [0, 0] for i, x in enumerate(y)] # 找出所有trigger，高度顯示設定1
                x_axis = np.arange(1, n + 1)
                plt.subplot2grid((4, 3), (3, 0), colspan = 3)
                plt.plot(x_axis, [x[1] for x in trigger], label=label_list[i], c=color, lw=1)
                
                for trigger in triggers: # 顯示trigger的值
                    plt.text(trigger[0], 1.5, int(trigger[1]), c=color, ha='center', va='center', fontsize=fontsize)
                plt.yticks([])
                plt.ylim([-0.2, 2])
                plt.margins(x = 0.01) 

            plt.xticks(x_tick, xticklabels, fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.legend(labelcolor='darkorange', fontsize=fontsize, loc='upper right')
        plt.suptitle("EEG (sec)")                        
        plt.tight_layout()
        plt.savefig(png_path, dpi=500)  

    def HEX_to_DEC(self, hexdata):  # 16轉10進制
        length = len(hexdata)
        num = 0
        for i in range(0, length, 2):
            data_short = hexdata[i] + hexdata[i + 1]

            data_dec = int(data_short, 16)
            num = num + data_dec * (256 ** (((length - i) / 2) - 1))
        return num

    def get_BCI(self, raw_data, show_progress = False):
        """
        解碼raw eeg data(16進制)成10進制

        Parameters
        ----------------------------------------
        `raw_data` : raw eeg data, e.g. EEG.txt 
  

        Return
        ----------------------------------------
        `output` : shape = (n, 10), 8ch + framid + trigger


        Examples
        ----------------------------------------        
        >>> with open('EEG.txt', 'r') as f:
        >>>     raw = f.read()
        >>> decoder = Decoder()
        >>> eeg_raw = decoder.get_BCI(raw, show_progress)
        >>> eeg_filtered = F.filter(eeg_raw)        
        """

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
                if show_progress:
                    if j % 1000 == 0 or j + 1000 >= n-1:
                        My_function.progress_bar("Decoding", j, n-1)
            else:
                i += 2  # 若沒有讀到頭尾封包，往後找1byte            
            
        output[:, :8] = output[:, :8] * 2.5 / (2**23 - 1)  # 將收到的data換成實際電壓
        return output  # 八通道腦波資料(每個通道以十進制存), shape = (n, 10)                        

from scipy.signal import butter, lfilter
class Filter():
    def __init__(self):
        self.fs = 1000
        self.hp_freq = 4
        self.lp_freq = 40
        
    def filter(self, eeg_raw):
        """
        濾波 : 60Hz市電、120諧波，4 - 40 Hz 帶通濾波取出腦波頻率範圍

        Parameters
        ----------------------------------------
        `eeg_raw` : shape = (n, 10)，8通道+frameID+Trigger
  

        Return
        ----------------------------------------
        `eeg_raw` : shape = (n, 10) 


        Examples
        ----------------------------------------
        >>> F = Filter()
        >>> eeg_filtered = F.filter(eeg_raw)        
        """

        for i in range(8): # ch1 ~ ch8
            # 60 Hz notch
            eeg_raw[:, i] = self.butter_bandstop_filter(eeg_raw[:, i], 55, 65, self.fs)            
            # 120 Hz notch, 60Hz 諧波
            eeg_raw[:, i] = self.butter_bandstop_filter(eeg_raw[:, i], 115, 125, self.fs)            
            # 4 - 40 Hz bandpass
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
# %% # ===========================[RealtimePlot-]============================ # 

# %% # ===========================[UI介面+]============================ # 

# -*- coding: utf-8 -*-
# Form implementation generated from reading ui file 'e:\mirror_BCI\NewGUI.ui'
# Created by: PyQt5 UI code generator 5.9.2
# WARNING! All changes made in this file will be lost!
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(427, 590)
        MainWindow.setStyleSheet("border-color: rgb(85, 170, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.btnCon = QtWidgets.QPushButton(self.centralwidget)
        self.btnCon.setCheckable(False)
        self.btnCon.setObjectName("btnCon")
        self.gridLayout.addWidget(self.btnCon, 1, 0, 1, 1)
        self.btnSave = QtWidgets.QPushButton(self.centralwidget)
        self.btnSave.setObjectName("btnSave")
        self.gridLayout.addWidget(self.btnSave, 3, 0, 1, 1)
        self.texbConStatus = QtWidgets.QTextBrowser(self.centralwidget)
        self.texbConStatus.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.texbConStatus.sizePolicy().hasHeightForWidth())
        self.texbConStatus.setSizePolicy(sizePolicy)
        self.texbConStatus.setMinimumSize(QtCore.QSize(256, 0))
        self.texbConStatus.setMaximumSize(QtCore.QSize(256, 192))
        self.texbConStatus.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.texbConStatus.setObjectName("texbConStatus")
        self.gridLayout.addWidget(self.texbConStatus, 0, 0, 1, 1)
        self.btnDisCon = QtWidgets.QPushButton(self.centralwidget)
        self.btnDisCon.setObjectName("btnDisCon")
        self.gridLayout.addWidget(self.btnDisCon, 6, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("NCU.png"))
        self.label.setScaledContents(False)
        self.label.setWordWrap(False)
        self.label.setOpenExternalLinks(False)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 7, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 427, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnCon.setText(_translate("MainWindow", "Connect"))
        self.btnSave.setText(_translate("MainWindow", "Save"))
        self.btnDisCon.setText(_translate("MainWindow", "Disconnect"))

class MyMainWindow(QtWidgets. QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        # 好像沒用，不影響程式執行
        # manager = multiprocessing.Manager()
        # self.readDataBuffer = manager.Queue(1024)
        # self.sockets = list()

        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('BCI_DATA')

        # 按鍵功能
        self.btnCon.   clicked.connect(self.StartConnection)  # 連線
        self.btnDisCon.clicked.connect(self.Disconnection)    # 斷線
        self.btnSave.  clicked.connect(self.Savedata)         # 存檔

        # 多線程
        self.queue = Queue()
        self.dt = DataReceiveThreads()  # 建立dt class
        self.multipDataRecv     = multiprocessing.Process(
            target=self.dt.data_recv,     args=(self.queue, )) # 多線程 : 開始接收封包
        self.multipRealtimePlot = multiprocessing.Process(
            target=self.dt.realtime_plot, args=(self.queue, )) # 多線程 : 開始繪圖
        self.texbConStatus.append('****** Program is running ******')

    def StartConnection(self):  # 連線
        self.texbConStatus.append("Waiting for Connections...")
        self.texbConStatus.append("Data Receiving...")

        self.multipDataRecv.start()

    def Savedata(self):  # 存檔 將check_save 變 1
        self.texbConStatus.append("Data Saving...")
        My_function.open_read_write(CHECK_FILE, "w", "1")

        localtime1 = time.asctime(time.localtime(time.time()))
        self.texbConStatus.append(localtime1)

        self.multipRealtimePlot.start()    

    def Disconnection(self):  # 斷線 將check_save 變 0
        self.texbConStatus.append("Data Receive Terminated.")
        My_function.open_read_write(CHECK_FILE, "w", "0")

        localtime2 = time.asctime(time.localtime(time.time()))
        self.texbConStatus.append(localtime2)

        self.multipDataRecv.terminate() # 多線程1 關閉
        self.multipRealtimePlot.terminate()
        self.dt.endRecv = True  
# %% # ===========================[UI介面-]============================ #

class DataReceiveThreads(Ui_MainWindow):
    def __init__(self):
        self.endRecv = False
        self.if_save = "0"
        self.if_done = True
        self.data = ""
        self.count = 0
        self.total_data = ""
        self.plot_ready = False
        self.flag = True
        self.small_data = np.empty((10, 1))

        # 創立當前時間的txt檔案
        data_time = datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H%M')
        self.fileDir = SAVE_FILE + '/{}'.format(data_time)

        if os.path.isdir(self.fileDir):
            shutil.rmtree(self.fileDir)
        os.mkdir(self.fileDir)
        os.mkdir(self.fileDir + '\\1\\')

    def data_recv(self, queue):
        self.queue = queue
        ser = serial.Serial('COM'+str(EEG_COM_PORT), 460800)
        print("Successfull Receive!")

        while True:
            # try:
            # 判斷 Savedata 按鍵是否觸發
            self.if_save = My_function.open_read_write(CHECK_FILE, "r")                   
            if self.if_save == "1":     

                self.time_start = time.time()

                ser.reset_output_buffer() 
                ser.reset_input_buffer()
                ser.flushInput()
                ser.flushOutput()

                self.data_recv_start(ser) 

    def data_recv_start(self, ser):
        while True:                        
            self.if_save = My_function.open_read_write(CHECK_FILE, "r")

            if self.if_save == "0" and self.if_done:
                self.data_recv_end()                
            elif self.if_save == "1":                    
                self.data_recv_loop(ser)
    
    def data_recv_end(self):
        # 結束後寫入最後收到的資料到EEG.txt
        self.save_raw_data()
        self.data = ""
        self.if_done = False

    def data_recv_loop(self, ser):
        self.data = ser.read(32).hex() # 每次讀取 32 bytes(一組EEG data的大小)並轉成16進位。收一次等於 1ms 的時間
        self.count = self.count + 1
        self.total_data = self.total_data + self.data 

        # 每 100 ms 寫入資料到txt的最尾端
        if self.count >= 100:
            self.small_data = self.total_data

            self.plot_ready = True
            self.queue.put(self.total_data)
            self.save_raw_data()

            self.time_end = time.time()
            time_gap = int((self.time_end - self.time_start) * 10) * 0.1
            print('time: {:.1f}'.format(time_gap))                     
        
    def save_raw_data(self):
        f = open('{}/1/{}'.format(self.fileDir, FILE_NAME), "a")
        f.write(self.total_data)
        f.close()
        self.count = 0
        self.total_data = "" 

    def realtime_plot(self, queue):
        rtplot = RealtimePlot(self.fileDir, queue, max_t = 2000)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())    