# ERD_code (use Python)

## 探究虛擬環境下視覺鏡像神經元結合震動刺激之事件非同步腦波律動影響
可參考上方我的論文內容： (如果有開放了的話)
https://hdl.handle.net/11296/56t6px

## 執行步驟

### 0-1. 準備好你的 8通道腦波資料並轉為.npy檔
### 0-2. 按照我的資料夾整理方式排好
   
1. 分析並畫出ERD (見論文 p.13)
-12_ERDs_0717_all.py
-輸入：
-腦波資料：1.npy
      輸出：
        ERD 圖表：ERD.png
        雜訊紀錄：antibias_list.txt
        左、右手 ERD 資料：ERDS_data_left.npy、ERDS_data_right.npy
   
# ERD 指標分析計算

`23_ERD_compare_9seq_0722.py` 是用於分析 ERD（事件相關去同步化）指標的 Python 腳本。該腳本從提供的 ERD 資料中計算並輸出 ERD 指標資料，根據研究所需對 ERD 數據進行比較和分析。

## 文件資訊
- **論文章節**：見論文第 15 頁
- **檔案名稱**：`23_ERD_compare_9seq_0722.py`

## 輸入資料
- **ERD 資料**：`ERDS_data_left.npy`  
  包含原始 ERD 數據，將用作分析的輸入資料。

## 輸出資料
- **ERD 指標資料**：`ERDS_9seq_left.npy`  
  生成的 ERD 指標結果資料，以 `.npy` 格式儲存，供進一步分析使用。

## 使用方式

1. 將 `ERDS_data_left.npy` 放置於腳本的相同資料夾下，並確認名稱無誤。
2. 在終端機執行以下指令：

   ```bash
   python 23_ERD_compare_9seq_0722.py


2-2. 計算雜訊比例 (見論文 p.14)
     24_biaslist_compare1.py
      輸入：
        雜訊紀錄：antibias_list.txt
      輸出：
        雜訊比例(excel整理)：biaslist_2A_ERD.csv
    
2-3. 畫出ERD趨勢圖 (見論文 p.17、p.29~33)
     25_compare_team_0718
      輸入：
        ERD 資料：ERDS_data_left.npy
      輸出：
        ERD 3週趨勢圖：A_left_持續時間.png
  
3. 畫出指標分析圖 (見論文 p.15、p.19~28)
   35_plotpage_0722.py
      輸入：
        ERD 資料：ERDS_9seq_left.npy
      輸出：
        8通道 ERD 趨勢圖：plot_team_A1_left.png
