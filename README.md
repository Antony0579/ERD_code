# ERD_code (use Python)

## 探究虛擬環境下視覺鏡像神經元結合震動刺激之事件非同步腦波律動影響
可參考上方我的論文內容： (如果有開放了的話)
https://hdl.handle.net/11296/56t6px

## 執行步驟

0-1. **準備腦波資料**
   - 準備好 8 通道腦波數據，並轉存為 `.npy` 格式文件。
   - 按照專案的資料夾結構整理所有資料，以確保腳本能正確讀取。
   
1-1. **分析並畫出ERD** (見論文 p.13)
   - `12_ERDs_0717_all.py`
   - 輸入：
   - 腦波資料：1.npy
   - 輸出：
   - ERD 圖表：ERD.png
   - 雜訊紀錄：antibias_list.txt
   - 左、右手 ERD 資料：ERDS_data_left.npy、ERDS_data_right.npy
   
2-1. **ERD 指標分析計算** (見論文 p.15)
   - `23_ERD_compare_9seq_0722.py`
   - 輸入：
   - ERD 資料：ERDS_data_left.npy
   - 輸出：
   - ERD 指標資料：ERDS_9seq_left.npy

2-2. **計算雜訊比例** (見論文 p.14)
   - `24_biaslist_compare1.py`
   - 輸入：
   - 雜訊紀錄：antibias_list.txt
   - 輸出：
   - 雜訊比例(excel整理)：biaslist_2A_ERD.csv
    
2-3. **畫出ERD趨勢圖** (見論文 p.17、p.29~33)
   - `25_compare_team_0718`
   - 輸入：
   - ERD 資料：ERDS_data_left.npy
   - 輸出：
   - ERD 3週趨勢圖：A_left_持續時間.png
  
3-1. 畫出指標分析圖 (見論文 p.15、p.19~28)
   - `35_plotpage_0722.py`
   - 輸入：
   - ERD 資料：ERDS_9seq_left.npy
   - 輸出：
   - 8通道 ERD 趨勢圖：plot_team_A1_left.png
