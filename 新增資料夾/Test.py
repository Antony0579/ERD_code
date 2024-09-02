import numpy as np
import csv

# 假設有一個 (4, 3, 3, 2) 維度的數據
data = np.random.random((4, 3, 3, 2))

# 重新排列數據為 (12, 6)
reshaped_data = data.reshape(12, 6)

# 將數據寫入 .csv 檔案
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 寫入每一行數據
    for row in reshaped_data:
        writer.writerow(row)

print("數據已成功寫入 output.csv")
