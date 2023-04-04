import pandas as pd
import csv


df = pd.read_excel('sheets_with_data.xlsx',engine='openpyxl',dtype=object,header=0)
excelList:list = df.values.tolist()

f = open('names.csv', 'w')
writer = csv.writer(f)
header = ['code', 'word']
writer.writerow(header)

for i in range(75):
    oneRow=excelList[i]
    writer.writerow([oneRow[8],oneRow[9]])
f.close()