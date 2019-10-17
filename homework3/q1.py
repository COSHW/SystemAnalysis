import csv
import numpy as np
f = open('DlyaAndrianovoy.csv', 'r+')
reader = csv.reader(f)
data = list()
for row in reader:
    data.append([row[0].split(';')[1], row[0].split(';')[2]])
f.close()
print(np.r_[np.array(data)])
