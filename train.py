import pandas as pd
import time, datetime
from deepctr import SingleFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from model import xDeepFM_MTL

loss_weights = [1, 1, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

def change_time(timeStamp):
    dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
    otherStyleTime = dateArray.strftime("%Y%m%d")

    return int(otherStyleTime)

if __name__ == "__main__":
    data = pd.read_csv('./data/bytecamp.data', sep=',', header=0)
   # Index(['duration', 'generate_time', 'finish', 'like', 'date', 'uid',
   #        'u_region_id', 'item_id', 'author_id', 'music_id', 'g_region_id'],
   #       dtype='object')
    data['time'] = data['generate_time'].apply(change_time)

    train_data = data[data['time'] <= 20190707]
    test_data = data[data['time'] == 20190708]

    print(len(train_data))
    print(len(test_data))
    print(len(data))