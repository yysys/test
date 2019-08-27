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
    # data['time'] = data['generate_time'].apply(change_time)



    sparse_features = ['uid', 'u_region_id', 'item_id', 'author_id', 'music_id', 'g_region_id']
    dense_features = ['duration']

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    target = ['finish', 'like']

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]

    train = data[data['date'] <= 20190707]
    test = data[data['date'] == 20190708]

    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
                        [train[feat.name].values for feat in dense_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
                       [test[feat.name].values for feat in dense_feature_list]

    train_labels = [train[target[0]].values, train[target[1]].values]
    test_labels = [test[target[0]].values, test[target[1]].values]

    model = xDeepFM_MTL({"sparse": sparse_feature_list,
                         "dense": dense_feature_list})
    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights, )
