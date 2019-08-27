import tensorflow as tf
from deepctr.inputs import input_from_feature_columns, get_linear_logit,build_input_features,combined_dnn_input
# from deepctr.input_embedding import preprocess_input_embedding
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.interaction import CIN
from deepctr.layers.utils import concat_fun


def xDeepFM_MTL(linear_feature_columns, dnn_feature_columns, embedding_size=8, hidden_size=(256, 256), cin_layer_size=(256, 256,),
                cin_split_half=True, init_std=0.0001,
                task_net_size=(128,), l2_reg_linear=0.00001, l2_reg_embedding=0.00001,
                seed=1024, ):
    # check_feature_config_dict(feature_dim_dict)
    if len(task_net_size) < 1:
        raise ValueError('task_net_size must be at least one layer')

    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         embedding_size,
                                                                         l2_reg_embedding, init_std,
                                                                         seed)

    linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg=l2_reg_linear, init_std=init_std,
                                    seed=seed, prefix='linear')

    fm_input = concat_fun(sparse_embedding_list, axis=1)

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, 'relu',
                       cin_split_half, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)

    deep_input = tf.keras.layers.Flatten()(fm_input)
    deep_out = DNN(hidden_size)(deep_input)

    finish_out = DNN(task_net_size)(deep_out)
    finish_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(finish_out)

    like_out = DNN(task_net_size)(deep_out)
    like_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(like_out)

    finish_logit = tf.keras.layers.add(
        [linear_logit, finish_logit, exFM_logit])
    like_logit = tf.keras.layers.add(
        [linear_logit, like_logit, exFM_logit])

    output_finish = PredictionLayer('sigmoid', name='finish')(finish_logit)
    output_like = PredictionLayer('sigmoid', name='like')(like_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=[
                                  output_finish, output_like])
    return model
