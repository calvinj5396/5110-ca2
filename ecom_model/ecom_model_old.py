# -*- encoding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

import sail.model as M
import sail.layers as L
from sail import layers
from sail import losses
from sail import initializers
from sail import optimizers
from sail import modules
from sail.feature import FeatureColumnDense

from collections import defaultdict
from features import *
from layer_utils import *
from utils import *
from transformer import *


###### LABEL #######
ecom_data = M.Dataflow(name='ecom_data')

def get_label(ecom_data, label_name, label_idx):
    ecom_data.add_label(label_name=label_name, label_idx=label_idx)
    label = ecom_data.get_label(label_name)
    return label
# no use
label_name2idx = {
    'VALID_DOC_LEN': 0
}
instance_labels = {
    _label_name: get_label(ecom_data, _label_name, _label_idx)
    for _label_name, _label_idx in label_name2idx.items()
}

labels = {}
LABEL_NAME = [
    'ctr_pdp_label_list',
    'buy_order_label_list',
    'cart_order_label_list',
    'click_label_list',
    "all_mid_click_label_list",
    "buy_mid_click_label_list",
    "cart_mid_click_label_list",
]
for cur_label_name in LABEL_NAME:
    vec = FeatureColumnDense("label_" + cur_label_name, MAX_DOC_LEN).get_tensor()
    labels[cur_label_name] = vec
    # labels[cur_label_name] = labels[cur_label_name][:, :MAX_DOC_LEN]
    tf.summary.histogram('label/{}'.format(cur_label_name), vec)
    print("label_" + cur_label_name, labels[cur_label_name])

show_labels = {}
SHOW_LABEL_NAME = [
    'show_label_list'
]
for cur_label_name in SHOW_LABEL_NAME:
    vec = FeatureColumnDense("label_" + cur_label_name, MAX_DOC_LEN).get_tensor()
    show_labels[cur_label_name] = vec
    tf.summary.histogram('label/{}'.format(cur_label_name), vec)
    print("label_" + cur_label_name, show_labels[cur_label_name])

ones_2d = tf.ones_like(labels['ctr_pdp_label_list'])  # batch_size, max_doc_len
zeros_2d = tf.zeros_like(labels['ctr_pdp_label_list'])

labels['CTR_LABEL'] = tf.cast(tf.add_n(list(labels.values())) > 0.5, M.get_dtype())
# tf.where(labels['ctr_pdp_label_list'] > 0.5, ones_2d, zeros_2d)
labels['CVR_LABEL'] = tf.where((labels['buy_order_label_list'] + labels['cart_order_label_list']) > 0.5, ones_2d, zeros_2d)
labels['SHOW_LABEL'] = tf.where(show_labels['show_label_list'] > 0.5, ones_2d, zeros_2d)
tf.summary.histogram('label/click_label_list_norm', labels['CTR_LABEL'])
tf.summary.histogram('label/order_label_list_norm', labels['CVR_LABEL'])
tf.summary.histogram('label/show_label_list_norm', labels['SHOW_LABEL'])
print('CTR_LABEL=', labels['CTR_LABEL'])
print('CVR_LABEL=', labels['CVR_LABEL'])
print('SHOW_LABEL=', labels['SHOW_LABEL'])

print()
invalid_label = tf.fill(tf.shape(labels['CTR_LABEL']), M.get_dtype().min)
show_dataset_mask = labels['SHOW_LABEL'] > 0.5  # bool
show_dataset_weight = tf.where(show_dataset_mask, ones_2d, zeros_2d)
click_dataset_mask = labels['CTR_LABEL'] > 0.5  # bool
click_dataset_weight = tf.where(click_dataset_mask, ones_2d, zeros_2d)
buy_dataset_mask = labels['CVR_LABEL'] > 0.5
buy_dataset_weight = tf.where(buy_dataset_mask, ones_2d, zeros_2d)
print('invalid_label=', invalid_label)
print('show_dataset_mask=', show_dataset_mask)
print('show_dataset_weight=', show_dataset_weight)
print('click_dataset_mask=', click_dataset_mask)
print('click_dataset_weight=', click_dataset_weight)
print('buy_dataset_mask=', buy_dataset_mask)
print('buy_dataset_weight=', buy_dataset_weight)
print()

###### MODEL #######
uid_universal_emb = get_dense('fc_tiktok_ue_uid_472510_uidtype')
uid_universal_emb = reset_ue(uid_universal_emb, S.is_training())
uid_universal_nn = modules.DenseTower(name='uid_universal_nn',
                                      output_dims=[32],
                                      activations=layers.LeakyRelu(),
                                      initializers=initializers.GlorotUniform(scale=1, mode='fan_avg'),
                                      use_bias=False)(uid_universal_emb)
USER_CONFIG = [
    (UID, 32, 'uid'),
    (USER_PROFILE, 4, 'user_profile'),
    (uid_universal_nn, 'user_universal_emb')
]
QUERY_CONFIG = [
    (QID, 32, 'qid'),
    (QUERY_INFO, 8, 'query_info'),
    (QUERY_INTENT, 'dense', 'query_intent')
]
DOC_CONFIG = [
    (GOODS_ID, 32, 'goods_id'),
    (VIDEO_ID, 32, 'video_id'),
    (LIVE_ID, 32, 'live_id'),
    (SHOP_ID, 16, 'shop_id'),
    # (AUTHOR_ID, 16, 'author_id'),
    (BRAND_ID, 16, 'brand_id'),
    (SIMID, 32, 'sim_id'),
    (DOC_TYPE, 4, 'doc_type'),
    (DOC_PRICE, 8, 'doc_price_slot'),
    (DOC_DENSE_INFO, 'dense', 'doc_price_dense')
]
CONTEXT_CONFIG = [
    (CONTEXT_INFO, 4, 'context_info')
]

LISTWISE_CONFIG = [
    (LINE_ID_SCORE, 'dense', 'finerank_score'),
    # (POSITION, 'dense', 'position'),
    # (ACCUMULATE_NUM, 'dense', 'accumulate_num'),
    (CROSS_SCORE, 'dense', 'cross_score'),
    # (RANK_FEATURE, 'dense', 'rank_feature')
    (DENSE_SCORE_BUCKET_NAME_CTR + DENSE_SCORE_BUCKET_NAME_CVR + DENSE_SCORE_BUCKET_NAME_CART_CVR + DENSE_SCORE_BUCKET_NAME_BUY_CART_CVR, 'dense', 'bucket_score')
]

def get_dense_layer(input_layer, name, dims):
    nn = modules.DenseTower(name='dense_tower_' + name,
                            output_dims=dims,
                            activations=layers.LeakyRelu(),
                            initializers=initializers.GlorotUniform(scale=1, mode='fan_avg'),
                            use_bias=False)
    tower_output = nn(input_layer)
    return tower_output

def load_embedding(FEATURE_CONFIG, name):
    embeddings = []
    embeddings_name = []
    embeddings_dim = []
    for config in FEATURE_CONFIG:
        if len(config) == 3:
            for c in config[0]:
                if config[1] == 'dense':
                    vec = get_dense(c)
                    if (name == 'doc_embed' or name == 'listwise_embed') and (config[2] != 'rank_feature' and config[2] != 'bucket_score'):
                        vec = tf.expand_dims(vec, axis=-1)
                    # if config[2] == 'bucket_score':
                    #     vec = tf.Print(vec, [vec], message="tf print bucket score" + c, summarize=1000, first_n=1000)
                else:
                    vec = get_vector(c, config[1], slice_name='DEEP')
                embeddings.append(vec)
                embeddings_name.append(config[2] + '_' + str(c))
                embeddings_dim.append(vec.get_shape().as_list()[-1])
                tf.summary.histogram('feature_' + name + '/' + config[2] + '_' + str(c), vec)
                # tf.summary.histogram('shape_' + name + '/' + config[2] + '_' + str(c), vec.shape)
                print("--load embed..." + config[2] + '_' + str(c), vec)
        else:
            embeddings.append(config[0])
            embeddings_name.append(config[1])
            embeddings_dim.append(config[0].get_shape().as_list()[-1])
            tf.summary.histogram('feature_' + name + '/' + config[1], config[0])
            # tf.summary.histogram('shape_' + name + '/' + config[1], config[0].shape)
            print("--load embed..." + config[1], config[0])
    return embeddings, embeddings_name, embeddings_dim

# mask tensor
print()
seq_weight = get_3d_length(GOODS_ID[0])  # shape is (Batch_Size, MAX_DOC_LEN), 1 or 0
seq_mask = tf.greater(seq_weight, 0.5)  # bool
print('seq_weight=', seq_weight)
print('seq_mask=', seq_mask)
tf.summary.histogram("doc_len/seq_len_mean", tf.reduce_mean(tf.reduce_sum(tf.cast(seq_weight, M.get_dtype()), axis=-1)))
tf.summary.histogram("doc_len/seq_len_max", tf.reduce_max(tf.reduce_sum(tf.cast(seq_weight, M.get_dtype()), axis=-1)))

input_doc_weight = tf.where(fc_dict['fc_shoptab_pure_ecom_doc_index_mask_list'] > 0.5, zeros_2d, ones_2d)
seq_weight = tf.multiply(seq_weight, input_doc_weight)
input_doc_weight_two = tf.where(fc_dict['raw_fc_shoptab_pure_ecom_doc_index_transform_list'] > MAX_DOC_LEN, zeros_2d, ones_2d)
seq_weight = tf.multiply(seq_weight, input_doc_weight_two)
print('input_doc_weight=', input_doc_weight)
print('input_doc_weight_two=', input_doc_weight_two)
print('seq_weight_new=', seq_weight)

seq_mask = tf.greater(seq_weight, 0.5)
print('seq_mask_new=', seq_mask)

transform_doc_ele_mask = tf.where(fc_dict['fc_shoptab_pure_ecom_doc_index_mask_list'] > 0.5, zeros_2d, ones_2d)
transform_doc_ele_mask = tf.multiply(transform_doc_ele_mask, input_doc_weight_two)
print('transform_doc_ele_mask', transform_doc_ele_mask)

print()
user_embeddings, user_embeddings_name, user_embeddings_dim = load_embedding(USER_CONFIG, name='user_embed')
print('user_embeddings=', user_embeddings)
print('user_embeddings_name=', user_embeddings_name)
print('user_embeddings_dim=', user_embeddings_dim)
user_embeddings = tf.concat(user_embeddings, axis=1)
print('user_embeddings_before_tile=', user_embeddings)
user_embeddings = tf.expand_dims(user_embeddings, axis=1)
user_embeddings = tf.tile(user_embeddings, [1, MAX_DOC_LEN, 1])  # need mask
print('user_embeddings_before_densetower=', user_embeddings)
user_embeddings_nn = get_dense_layer(user_embeddings, 'user_embedding', [128, 64])
print('user_embeddings_after_densetower=', user_embeddings_nn)
print()

query_embeddings, query_embeddings_name, query_embeddings_dim = load_embedding(QUERY_CONFIG, name='query_embed')
print('query_embeddings=', query_embeddings)
print('query_embeddings_name=', query_embeddings_name)
print('query_embeddings_dim=', query_embeddings_dim)
query_embeddings = tf.concat(query_embeddings, axis=1)
print('query_embeddings_before_tile=', query_embeddings)
query_embeddings = tf.expand_dims(query_embeddings, axis=1)
query_embeddings = tf.tile(query_embeddings, [1, MAX_DOC_LEN, 1])
print('query_embeddings_before_densetower=', query_embeddings)
query_embeddings_nn = get_dense_layer(query_embeddings, 'query_embedding', [128, 64])
print('query_embeddings_after_densetower=', query_embeddings_nn)
print()

context_embeddings, context_embeddings_name, context_embeddings_dim = load_embedding(CONTEXT_CONFIG, name='context_embed')
print('context_embeddings=', context_embeddings)
print('context_embeddings_name=', context_embeddings_name)
print('context_embeddings_dim=', context_embeddings_dim)
context_embeddings = tf.concat(context_embeddings, axis=1)
print('context_embeddings_before_tile=', context_embeddings)
context_embeddings = tf.expand_dims(context_embeddings, axis=1)
context_embeddings = tf.tile(context_embeddings, [1, MAX_DOC_LEN, 1])
print('context_embeddings_before_densetower=', context_embeddings)
context_embeddings_nn = get_dense_layer(context_embeddings, 'context_embedding', [64, 32])
print('context_embeddings_after_densetower=', context_embeddings_nn)
print()

doc_embeddings, doc_embeddings_name, doc_embeddings_dim = load_embedding(DOC_CONFIG, name='doc_embed')
print('doc_embeddings=', doc_embeddings)
print('doc_embeddings_name=', doc_embeddings_name)
print('doc_embeddings_dim=', doc_embeddings_dim)
doc_embeddings = tf.concat(doc_embeddings, axis=2)
print('doc_embeddings_before_densetower=', doc_embeddings)
doc_embeddings_nn = get_dense_layer(doc_embeddings, 'doc_embedding', [128, 64])
print('doc_embeddings_after_densetower=', doc_embeddings_nn)
print()

listwise_embeddings, listwise_embeddings_name, listwise_embeddings_dim = load_embedding(LISTWISE_CONFIG, name='listwise_embed')
print('listwise_embeddings=', listwise_embeddings)
print('listwise_embeddings_name=', listwise_embeddings_name)
print('listwise_embeddings_dim=', listwise_embeddings_dim)
listwise_embeddings = tf.concat(listwise_embeddings, axis=2)
print('listwise_embeddings_before_densetower=', listwise_embeddings)
listwise_embeddings_nn = get_dense_layer(listwise_embeddings, 'listwise_embedding', [128, 64])
print('listwise_embeddings_after_densetower=', listwise_embeddings_nn)
print()

all_doc_embeddings = tf.concat([user_embeddings_nn, query_embeddings_nn, context_embeddings_nn, doc_embeddings_nn, listwise_embeddings_nn], axis=2)  # (batch_size, doc_len, emb_dim)
print('all_doc_embeddings_before_densetower=', all_doc_embeddings)
all_doc_embeddings = get_dense_layer(all_doc_embeddings, 'all_doc_embedding', [256])
print('all_doc_embeddings_after_densetower=', all_doc_embeddings)

# add position embedding
# position_embedding_layer = PositionEmbedding(name='position_embedding_layer')
# all_doc_embeddings_with_pos = position_embedding_layer(all_doc_embeddings)
# print('all_doc_embeddings_with_pos=', all_doc_embeddings_with_pos)
# print()
# add position embedding based on index feature
all_doc_embeddings_with_pos  = all_doc_embeddings + fc_dict['fc_shoptab_pure_ecom_doc_index_transform_list']
print('all_doc_embeddings_with_pos=', all_doc_embeddings_with_pos)

transformer1 = TransformerEncoder(d_word_emb=64, num_head=4, mask=transform_doc_ele_mask, name='transformer_layer_1')
transformer1_output = transformer1(all_doc_embeddings_with_pos, mask=transform_doc_ele_mask)
print('transformer1_output=', transformer1_output)

transformer2 = TransformerEncoder(d_word_emb=64, num_head=4, mask=transform_doc_ele_mask, name='transformer_layer_2')
transformer2_output = transformer2(transformer1_output, mask=transform_doc_ele_mask)
print('transformer2_output=', transformer2_output)

transformer3 = TransformerEncoder(d_word_emb=64, num_head=4, mask=transform_doc_ele_mask, name='transformer_layer_3')
transformer3_output = transformer3(transformer2_output, mask=transform_doc_ele_mask)
print('transformer3_output=', transformer3_output)
print()

def get_output_dense_layer(input_layer, name, dims):
    nn = modules.DenseTower(name='output_dense_tower_' + name,
                            output_dims=dims,
                            activations=layers.LeakyRelu(),
                            initializers=initializers.GlorotUniform(scale=1, mode='fan_avg'),
                            use_bias=False)
    output = nn(input_layer)
    logit = tf.reduce_sum(output, axis=-1)
    pred = tf.sigmoid(logit)
    return logit, pred

show_head_dense_input = transformer3_output
ctr_head_dense_input = tf.concat([transformer3_output, tf.expand_dims(fc_dict['fc_ecom_predict_ctr_list'], axis=-1), fc_dict['fc_ecom_predict_ctr_bucket_list']], axis=2)
cvr_head_dense_input = tf.concat([transformer3_output, tf.expand_dims(fc_dict['fc_ecom_predict_cvr_list'], axis=-1), fc_dict['fc_ecom_predict_cvr_bucket_list'], tf.expand_dims(fc_dict['fc_ecom_predict_cart_cvr_list'], axis=-1), fc_dict['fc_ecom_predict_cart_cvr_bucket_list'], tf.expand_dims(fc_dict['fc_ecom_predict_buy_cart_cvr_list'], axis=-1), fc_dict['fc_ecom_predict_buy_cart_cvr_bucket_list']], axis=2)
print("show_head_dense_input=", show_head_dense_input)
print("ctr_head_dense_input=", ctr_head_dense_input)
print("cvr_head_dense_input=", cvr_head_dense_input)

show_head_logit, show_head_pred = get_output_dense_layer(show_head_dense_input, 'show_head', [512, 128, 1])
tf.summary.histogram('output/show_head_logit', show_head_logit)
tf.summary.histogram('output/show_head_pred', show_head_pred)
print('show_head_logit=', show_head_logit)
print('show_head_pred=', show_head_pred)

ctr_head_logit, ctr_head_pred = get_output_dense_layer(ctr_head_dense_input, 'ctr_head', [512, 128, 1])
tf.summary.histogram('output/ctr_head_logit', ctr_head_logit)
tf.summary.histogram('output/ctr_head_pred', ctr_head_pred)
print('ctr_head_logit=', ctr_head_logit)
print('ctr_head_pred=', ctr_head_pred)

cvr_head_logit, cvr_head_pred = get_output_dense_layer(cvr_head_dense_input, 'cvr_head', [512, 128, 1])
tf.summary.histogram('output/cvr_head_logit', cvr_head_logit)
tf.summary.histogram('output/cvr_head_pred', cvr_head_pred)
print('cvr_head_logit=', cvr_head_logit)
print('cvr_head_pred=', cvr_head_pred)

# session wise label and pred
show_head_pred_session = tf.reduce_sum(tf.where(seq_mask, show_head_pred, tf.zeros_like(show_head_pred)), axis=-1, keepdims=True)
ctr_head_pred_session = tf.reduce_sum(tf.where(seq_mask, show_head_pred * ctr_head_pred, tf.zeros_like(show_head_pred)), axis=-1, keepdims=True)
cvr_head_pred_session = tf.reduce_sum(tf.where(seq_mask, show_head_pred * ctr_head_pred * cvr_head_pred, tf.zeros_like(show_head_pred)), axis=-1, keepdims=True)
tf.summary.histogram('output/show_head_pred_session', show_head_pred_session)
tf.summary.histogram('output/ctr_head_pred_session', ctr_head_pred_session)
tf.summary.histogram('output/cvr_head_pred_session', cvr_head_pred_session)
print('show_head_pred_session=', show_head_pred_session)
print('ctr_head_pred_session=', ctr_head_pred_session)
print('cvr_head_pred_session=', cvr_head_pred_session)
ctr_label_session = tf.reduce_sum(labels['CTR_LABEL'], axis=-1, keepdims=True)
cvr_label_session = tf.reduce_sum(labels['CVR_LABEL'], axis=-1, keepdims=True)
show_label_session = tf.reduce_sum(labels['SHOW_LABEL'], axis=-1, keepdims=True)
tf.summary.histogram('output/ctr_label_session', ctr_label_session)
tf.summary.histogram('output/cvr_label_session', cvr_label_session)
tf.summary.histogram('output/show_label_session', show_label_session)
print('ctr_label_session=', ctr_label_session)
print('cvr_label_session=', cvr_label_session)
print('show_label_session=', show_label_session)

###### INFERENCE #######
run_predict = M.RunStep(name='predict_online', run_type='INFERENCE')
run_predict.add_feeds(M.get_all_input_feature_columns())
run_predict.add_head(name='pred_show_list', prediction=show_head_pred)
run_predict.add_head(name='pred_ctr_list', prediction=ctr_head_pred)
run_predict.add_head(name='pred_cvr_list', prediction=cvr_head_pred)

###### TRAIN #######
run_train = M.RunStep(name='step_train', run_type='TRAIN', data_flow=ecom_data)
run_train.add_feeds(M.get_all_input_feature_columns())

show_head_logit_flatten = tf.reshape(show_head_logit, [-1])
ctr_head_logit_flatten = tf.reshape(ctr_head_logit, [-1])
cvr_head_logit_flatten = tf.reshape(cvr_head_logit, [-1])
print()
print('show_head_logit_flatten=', show_head_logit_flatten)
print('ctr_head_logit_flatten=', ctr_head_logit_flatten)
print('cvr_head_logit_flatten=', cvr_head_logit_flatten)
# show_head_pred_flatten = tf.reshape(show_head_pred, [-1])
# ctr_head_pred_flatten = tf.reshape(ctr_head_pred, [-1])
# cvr_head_pred_flatten = tf.reshape(cvr_head_pred, [-1])
# print()
# print('show_head_pred_flatten=', show_head_pred_flatten)
# print('ctr_head_pred_flatten=', ctr_head_pred_flatten)
# print('cvr_head_pred_flatten=', cvr_head_pred_flatten)

ctcvr_head_pred = ctr_head_pred * cvr_head_pred
# ctcvr_head_pred_flatten = tf.reshape(ctcvr_head_pred, [-1])
print('ctcvr_head_pred=', ctcvr_head_pred)
# print('ctcvr_head_pred_flatten=', ctcvr_head_pred_flatten)

show_label_flatten = tf.reshape(labels['SHOW_LABEL'], [-1])
ctr_label_flatten = tf.reshape(labels['CTR_LABEL'], [-1])
cvr_label_flatten = tf.reshape(labels['CVR_LABEL'], [-1])
invalid_label_flatten = tf.reshape(invalid_label, [-1])
print()
print('show_label_flatten=', show_label_flatten)
print('ctr_label_flatten=', ctr_label_flatten)
print('cvr_label_flatten=', cvr_label_flatten)
print('invalid_label_flatten=', invalid_label_flatten)
print()

seq_weight_flatten = tf.reshape(seq_weight, [-1])
seq_mask_flatten = tf.reshape(seq_mask, [-1])
print('seq_weight_flatten=', seq_weight_flatten)  # 0 or 1
print('seq_mask_flatten=', seq_mask_flatten)  # bool
print()

show_dataset_weight_flatten = tf.reshape(show_dataset_weight, [-1])
click_dataset_weight_flatten = tf.reshape(click_dataset_weight, [-1])
buy_dataset_weight_flatten = tf.reshape(buy_dataset_weight, [-1])
print('show_dataset_weight_flatten=', show_dataset_weight_flatten)
print('click_dataset_weight_flatten=', click_dataset_weight_flatten)
print('buy_dataset_weight_flatten=', buy_dataset_weight_flatten)

# show_dataset_mask_flatten = tf.reshape(show_dataset_mask, [-1])
# click_dataset_mask_flatten = tf.reshape(click_dataset_mask, [-1])
# buy_dataset_mask_flatten = tf.reshape(buy_dataset_mask, [-1])
# print('show_dataset_mask_flatten=', show_dataset_mask_flatten)
# print('click_dataset_mask_flatten=', click_dataset_mask_flatten)
# print('buy_dataset_mask_flatten=', buy_dataset_mask_flatten)
# print()

show_loss = losses.binary_cross_entropy(
                        y_label=show_label_flatten,
                        y_pred=show_head_logit_flatten,
                        name='show_loss',
                        from_logits=True,
                        reduce_sum=False,
                        weight=seq_weight_flatten)
ctr_loss = losses.binary_cross_entropy(
                        y_label=ctr_label_flatten,
                        y_pred=ctr_head_logit_flatten,
                        name='ctr_loss',
                        from_logits=True,
                        reduce_sum=False,
                        weight=seq_weight_flatten * show_dataset_weight_flatten)
cvr_loss = losses.binary_cross_entropy(
                        y_label=cvr_label_flatten,
                        y_pred=cvr_head_logit_flatten,
                        name='cvr_loss',
                        from_logits=True,
                        reduce_sum=False,
                        weight=seq_weight_flatten * click_dataset_weight_flatten)
tf.summary.histogram('loss/show_loss', show_loss)
tf.summary.histogram('loss/ctr_loss', ctr_loss)
tf.summary.histogram('loss/cvr_loss', cvr_loss)
print('show_loss=', show_loss)
print('ctr_loss=', ctr_loss)
print('cvr_loss=', cvr_loss)

run_train.add_head(name='show_head',
                   prediction=show_head_pred,
                   label=tf.where(seq_mask, labels['SHOW_LABEL'], invalid_label),
                   loss=show_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='ctr_head',
                   prediction=ctr_head_pred,
                   label=tf.where(tf.logical_and(seq_mask, show_dataset_mask), labels['CTR_LABEL'], invalid_label),
                   loss=5 * ctr_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='cvr_head',
                   prediction=cvr_head_pred,
                   label=tf.where(tf.logical_and(seq_mask, click_dataset_mask), labels['CVR_LABEL'], invalid_label),
                   loss=200 * cvr_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='ctcvr_head',
                   prediction=ctcvr_head_pred,
                   label=tf.where(tf.logical_and(seq_mask, show_dataset_mask), labels['CVR_LABEL'], invalid_label),
                   loss=0.0 * cvr_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='show_head_session',
                   prediction=show_head_pred_session,
                   label=show_label_session,
                   loss=0.0 * show_loss,
                   classifier_type='REGRESSION',
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='ctr_head_session',
                   prediction=ctr_head_pred_session,
                   label=ctr_label_session,
                   loss=0.0 * ctr_loss,
                   classifier_type='REGRESSION',
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='cvr_head_session',
                   prediction=cvr_head_pred_session,
                   label=cvr_label_session,
                   loss=0.0 * cvr_loss,
                   classifier_type='REGRESSION',
                   sample_rate=M.get_sample_rate())

def print_model():
    nn = defaultdict(list)
    all_var_cnt = 0
    nn_shape = {}
    trainable_vars = {
        vec.name: vec.get_shape().as_list()
        for vec in tf.trainable_variables() if '_shard_' not in vec.name
    }
    for name, shape in trainable_vars.items():
        tower = name.split('/')[0].replace('virtual_layer_for_variable_', '')
        nn[tower].append((name, shape))

    for k, v in nn.items():
        nn[k] = sorted(v)
        shape = [d for v in nn[k] if 'bias' not in v[0] for d in v[1]]
        shape = [shape[0]] + shape + [shape[-1]]
        shape = shape[::2]
        cnt = sum([np.multiply.reduce(v[1]) for v in nn[k]])
        all_var_cnt += cnt
        nn_shape[k] = (tuple(shape), cnt)

    print('nn size:%.2fM' % (all_var_cnt / 1024.0 / 1024))
    for d in sorted(list(nn_shape.items())):
        print(d, '%.4f' % (1.0 * d[1][1] / all_var_cnt))
    print("model struct")
    print(nn_shape)

    return nn_shape

M.set_global_gradient_clip_norm(5000)
# compile the whole model for training
M.compile(default_hidden_layer_optimizer=optimizers.RMSPropV2(lr=.01, momentum=.99995, init_factor=0.015625,
          lr_warmup_linear_steps=100000, lr_warmup_linear_init=0.000001),
          run_steps=[run_train, run_predict],
          num_estimated_bias_features=10 * 1000 * 1000,
          num_estimated_vec_features=10 * 1000 * 1000)

model = print_model()
