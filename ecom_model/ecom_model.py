# -*- encoding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

import sail.model as M
import sail.common as S
from sail import layers
from sail import losses
from sail import initializers
from sail import optimizers
from sail import modules
from sail.feature import FeatureColumnDense
from layer_utils import LayerNorm
from collections import defaultdict
from features import *
from utils import *
from transformer import *
from rq_vae_encoder import create_rq_vae_tokenmixer_integration

S.enable_option_use_layer_id_in_namespace(use_name_scope=True)

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
    tf.summary.scalar('label/{}'.format(cur_label_name), tf.reduce_mean(vec))
    print("label_" + cur_label_name, labels[cur_label_name])

show_labels = {}
SHOW_LABEL_NAME = [
    'show_label_list'
]
for cur_label_name in SHOW_LABEL_NAME:
    vec = FeatureColumnDense("label_" + cur_label_name, MAX_DOC_LEN).get_tensor()
    show_labels[cur_label_name] = vec
    tf.summary.scalar('label/{}'.format(cur_label_name), tf.reduce_mean(vec))
    print("label_" + cur_label_name, show_labels[cur_label_name])

ones_2d = tf.ones_like(labels['ctr_pdp_label_list'])  # batch_size, max_doc_len
zeros_2d = tf.zeros_like(labels['ctr_pdp_label_list'])

labels['CTR_LABEL'] = tf.cast(tf.add_n(list(labels.values())) > 0.5, M.get_dtype())
# tf.where(labels['ctr_pdp_label_list'] > 0.5, ones_2d, zeros_2d)
labels['CVR_LABEL'] = tf.where((labels['buy_order_label_list'] + labels['cart_order_label_list']) > 0.5, ones_2d,
                               zeros_2d)
labels['SHOW_LABEL'] = tf.where(show_labels['show_label_list'] > 0.5, ones_2d, zeros_2d)
tf.summary.scalar('label/click_label_list_norm', tf.reduce_mean(labels['CTR_LABEL']))
tf.summary.scalar('label/order_label_list_norm', tf.reduce_mean(labels['CVR_LABEL']))
tf.summary.scalar('label/show_label_list_norm', tf.reduce_mean(labels['SHOW_LABEL']))

invalid_label = tf.fill(tf.shape(labels['CTR_LABEL']), M.get_dtype().min)
show_dataset_mask = labels['SHOW_LABEL'] > 0.5  # bool
show_dataset_weight = tf.where(show_dataset_mask, ones_2d, zeros_2d)
click_dataset_mask = labels['CTR_LABEL'] > 0.5  # bool
click_dataset_weight = tf.where(click_dataset_mask, ones_2d, zeros_2d)
buy_dataset_mask = labels['CVR_LABEL'] > 0.5
buy_dataset_weight = tf.where(buy_dataset_mask, ones_2d, zeros_2d)

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
NEW_FEATURE_FIRST_CONFIG = [
    (SLOT_FIRST_NEW8_FIL, 4, 'new_slot_share_feature')
]

NEW_FEATURE_CONFIG = [
    (SLOT_NEW8_FIL, 4, 'new_slot_doc_feature')
]

print("total new first feature", len(SLOT_FIRST_NEW8_FIL))
print("total new doc features", len(SLOT_NEW8_FIL))
LISTWISE_CONFIG = [
    (LINE_ID_SCORE, 'dense', 'finerank_score'),
    # (POSITION, 'dense', 'position'),
    # (ACCUMULATE_NUM, 'dense', 'accumulate_num'),
    (CROSS_SCORE, 'dense', 'cross_score'),
    # (RANK_FEATURE, 'dense', 'rank_feature')
    (
    DENSE_SCORE_BUCKET_NAME_CTR + DENSE_SCORE_BUCKET_NAME_CVR + DENSE_SCORE_BUCKET_NAME_CART_CVR + DENSE_SCORE_BUCKET_NAME_BUY_CART_CVR,
    'dense', 'bucket_score')
]

# mask tensor
seq_weight = get_3d_length(GOODS_ID[0])
tf.summary.scalar("doc_len/seq_len_mean", tf.reduce_mean(tf.reduce_sum(tf.cast(seq_weight, M.get_dtype()), axis=-1)))
tf.summary.scalar("doc_len/seq_len_max", tf.reduce_max(tf.reduce_sum(tf.cast(seq_weight, M.get_dtype()), axis=-1)))

# input_doc_weight = tf.where(fc_dict['fc_shoptab_pure_ecom_doc_index_mask_list'] > 0.5, zeros_2d, ones_2d)
# seq_weight = tf.multiply(seq_weight, input_doc_weight)
# input_doc_weight_two = tf.where(fc_dict['raw_fc_shoptab_pure_ecom_doc_index_transform_list'] > MAX_DOC_LEN, zeros_2d, ones_2d)
# seq_weight = tf.multiply(seq_weight, input_doc_weight_two)
# print('input_doc_weight=', input_doc_weight)
# print('input_doc_weight_two=', input_doc_weight_two)
print('seq_weight_new=', seq_weight)

seq_mask = tf.greater(seq_weight, 0.5)
print('seq_mask_new=', seq_mask)

transform_doc_ele_mask = tf.cast(seq_mask, dtype=tf.float32)
print('transform_doc_ele_mask', transform_doc_ele_mask)

user_embeddings_nn = process_all_embeddings(USER_CONFIG, 'user_embed', [128, 64])
layers_map["user_embeddings_nn"] = user_embeddings_nn
print('User_embedding_dim', user_embeddings_nn)
query_embeddings_nn = process_all_embeddings(QUERY_CONFIG, 'query_embed', [128, 64])
layers_map["query_embeddings_nn"] = query_embeddings_nn
context_embeddings_nn = process_all_embeddings(CONTEXT_CONFIG, 'context_embed', [64, 32])
layers_map["context_embeddings_nn"] = context_embeddings_nn
doc_embeddings_nn = process_all_embeddings(DOC_CONFIG, 'doc_embed', [128, 64], need_tile=False, concat_axis=2)
layers_map["doc_embeddings_nn"] = doc_embeddings_nn
listwise_embeddings_nn = process_all_embeddings(LISTWISE_CONFIG, 'listwise_embed', [128, 64], need_tile=False,
                                                concat_axis=2)
layers_map["listwise_embeddings_nn"] = listwise_embeddings_nn
newfeature_first_embeddings_nn = process_all_embeddings(NEW_FEATURE_FIRST_CONFIG, 'new_first_feature', [256, 128])
layers_map["newfeature_first_embeddings_nn"] = newfeature_first_embeddings_nn
newfeature_embeddings_nn = process_all_embeddings(NEW_FEATURE_CONFIG, 'new_feature', [256, 128], need_tile=False,
                                                  concat_axis=2)
layers_map["newfeature_embeddings_nn"] = newfeature_embeddings_nn

for feature_name in layers_map:
    fea_emb = layers_map[feature_name]
    tf.summary.histogram("Feature_Input_%s_dense_in" % (feature_name), fea_emb)

all_doc_embeddings = tf.concat(
    [user_embeddings_nn, query_embeddings_nn, context_embeddings_nn, doc_embeddings_nn, listwise_embeddings_nn,
     newfeature_first_embeddings_nn, newfeature_embeddings_nn], axis=2)  # (batch_size, doc_len, emb_dim)
layers_map["all_doc_embeddings"] = all_doc_embeddings
print("original_all_doc_embedding", all_doc_embeddings)

MODEL_OPTIONS = ["transformer", "tokenmixer", "mixtransformer", "rq_vae_tokenmixer"]
MODEL_PATH = 'transformer'
# MODEL_PATH = 'tokenmixer'
# MODEL_PATH = 'mixtransformer'
# MODEL_PATH = 'rq_vae_tokenmixer'
SHARED_FFN = False
combine_trans_layers = 1
print("Current MODEL_PATH is:", MODEL_PATH)
if MODEL_PATH == 'transformer':
    ## option 1: use self-attention structure to process all the features for a doc-list
    all_doc_embeddings = get_dense_layer(all_doc_embeddings, 'all_doc_embedding', [256])  # input shape[?, 30, 256]
    print("all_doc_embedding", all_doc_embeddings)
    print("type:", type(all_doc_embeddings))
    transformer_output = process_transformer(all_doc_embeddings, transform_doc_ele_mask, use_causal_mask=True,
                                             num_layers=1, d_word_emb=8, num_head=1)
    # output shape [?, 30, 256]

elif MODEL_PATH == 'tokenmixer':
    ##option2, direct change dim to 300 for all 30 docs, each doc is considered as a unique token, use token-mixer of dimension [?, 30, 300],
    all_doc_embeddings = get_dense_layer(all_doc_embeddings, 'all_doc_embedding', [300])
    print("all_doc_embedding", all_doc_embeddings)
    print("type:", type(all_doc_embeddings))
    combine_trans_input = all_doc_embeddings
    trans_dims = combine_trans_input.get_shape().as_list()[-1]
    seq_lens = combine_trans_input.get_shape().as_list()[-2]
    if SHARED_FFN:
        emb_divide_ratio = 1
    else:
        emb_divide_ratio = None
    trans_opts = [None] * combine_trans_layers
    combine_trans = Transformer(width=trans_dims,
                                layers=1,
                                heads=8,
                                dropout=0.0,
                                use_mask=False,
                                name="combine_trans",
                                ##is_vit=False, mixed_precision=mixed_precision,optimizers=trans_opts,
                                use_trick=True,  # trigger for mixup
                                mix_dim=3,
                                optimizers=trans_opts,
                                hidden_emb_divide_ratio=emb_divide_ratio,
                                ##set to None for using individual adaptive FFN for each token, 1 for using shared FFN
                                experts_num=seq_lens,
                                use_adaptiveffn=True
                                )
    combine_trans_out = combine_trans(combine_trans_input)  # output shape [?, 30, 300]
    print("mix-up out", combine_trans_out)
    print("shape mix", tf.shape(combine_trans_out))
    transformer_output = combine_trans_out

elif MODEL_PATH == 'mixtransformer':

    # 1. prepare for the tokens before tokenmixer
    mix_trans_token_list = ["user_embeddings_nn", "query_embeddings_nn", "context_embeddings_nn", "doc_embeddings_nn",
                            "listwise_embeddings_nn"] + ["newtokens" + str(i) for i in range(1, 4)]
    mix_merge_token(["user_embeddings_nn", "query_embeddings_nn", "context_embeddings_nn"], "newtokens1",
                    use_compress=False)
    mix_merge_token(["doc_embeddings_nn", "listwise_embeddings_nn"], "newtokens2", use_compress=False)
    mix_merge_token(["all_doc_embeddings"], "newtokens3", use_compress=True)
    skip_ln_list = []  ## if needed to skip normalization
    concat_list_ln = []
    for idx in range(len(mix_trans_token_list)):
        tensor_name = mix_trans_token_list[idx]
        tensor_in = layers_map[tensor_name]
        if tensor_name in skip_ln_list:
            print('---- %s is skipped ln for trans ----' % tensor_name)
            tensor_out = tensor_in
        else:
            print('---- %s is layernorm for trans ----' % tensor_name)
            print(tensor_in.get_shape())
            tensor_out = LayerNorm(name="mix_ln_" + tensor_name)(tensor_in)
        tf.summary.histogram("all_concat_%s_dense_in" % (tensor_name), tensor_out)
        if tensor_out.get_shape()[-1] > 128:
            trans_dims = [256, 128]
        else:
            trans_dims = [64, 128]
        tensor_out_dense = modules.DenseTower(
            name='combine_trans_%s_dense' % tensor_name,
            output_dims=trans_dims,
            activations=layers.Relu(),
            initializers=initializers.GlorotNormal())(tensor_out)
        tf.summary.histogram("all_concat_%s_dense_out" % (tensor_name), tensor_out_dense)
        concat_list_ln.append(tensor_out_dense)
    combine_trans_input = tf.stack(concat_list_ln, axis=2)
    trans_dims = combine_trans_input.get_shape().as_list()[-1]
    seq_lens = combine_trans_input.get_shape().as_list()[-2]

    # 2. token-mixer
    # convert the input shape from [?, 30, 8, 128] --> [?*30, 8, 128]
    print("input_dimension:", seq_lens, combine_trans_input.get_shape().as_list())
    B = tf.shape(combine_trans_input)
    print("shape B", B[0], B[1], B[2], B[-1])

    combine_trans_input = tf.reshape(combine_trans_input, [B[0] * B[1], B[2], B[3]])
    combine_trans_input.set_shape([None, 8, 128])
    trans_opts = [None] * combine_trans_layers
    combine_trans = Transformer(width=trans_dims,
                                layers=combine_trans_layers,
                                heads=trans_dims // 64,
                                dropout=0.0,
                                use_mask=False,
                                name="combine_trans",
                                ##is_vit=False, mixed_precision=mixed_precision,optimizers=trans_opts,
                                use_trick=True,
                                mix_dim=3,
                                optimizers=trans_opts,
                                experts_num=seq_lens,
                                use_adaptiveffn=True
                                )
    combine_trans_out = combine_trans(combine_trans_input)
    print("mix-up out", combine_trans_out)
    print("shape mix", tf.shape(combine_trans_out))
    transformer_input = tf.reshape(combine_trans_out, [B[0], B[1], B[2] * B[
        3]])  # output shape convert from [?*30, 8, 128] --> [?, 30, 8*128]
    transformer_input.set_shape([None, 30, 1024])
    print("output_shape", transformer_input.shape)

    # 3. the results of token-mixer will be fed into transformer
    transformer_output = process_transformer(transformer_input, transform_doc_ele_mask, use_causal_mask=True,
                                             num_layers=1, d_word_emb=8, num_head=1)

elif MODEL_PATH == 'rq_vae_tokenmixer':
    ## option3: use RQ-VAE to encode all features and use the processed embeddings as tokens for tokenmixer
    print("Using RQ-VAE with tokenmixer")

    # Create RQ-VAE integration with tokenmixer
    transformer_output, rq_vae_loss = create_rq_vae_tokenmixer_integration(
        layers_map=layers_map,
        seq_mask=seq_mask,
        MODEL_PATH='tokenmixer'
    )

    # Add RQ-VAE loss to training objectives
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 0.1 * rq_vae_loss)
    tf.summary.scalar('loss/rq_vae_loss', rq_vae_loss)

    print("RQ-VAE tokenmixer output:", transformer_output)


def get_output_dense_layer(input_layer, name, dims, reduce_output=True, use_res=None):
    nn = modules.DenseTower(name='output_dense_tower_' + name,
                            output_dims=dims,
                            activations=layers.LeakyRelu(),
                            initializers=initializers.GlorotUniform(scale=1, mode='fan_avg'),
                            use_bias=False)
    logit = nn(input_layer)
    if use_res is not None:
        logit += use_res
    if reduce_output:
        logit = tf.reduce_sum(logit, axis=-1)
    tf.summary.histogram('output_dense_tower_' + name + '/' + name, logit)
    pred = tf.sigmoid(logit)
    return logit, pred


def get_logic_and_pred(input_layer, name, dims, append_list, use_res=None):
    dense_input = tf.concat([input_layer] + append_list, axis=-1)
    head_logit, head_pred = get_output_dense_layer(dense_input, name, dims, use_res=use_res)
    logit_flattern = tf.reshape(head_logit, [-1])
    pred_flattern = tf.reshape(head_pred, [-1])
    return head_logit, head_pred, logit_flattern, pred_flattern


fc_ecom_predict_ctr_list = tf.expand_dims(fc_dict['fc_ecom_predict_ctr_list'], axis=-1)
fc_ecom_predict_cvr_list = tf.expand_dims(fc_dict['fc_ecom_predict_cvr_list'], axis=-1)
fc_ecom_predict_cart_cvr_list = tf.expand_dims(fc_dict['fc_ecom_predict_cart_cvr_list'], axis=-1)
fc_ecom_predict_buy_cart_cvr_list = tf.expand_dims(fc_dict['fc_ecom_predict_buy_cart_cvr_list'], axis=-1)

ctr_head_logit, ctr_head_pred, ctr_head_logit_flatten, ctr_head_pred_flatten = get_logic_and_pred(transformer_output,
                                                                                                  'ctr_head',
                                                                                                  [512, 128, 1], [
                                                                                                      fc_ecom_predict_ctr_list,
                                                                                                      fc_dict[
                                                                                                          'fc_ecom_predict_ctr_bucket_list']],
                                                                                                  use_res=fc_ecom_predict_ctr_list)
cvr_head_logit, cvr_head_pred, cvr_head_logit_flatten, cvr_head_pred_flatten = get_logic_and_pred(transformer_output,
                                                                                                  'cvr_head',
                                                                                                  [512, 128, 1], [
                                                                                                      fc_ecom_predict_cvr_list,
                                                                                                      fc_dict[
                                                                                                          'fc_ecom_predict_cvr_bucket_list'],
                                                                                                      fc_ecom_predict_cart_cvr_list,
                                                                                                      fc_dict[
                                                                                                          'fc_ecom_predict_cart_cvr_bucket_list'],
                                                                                                      fc_ecom_predict_buy_cart_cvr_list,
                                                                                                      fc_dict[
                                                                                                          'fc_ecom_predict_buy_cart_cvr_bucket_list']],
                                                                                                  use_res=fc_ecom_predict_cvr_list + fc_ecom_predict_buy_cart_cvr_list)
show_head_logit, show_head_pred = get_output_dense_layer(transformer_output, 'show_head', [512, 128, 1])
show_head_logit_flatten = tf.reshape(show_head_logit, [-1])

print('ctr_head_logit=', ctr_head_logit)
print('cvr_head_logit=', cvr_head_logit)

ctr_session_logit, ctr_session_pred = get_output_dense_layer(tf.stop_gradient(ctr_head_logit), 'ctr_session_head',
                                                             [10, 1], reduce_output=False)
cvr_session_logit, cvr_session_pred = get_output_dense_layer(
    tf.stop_gradient(tf.concat([ctr_head_logit, cvr_head_logit], axis=-1)), 'cvr_session_head', [10, 1],
    reduce_output=False)

# session wise label and pred
show_head_pred_session = tf.reduce_sum(tf.where(seq_mask, show_head_pred, tf.zeros_like(show_head_pred)), axis=-1,
                                       keepdims=True)
ctr_head_pred_session = tf.reduce_sum(tf.where(seq_mask, show_head_pred * ctr_head_pred, tf.zeros_like(show_head_pred)),
                                      axis=-1, keepdims=True)
cvr_head_pred_session = tf.reduce_sum(
    tf.where(seq_mask, show_head_pred * ctr_head_pred * cvr_head_pred, tf.zeros_like(show_head_pred)), axis=-1,
    keepdims=True)
tf.summary.scalar('output/show_head_pred_session', tf.reduce_mean(show_head_pred_session))
tf.summary.scalar('output/ctr_head_pred_session', tf.reduce_mean(ctr_head_pred_session))
tf.summary.scalar('output/cvr_head_pred_session', tf.reduce_mean(cvr_head_pred_session))
ctr_label_session = tf.reduce_sum(labels['CTR_LABEL'], axis=-1, keepdims=True)
cvr_label_session = tf.reduce_sum(labels['CVR_LABEL'], axis=-1, keepdims=True)
show_label_session = tf.reduce_sum(labels['SHOW_LABEL'], axis=-1, keepdims=True)
ones_2d_session = tf.ones_like(show_label_session)
zeros_2d_session = tf.zeros_like(show_label_session)
tf.summary.scalar('output/ctr_label_session', tf.reduce_mean(ctr_label_session))
tf.summary.scalar('output/cvr_label_session', tf.reduce_mean(cvr_label_session))
tf.summary.scalar('output/show_label_session', tf.reduce_mean(show_label_session))

###### INFERENCE #######
run_predict = M.RunStep(name='predict_online', run_type='INFERENCE')
run_predict.add_feeds(M.get_all_input_feature_columns())
run_predict.add_head(name='pred_show_list', prediction=show_head_pred)
run_predict.add_head(name='pred_ctr_list', prediction=ctr_head_pred)
run_predict.add_head(name='pred_cvr_list', prediction=cvr_head_pred)

###### TRAIN #######
run_train = M.RunStep(name='step_train', run_type='TRAIN', data_flow=ecom_data)
run_train.add_feeds(M.get_all_input_feature_columns())

ctcvr_head_pred = ctr_head_pred * cvr_head_pred
print('ctcvr_head_pred=', ctcvr_head_pred)

show_label_flatten = tf.reshape(labels['SHOW_LABEL'], [-1])
ctr_label_flatten = tf.reshape(labels['CTR_LABEL'], [-1])
cvr_label_flatten = tf.reshape(labels['CVR_LABEL'], [-1])
invalid_label_flatten = tf.reshape(invalid_label, [-1])
print('show_label_flatten=', show_label_flatten)
print('ctr_label_flatten=', ctr_label_flatten)
print('cvr_label_flatten=', cvr_label_flatten)
print('invalid_label_flatten=', invalid_label_flatten)

seq_weight_flatten = tf.reshape(seq_weight, [-1])
seq_mask_flatten = tf.reshape(seq_mask, [-1])
print('seq_weight_flatten=', seq_weight_flatten)  # 0 or 1
print('seq_mask_flatten=', seq_mask_flatten)  # bool

show_dataset_weight_flatten = tf.reshape(show_dataset_weight, [-1])
click_dataset_weight_flatten = tf.reshape(click_dataset_weight, [-1])
buy_dataset_weight_flatten = tf.reshape(buy_dataset_weight, [-1])
print('show_dataset_weight_flatten=', show_dataset_weight_flatten)
print('click_dataset_weight_flatten=', click_dataset_weight_flatten)
print('buy_dataset_weight_flatten=', buy_dataset_weight_flatten)

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
ctr_loss_session = losses.binary_cross_entropy(
    y_label=ctr_label_session,
    y_pred=ctr_session_logit,
    name='ctr_loss_session',
    from_logits=True,
    reduce_sum=False,
    weight=ones_2d_session)
cvr_loss_session = losses.binary_cross_entropy(
    y_label=cvr_label_session,
    y_pred=cvr_session_logit,
    name='cvr_loss_session',
    from_logits=True,
    reduce_sum=False,
    weight=ones_2d_session * click_dataset_weight)
tf.summary.scalar('loss/show_loss', show_loss)
tf.summary.scalar('loss/ctr_loss', ctr_loss)
tf.summary.scalar('loss/cvr_loss', cvr_loss)

run_train.add_head(name='seq_mask_head',
                   prediction=tf.cast(seq_mask, dtype=tf.float32),
                   label=tf.ones_like(seq_mask, dtype=tf.float32),
                   loss=0.0 * show_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='show_head',
                   prediction=show_head_pred,
                   label=tf.where(seq_mask, labels['SHOW_LABEL'], invalid_label),
                   loss=0.0 * show_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='ctr_head',
                   prediction=ctr_head_pred,
                   label=tf.where(tf.logical_and(seq_mask, show_dataset_mask), labels['CTR_LABEL'], invalid_label),
                   loss=10 * ctr_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='cvr_head',
                   prediction=cvr_head_pred,
                   label=tf.where(tf.logical_and(seq_mask, click_dataset_mask), labels['CVR_LABEL'], invalid_label),
                   loss=100 * cvr_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='ctcvr_head',
                   prediction=ctcvr_head_pred,
                   label=tf.where(tf.logical_and(seq_mask, show_dataset_mask), labels['CVR_LABEL'], invalid_label),
                   loss=0.0 * cvr_loss_session,
                   sample_rate=M.get_sample_rate())


def add_topk_auc_heads(pred_tensor, label_tensor_with_invalid, run_train, prefix='ctr', topk_list=[4, 8, 10, 20]):
    """
    提取每个样本前 k 个位置（position 顺序，不排序），label_tensor 已包含 invalid_label。
    输出 shape: [batch_size, k]
    """
    for k in topk_list:
        run_train.add_head(
            name='top{}_pos_{}'.format(k, prefix),
            prediction=pred_tensor[:, :k],  # 前 k 个位置
            label=label_tensor_with_invalid[:, :k],  # 对应位置的 label（已包含 invalid_label）
            loss=tf.constant(0.0),
            sample_rate=M.get_sample_rate()
        )


# ground truth: baseline
finerank_ctr = get_dense('fc_ecom_predict_ctr_list')
post_show_rate = [1.0, 0.9986856746112356, 0.9952602045921043, 0.9905639926385651, 0.940915805861861,
                  0.9105900593341357, 0.7890183717524641, 0.7555557534707589, 0.725454556479609, 0.7134238176377313,
                  0.6829229793194614, 0.6716238740309314, 0.6425578788273852, 0.6314492142850646, 0.6032512877963296,
                  0.5863132414872039, 0.5670360901320995, 0.5497800740994522, 0.531806447016163, 0.5172065582866169,
                  0.5013677624936709, 0.4892566785011536, 0.4747278497515364, 0.4636730013006819, 0.4503988075566888,
                  0.4400479266403296, 0.42775985592446947, 0.418115595448523, 0.4066827079044469, 0.3975082012691502,
                  0.3865124747215965, 0.3774512850676264, 0.367331396459356, 0.35909033372040494, 0.34987698096274716,
                  0.3422578351681336, 0.3336995606787801, 0.32650848020325807, 0.3186464040239277, 0.31193985786488065,
                  0.30452346987966417, 0.2981739290597798, 0.2910777847174428, 0.2843721860247948, 0.27706456718218947,
                  0.27156660916177267, 0.2641557164816707, 0.25884516731498636, 0.25224227397999555,
                  0.24751915398072322]
post_show_rate = post_show_rate[:MAX_DOC_LEN]
post_show_rate_tensor = tf.constant(post_show_rate, shape=[1, MAX_DOC_LEN])
post_show_rate_tensor = tf.tile(post_show_rate_tensor, [tf.shape(finerank_ctr)[0], 1])
post_show_rate_tensor_session = tf.reduce_sum(tf.where(seq_mask, post_show_rate_tensor, tf.zeros_like(show_head_pred)),
                                              axis=-1, keepdims=True)

finerank_ctr_head_pred_session = tf.reduce_sum(
    tf.where(seq_mask, post_show_rate_tensor * finerank_ctr, tf.zeros_like(show_head_pred)), axis=-1, keepdims=True)
finerank_cvr = get_dense('fc_ecom_predict_cvr_list') + get_dense('fc_ecom_predict_cart_cvr_list')
finerank_cvr_head_pred_session = tf.reduce_sum(
    tf.where(seq_mask, post_show_rate_tensor * finerank_ctr * finerank_cvr, tf.zeros_like(show_head_pred)), axis=-1,
    keepdims=True)

finerank_ctcvr = finerank_ctr * finerank_cvr

run_train.add_head(name='ground_truth_show_session_head',
                   prediction=post_show_rate_tensor_session,
                   label=show_label_session,
                   loss=0.0 * show_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='ground_truth_ctr_session_head',
                   prediction=finerank_ctr_head_pred_session,
                   label=ctr_label_session,
                   loss=0.0 * ctr_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='ground_truth_cvr_session_head',
                   prediction=finerank_cvr_head_pred_session,
                   label=cvr_label_session,
                   loss=0.0 * cvr_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='ground_truth_ctr_head',
                   prediction=finerank_ctr,
                   label=tf.where(tf.logical_and(seq_mask, show_dataset_mask), labels['CTR_LABEL'], invalid_label),
                   loss=0.0 * ctr_loss,
                   sample_rate=M.get_sample_rate())

run_train.add_head(name='ground_truth_cvr_head',
                   prediction=finerank_cvr,
                   label=tf.where(tf.logical_and(seq_mask, click_dataset_mask), labels['CVR_LABEL'], invalid_label),
                   loss=0.0 * cvr_loss,
                   sample_rate=M.get_sample_rate())

valid_ctr_label = tf.where(tf.logical_and(seq_mask, show_dataset_mask), labels['CTR_LABEL'], invalid_label)
valid_cvr_label = tf.where(tf.logical_and(seq_mask, click_dataset_mask), labels['CVR_LABEL'], invalid_label)

add_topk_auc_heads(ctr_head_pred, valid_ctr_label, run_train, prefix='ctr')
add_topk_auc_heads(cvr_head_pred, valid_cvr_label, run_train, prefix='cvr')
add_topk_auc_heads(finerank_ctr, valid_ctr_label, run_train, prefix='ground_truth_ctr')
add_topk_auc_heads(finerank_cvr, valid_cvr_label, run_train, prefix='ground_truth_cvr')


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
M.compile(default_hidden_layer_optimizer=optimizers.RMSPropV2(lr=.005, momentum=.99995, init_factor=0.015625,
                                                              lr_warmup_linear_steps=100000,
                                                              lr_warmup_linear_init=0.000001),
          run_steps=[run_train, run_predict],
          num_estimated_bias_features=10 * 1000 * 1000,
          num_estimated_vec_features=10 * 1000 * 1000)

model = print_model()
class TransformerEncoder(Module):

    def __init__(self, d_word_emb=16, num_head=4, mask=None, use_causal_mask=False, name=None):
        self.num_head = num_head
        self.name = name
        self.mask = mask
        self.use_causal_mask = use_causal_mask
        d_model = d_word_emb * num_head

        self.wq = layers.Dense(d_model, name=name + '/wq')
        self.wk = layers.Dense(d_model, name=name + '/wk')
        self.wv = layers.Dense(d_model, name=name + '/wv')
        # self.ffn = layers.Dense(d_model, name=name + '/ffn', activation=gelu)
        self.ffn_swiglu_w1 = layers.Dense(d_model, name=name + '/ffn_swiglu_w1')
        self.ffn_swiglu_w2 = layers.Dense(d_model, name=name + '/ffn_swiglu_w2')
        self.ffn_swiglu_w3 = layers.Dense(d_model, name=name + '/ffn_swiglu_w3')
        # self.ffn_two = layers.Dense(d_model, name=name + '/ffn_two')
        self.att_ln = LayerNorm(name=name + '/att_ln')
        self.ffn_ln = LayerNorm(name=name + '/ff_ln')

        if S.is_training():
            self.train = True
        else:
            self.train = False

    def call(self, x, drop_rate=0., mask=None, use_causal_mask=False):
        use_causal_mask = self.use_causal_mask
        mask = self.mask
        att = self._multihead_self_attention(x, x, x, mask, use_causal_mask, drop_rate)
        # print('-------- att', att)
        return self._ffn(att)

    def _multihead_self_attention(self, q, k, v, mask=None, use_causal_mask=False, drop_rate=0.):
        # linear projections: d_model > d_word_emb
        sq_ = self.wq(q)  # (N, T_q, d_model)
        sk_ = self.wk(k)  # (N, T_k, d_model)
        sv_ = self.wv(v)  # (N, T_k, d_model)
        # split and concat
        q_ = tf.concat(tf.split(sq_, self.num_head, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        k_ = tf.concat(tf.split(sk_, self.num_head, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        v_ = tf.concat(tf.split(sv_, self.num_head, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        # print('-------- q', q_)
        # scaled dot product attention
        output = self._scaled_dot_product_attention(q_, k_, v_, drop_rate, mask, use_causal_mask)  # (h*N, T_q, d_model/h)
        # print('-------- output1', output)
        # reshape
        output = tf.concat(tf.split(output, self.num_head, axis=0), axis=2)  # (N, T_q, d_model)
        # print('-------- output2', output)
        # residual
        output += sq_
        # normalization
        output = self.att_ln(output)  # (N, T_q, d_word_emb)
        return output

    def _scaled_dot_product_attention(self, q, k, v, drop_rate=0., mask=None, use_causal_mask=False):
        d_k = q.get_shape().as_list()[-1]
        # dot product
        att = tf.matmul(q, tf.transpose(k, [0, 2, 1]))  # (N, T_q, T_k), seq_len, seq_len
        # scale
        att /= d_k**0.5
        # print('------att', att)
        # print("------mask", mask)
        # mask
        # att = tf.Print(att, [att], message="tf print before att", summarize=1000, first_n=1000) print tensor example
        if mask is not None:
            att = self._mask(att, mask)
        if use_causal_mask:
            T_q = tf.shape(q)[1]
            T_k = tf.shape(k)[1]
            #更改casual
            causal = self._build_causal_mask(T_q, T_k, att.dtype)      # [T_q, T_k]
            causal = tf.expand_dims(causal, 0)                          # [1, T_q, T_k]
            causal = tf.tile(causal, [tf.shape(att)[0], 1, 1])         # [h*N, T_q, T_k]
            att = att - (1.0 - causal) * 1e4
        # softmax
        att = tf.nn.softmax(att)  # (N, T_q, T_k)
        tf.summary.histogram('transformer/attention', att)
        # dropout
        if self.train:
            att = tf.nn.dropout(att, rate=drop_rate)
        # weighted sum
        output = tf.matmul(att, v)  # (N, T_q, d_v)
        return output

    def _ffn(self, x):
        # inner layer
        # x_dense = self.ffn(x)  # (N, T_q, d_word_emb)
        # output = self.ffn_two(x_dense_activation)
        x_swiglu_1 = self.ffn_swiglu_w1(x)
        x_swiglu_1 = x_swiglu_1 * tf.sigmoid(x_swiglu_1)
        x_swiglu_2 = self.ffn_swiglu_w2(x)
        x_dense = x_swiglu_1 * x_swiglu_2
        output = self.ffn_swiglu_w3(x_dense)
        # print('-------- output 3', output)
        # print('-------- x', x)
        # redisual
        output += x
        # normalization
        output = self.ffn_ln(output)
        return output

    def _mask(self, x, mask):
        # x.shape = [h*N, T_q, T_k]
        # mask.shape = [N, T_k]
        mask = tf.tile(mask, [tf.shape(x)[0] // tf.shape(mask)[0], 1])  # [h*N, T_k]
        # print("------mask 1", mask)
        mask = tf.expand_dims(mask, 1)  # [h*N, 1, T_k]
        # x += mask * -1e4

        ### 0代表mask, 前序mask
        # length = x.get_shape().as_list()[1]  # T_q
        # mask_post_ele = tf.linalg.band_part(tf.ones((length, length)), -1, 0)  # 下三角矩阵 [T_q, T_q]
        # # mask_pre_ele = tf.linalg.band_part(tf.ones((length, length)), 0, -1)  # 上三角矩阵
        # mask_post_ele = tf.expand_dims(mask_post_ele, 0)  # [1, T_q, T_q]
        # # mask_post_ele = tf.Print(mask_post_ele, [mask_post_ele], message="tf print mask_post_ele", summarize=1000, first_n=1000)

        x = x - (1.0 - mask) * 10000.0
        # x = x - (1.0 - mask_post_ele) * 10000.0

        return x
    def _build_causal_mask(self, q_len, k_len, dtype):
        # [T_q, T_k] 下三角 1，其余 0
        # q_len/k_len 动态：直接用 tf.shape(...) 传进来
        i = tf.range(q_len)[:, None]      # [T_q, 1]
        j = tf.range(k_len)[None, :]      # [1, T_k]
        mask = tf.cast(i >= j, dtype)     # [T_q, T_k]
        return mask  # 1=允许，0=遮