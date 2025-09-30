#!/usr/bin/env python
# -*- coding=utf-8 -*-
from __future__ import absolute_import, division, print_function

import math
import numpy as np
from sail import initializers
from sail.layers.base_layer import Layer
from sail.layers import Dense, Gelu, LayerNorm
from sail.core import graph_transform
from sail import tf
import sail.common as S
from sail.initializers.base_initializer import Initializer


class PositionWiseFeedForward(Layer):
    """FeedForward Neural Networks for each position"""

    def __init__(
            self,
            hidden_size,
            dropout_p=0.0,
            hidden_multiplier=3,
            mixed_precision=False,
            **xargs
    ):
        super(PositionWiseFeedForward, self).__init__(**xargs)
        self.hidden_size = hidden_size
        # self.activ = lambda x: activ_fn(cfg.activ_fn, x)
        self.dropout_p = dropout_p
        self.hidden_multiplier = hidden_multiplier
        self.mixed_precision = mixed_precision
        self._init_call = False
        self.switch_gpu_optimize = S.is_gpu_training() and S.enable_optimize_mode(
            "GPU_OPTIMIZE"
        )

    def build(self, in_shape):
        self.fc1 = Dense(
            name="pwff_fc1",
            units=self.hidden_size * self.hidden_multiplier,
            kernel_initializer=initializers.GlorotNormal(),
            bias_initializer=initializers.RandomNormal(),
            mixed_precision=self.mixed_precision,
        )
        self.fc2 = Dense(
            name="pwff_fc2",
            units=self.hidden_size,
            kernel_initializer=initializers.GlorotNormal(),
            bias_initializer=initializers.RandomNormal(),
            mixed_precision=self.mixed_precision,
        )
        self.act = Gelu()

    def call(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        if self.switch_gpu_optimize:
            lego_modules = graph_transform.g_lego_module
            if not self._init_call:
                tmp_x = self.act(self.fc1(x))
                tmp_x = tf.nn.dropout(tmp_x, rate=self.dropout_p)

                tmp_x = self.fc2(tmp_x)
                tmp_x = tf.nn.dropout(tmp_x, rate=self.dropout_p)
                self._init_call = True

            lego_modules = graph_transform.g_lego_module
            x, bias_out, dropout_mask = lego_modules.linear_forward(
                input_tensor=x,
                weight=self.fc1.kernel,
                bias=self.fc1.bias,
                from_dim=x.shape.as_list()[-1],
                to_dim=self.fc1.units,
                act_gelu=True,
                dropout_rate=self.dropout_p,
            )

            x, bias_out, dropout_mask = lego_modules.linear_forward(
                input_tensor=x,
                weight=self.fc2.kernel,
                bias=self.fc2.bias,
                from_dim=x.shape.as_list()[-1],
                to_dim=self.fc2.units,
                act_gelu=False,
                dropout_rate=self.dropout_p,
            )
        else:
            x = self.act(self.fc1(x))
            x = tf.nn.dropout(x, rate=self.dropout_p)

            x = self.fc2(x)
            x = tf.nn.dropout(x, rate=self.dropout_p)

        return x  # Gelu


class VarianceScalingBatchMM(Initializer):
    def __init__(self, scale=1.0, mode="fan_avg", distribution="normal", seed=None):
        if scale <= 0.0:
            raise SailValueError("`scale` must be a positive float. Got:", scale)
        mode = mode.lower()
        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise SailValueError(
                "Invalid `mode` argument: "
                'expected on of {"fan_in", "fan_out", "fan_avg"}'
                "bug got",
                mode,
            )
        distribution = distribution.lower()
        if distribution not in {"normal", "uniform"}:
            raise SailValueError(
                "Invalid `distribution` argument: "
                'expected one of {"normal", "uniform"} '
                "but got",
                distribution,
            )

        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        # batchmm weights shape is (S, D, D')
        fan_in, fan_out = shape[-2], shape[-1]
        scale = self.scale
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, float(fan_in + fan_out) / 2)

        if self.distribution == "normal":
            # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(scale) / 0.87962566103423978
            return tf.truncated_normal(shape, 0.0, stddev, dtype=dtype, seed=self.seed)
        else:
            limit = np.sqrt(3.0 * scale)
            return tf.random_uniform(shape, -limit, limit, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {
            "scale": self.scale,
            "mode": self.mode,
            "distribution": self.distribution,
            "seed": self.seed,
        }


class BatchMatMulDense(Layer):
    def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(),
            bias_initializer=initializers.Zeros(),
            mixed_precision=False,
            optimizer=None,
            **xargs
    ):
        super(BatchMatMulDense, self).__init__(**xargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.no_clip = xargs.get("no_clip", False)

    def build(self, input_shape):
        self.token_num, self.token_dim = input_shape[0], input_shape[-1]
        # (S, D, D')
        kernel_shape = (self.token_num, self.token_dim, self.units)
        print('batchmm kernel:', kernel_shape)
        init_kernel = self.kernel_initializer(kernel_shape, self.dtype)
        self.kernel = self.add_weight(initial_value=init_kernel, name="kernel", no_clip=self.no_clip)
        self._snapshot_for_serving(self.kernel, "kernel")

        if self.use_bias:
            # (S, 1, D')
            bias_shape = (self.token_num, 1, self.units)
            init_bias = self.bias_initializer(bias_shape, self.dtype)
            self.bias = self.add_weight(initial_value=init_bias, name="bias", no_clip=self.no_clip)
            self._snapshot_for_serving(self.bias, "bias")
        else:
            self.bias = None

        if self.mixed_precision:
            self.kernel = tf.cast(self.kernel, tf.float16)
            if self.use_bias:
                self.bias = tf.cast(self.bias, tf.float16)

        self.built = True

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            # output = tf.nn.bias_add(output, self.bias)
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        self._register_for_debug("output_for_layer_{}".format(self.name), output)
        return output


class AdaptiveFFN(Layer):
    """ Adaptive FFN Layer """

    def __init__(self, hidden_size, dropout=0., mixed_precision=False, optimizer=None, hidden_emb_divide_ratio=None,
                 idx=0, enable_batch_matmul=False, **kwargs):
        super(AdaptiveFFN, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.hidden_emb_divide_flag = False if hidden_emb_divide_ratio is None else True
        self.hidden_emb_divide_ratio = 1 if hidden_emb_divide_ratio is None else hidden_emb_divide_ratio
        self.idx = idx
        self.enable_batch_matmul = enable_batch_matmul
        # logger.info("AdaptiveFFN init, optimizer=%s"%optimizer)

    def build(self, in_shape):
        if self.enable_batch_matmul:
            print("AdaptiveFFN enable_batch_matmul")
            self.fc1 = BatchMatMulDense(name="pwff_fc1_" + str(self.idx),
                                        units=self.hidden_size * 2 // self.hidden_emb_divide_ratio,
                                        kernel_initializer=VarianceScalingBatchMM(),
                                        bias_initializer=initializers.RandomNormal(),
                                        mixed_precision=self.mixed_precision,
                                        optimizer=self.optimizer)
            if not self.hidden_emb_divide_flag:
                self.fc2 = BatchMatMulDense(name="pwff_fc2_" + str(self.idx),
                                            units=self.hidden_size,
                                            kernel_initializer=VarianceScalingBatchMM(),
                                            bias_initializer=initializers.RandomNormal(),
                                            mixed_precision=self.mixed_precision,
                                            optimizer=self.optimizer)
        else:
            self.fc1 = Dense(name="pwff_fc1_" + str(self.idx),
                             units=self.hidden_size * 2 // self.hidden_emb_divide_ratio,
                             kernel_initializer=initializers.GlorotNormal(),
                             bias_initializer=initializers.RandomNormal(),
                             mixed_precision=self.mixed_precision,
                             optimizer=self.optimizer)
            if not self.hidden_emb_divide_flag:
                self.fc2 = Dense(name="pwff_fc2_" + str(self.idx),
                                 units=self.hidden_size,
                                 kernel_initializer=initializers.GlorotNormal(),
                                 bias_initializer=initializers.RandomNormal(),
                                 mixed_precision=self.mixed_precision,
                                 optimizer=self.optimizer)
        self.act = Gelu()

        last_dim = in_shape[-1]
        # self.token_num = in_shape[-2]

        if self.enable_batch_matmul:
            self.adapt1_tower1 = BatchMatMulDense(name="pwff_adapt_tower1_1_" + str(self.idx),
                                                  units=last_dim,
                                                  kernel_initializer=VarianceScalingBatchMM(),
                                                  bias_initializer=initializers.RandomNormal(),
                                                  activation=Gelu(),
                                                  mixed_precision=self.mixed_precision,
                                                  optimizer=self.optimizer)
            self.adapt1_tower2 = BatchMatMulDense(name="pwff_adapt_tower1_2_" + str(self.idx),
                                                  units=last_dim,
                                                  kernel_initializer=VarianceScalingBatchMM(),
                                                  bias_initializer=initializers.RandomNormal(),
                                                  activation=None,
                                                  mixed_precision=self.mixed_precision,
                                                  optimizer=self.optimizer)

            self.adapt2_tower1 = BatchMatMulDense(name="pwff_adapt_tower2_1_" + str(self.idx),
                                                  units=int(1.5 * self.hidden_size),
                                                  kernel_initializer=VarianceScalingBatchMM(),
                                                  bias_initializer=initializers.RandomNormal(),
                                                  activation=Gelu(),
                                                  mixed_precision=self.mixed_precision,
                                                  optimizer=self.optimizer)
            self.adapt2_tower2 = BatchMatMulDense(name="pwff_adapt_tower2_2_" + str(self.idx),
                                                  units=2 * self.hidden_size // self.hidden_emb_divide_ratio,
                                                  kernel_initializer=VarianceScalingBatchMM(),
                                                  bias_initializer=initializers.RandomNormal(),
                                                  activation=None,
                                                  mixed_precision=self.mixed_precision,
                                                  optimizer=self.optimizer)
        else:
            self.adapt1_tower1 = Dense(name="pwff_adapt_tower1_1_" + str(self.idx),
                                       units=last_dim,
                                       kernel_initializer=initializers.GlorotNormal(),
                                       bias_initializer=initializers.RandomNormal(),
                                       activation=Gelu(),
                                       mixed_precision=self.mixed_precision,
                                       optimizer=self.optimizer)
            self.adapt1_tower2 = Dense(name="pwff_adapt_tower1_2_" + str(self.idx),
                                       units=last_dim,
                                       kernel_initializer=initializers.GlorotNormal(),
                                       bias_initializer=initializers.RandomNormal(),
                                       activation=None,
                                       mixed_precision=self.mixed_precision,
                                       optimizer=self.optimizer)

            self.adapt2_tower1 = Dense(name="pwff_adapt_tower2_1_" + str(self.idx),
                                       units=int(1.5 * self.hidden_size),
                                       kernel_initializer=initializers.GlorotNormal(),
                                       bias_initializer=initializers.RandomNormal(),
                                       activation=Gelu(),
                                       mixed_precision=self.mixed_precision,
                                       optimizer=self.optimizer)
            self.adapt2_tower2 = Dense(name="pwff_adapt_tower2_2_" + str(self.idx),
                                       units=2 * self.hidden_size // self.hidden_emb_divide_ratio,
                                       kernel_initializer=initializers.GlorotNormal(),
                                       bias_initializer=initializers.RandomNormal(),
                                       activation=None,
                                       mixed_precision=self.mixed_precision,
                                       optimizer=self.optimizer)

    def _transfer(self, x, func1, func2):
        x = func1(x)
        x = func2(x)
        # if len(x.shape) < 3:
        #     x = tf.tile(
        #         tf.expand_dims(x, axis=1),
        #         [1, self.token_num, 1]
        #     )
        return x

    def call(self, x, adapt_emb=None):
        # compress x
        print('adapt_emb:', adapt_emb)
        a_in = adapt_emb if adapt_emb is not None else x
        print('a_in:', a_in)
        x_adapt1 = self._transfer(a_in, self.adapt1_tower1, self.adapt1_tower2)
        x_adapt2 = self._transfer(a_in, self.adapt2_tower1, self.adapt2_tower2)
        print('x_adapt1:', x_adapt1)
        print('x_adapt2:', x_adapt2)

        # do adaptive
        x = self.act(self.fc1(x * tf.tanh(x_adapt1)))
        x = tf.nn.dropout(x, rate=self.dropout)
        if not self.hidden_emb_divide_flag:
            x = self.fc2(x * tf.tanh(x_adapt2))
        else:
            x = x * tf.tanh(x_adapt2)
        x = tf.nn.dropout(x, rate=self.dropout)

        # summary
        tf.summary.histogram('adaptive_ffn/adapt1', x_adapt1)
        tf.summary.histogram('adaptive_ffn/adapt2', x_adapt2)
        return x


class PerTokensAFFN(Layer):
    """ PerTokens FFN Layer """

    def __init__(self, hidden_size, dropout=0., mixed_precision=False, optimizer=None, hidden_emb_divide_ratio=None,
                 idx=0, enable_batch_matmul=False, use_adaptiveffn=False, **kwargs):
        super(PerTokensAFFN, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.hidden_emb_divide_ratio = hidden_emb_divide_ratio
        self.idx = idx
        self.enable_batch_matmul = enable_batch_matmul
        self.use_adaptiveffn = use_adaptiveffn
        # logger.info("PerTokensAFFN init, optimizer=%s"%optimizer)
        print("use PerTokensAFFN!")

    def build(self, in_shape):
        # FC1: (E, D, 3D)
        # FC2: (E, 3D, D)
        if self.use_adaptiveffn:
            self.ffns = AdaptiveFFN(
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                mixed_precision=self.mixed_precision,
                optimizer=self.optimizer,
                hidden_emb_divide_ratio=self.hidden_emb_divide_ratio,
                idx=self.idx,
                enable_batch_matmul=True
            )
        else:
            self.ffns = PositionWiseFeedForward(
                hidden_size=self.hidden_size,
                dropout_p=self.dropout,
                mixed_precision=self.mixed_precision,
                optimizer=self.optimizer,
                hidden_emb_divide_ratio=self.hidden_emb_divide_ratio,
                enable_batch_matmul=True
            )

    def call(self, x, adapt_emb=None):
        # (B, T, D)  -> (T, B, D)
        x = tf.transpose(x, perm=[1, 0, 2])
        if adapt_emb is not None:
            adapt_emb = tf.transpose(adapt_emb, perm=[1, 0, 2])

        # (T, B, D) -> FC1 -> (T, B, 4D) -> FC2 -> (T, B, D)
        x = self.ffns(x, adapt_emb=adapt_emb)
        print("x ffns", x)

        # (T, B, D) -> (B, T, D)
        x = tf.transpose(x, perm=[1, 0, 2])
        print('combine tokens:', x)
        tf.summary.histogram('combiner_out', x)
        return x


class Attention(Layer):
    def __init__(
            self,
            embed_dim,
            num_heads,
            kernel_initializer=initializers.RandomNormal(),
            bias_initializer=initializers.Zeros(),
            drop_out_rate=0.0,
            use_mask=False,
            use_causal_mask=False,
            enable_adp_tao=False,
            enable_vis_score=False,
            mixed_precision=False,
            **kwargs
    ):
        super(Attention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # self.optimizer = optimizer
        self.kernel_initializer = kernel_initializer
        self.drop_out_rate = drop_out_rate
        self.use_mask = use_mask
        self.use_causal_mask = use_causal_mask
        self.enable_adp_tao = enable_adp_tao
        self.enable_vis_score = enable_vis_score
        self.bias_initializer = bias_initializer
        self.mixed_precision = mixed_precision
        self._init_call = False
        self.switch_gpu_optimize = S.is_gpu_training() and S.enable_optimize_mode(
            "GPU_OPTIMIZE"
        )

    def build(self, input_shape, **kwargs):
        assert len(input_shape) >= 3
        assert (
                self.embed_dim % self.num_heads == 0
        ), "Transformer d_model must be divisible by num_heads!"
        self.query_dense = Dense(
            self.embed_dim,
            name="query_dense",
            activation=None,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            mixed_precision=self.mixed_precision,
        )
        self.key_dense = Dense(
            self.embed_dim,
            name="key_dense",
            activation=None,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            mixed_precision=self.mixed_precision,
        )
        self.value_dense = Dense(
            self.embed_dim,
            name="value_dense",
            activation=None,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            mixed_precision=self.mixed_precision,
        )
        self.out_dense = Dense(
            self.embed_dim,
            name="combine_heads",
            activation=None,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            mixed_precision=self.mixed_precision,
        )

        if self.enable_adp_tao:
            kernel_initializer = initializers.TruncatedNormal(mean=1.0, stddev=0.0)
            self.tao = self.add_weight(
                name="tao", initial_value=kernel_initializer([1], dtype=tf.float32)
            )
            self._snapshot_for_serving(snapshot_tensor=self.tao, group_name="tao")

    def add_attn_summary(self, name, attn_w, unit_pixel=5):
        image_w = tf.reduce_mean(attn_w, axis=0, keepdims=True)
        image_w = tf.nn.sigmoid(image_w)
        image_w = tf.cast(image_w * 255, tf.uint8)
        image = tf.repeat(
            tf.repeat(image_w, repeats=unit_pixel, axis=1), repeats=unit_pixel, axis=2
        )
        image = tf.expand_dims(image, 3)
        tf.summary.image(name, image, max_outputs=2)

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.embed_dim // self.num_heads)
        )
        if self.switch_gpu_optimize:
            lego_modules = graph_transform.g_lego_module
            output = lego_modules.transpose_forward(x, 0)
        else:
            output = tf.transpose(x, perm=[0, 2, 1, 3])
        return output

    def attention(self, query, key, value, mask, use_mask=False, use_causal_mask=False):
        # b, n, s, s
        if self.switch_gpu_optimize:
            lego_modules = graph_transform.g_lego_module
            scale = 1 / float(math.sqrt(float(self.embed_dim // self.num_heads)))
            scaled_score = lego_modules.mat_mul_forward(
                input_a=query,
                input_b=key,
                transpose_a=False,
                transpose_b=True,
                scale=scale,
            )
        else:
            score = tf.matmul(query, key, transpose_b=True)
            tf.summary.histogram("qkT", score)
            # Note: the score template should be taken attention along with the divison
            scaled_score = score / tf.cast(
                tf.math.sqrt(float(self.embed_dim // self.num_heads)), score.dtype
            )

        if self.enable_adp_tao:
            scaled_score = scaled_score / tf.cast(self.tao, scaled_score.dtype)

        # mask
        if use_mask:
            mask = mask[:, None, None, :]
            scaled_score = scaled_score - (10000.0 * (1.0 - mask))
        if use_causal_mask:
            B = tf.shape(query)[0]
            H = self.num_heads
            T = tf.shape(query)[2]  # 注意 query shape: [B, H, T, D_head]

            # 构造下三角 mask: [T, T] → [1, 1, T, T] → [B, H, T, T]
            causal_mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
            causal_mask = tf.expand_dims(tf.expand_dims(causal_mask, 0), 0)
            mask = tf.tile(causal_mask, [B, H, 1, 1])  # shape: [B, H, T, T]

            scaled_score = scaled_score - (10000.0 * (1.0 - mask))
        if self.switch_gpu_optimize:
            lego_modules = graph_transform.g_lego_module
            weights = lego_modules.softmax_forward(
                scaled_score,
                mask if use_mask else 1,
                add_mask=use_mask,
                head_num=self.num_heads,
                apply_dropout=True,
                dropout_rate=self.drop_out_rate,
                batch_first=True,
            )
            weights = weights[1]
        else:
            weights = tf.nn.softmax(scaled_score, axis=-1)
            weights = tf.nn.dropout(weights, rate=self.drop_out_rate)

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        output = tf.matmul(weights, value)
        return output, weights

    def _do_attention(self, query, key, value, mask=None, need_weights=False):
        batch_size = tf.shape(query)[0]
        # proj
        if self.switch_gpu_optimize:
            lego_modules = graph_transform.g_lego_module
            if not self._init_call:
                self.query_dense(query)
                self.key_dense(key)
                self.value_dense(value)
                self._init_call = True
            query = lego_modules.linear_transpose_forward(
                input_tensor=query,
                weight=self.query_dense.kernel,
                bias=self.query_dense.bias,
                head_num=self.num_heads,
                from_dim=query.shape.as_list()[-1],
                to_dim=self.query_dense.units,
                transpose_type=0,
            )
            key = lego_modules.linear_transpose_forward(
                input_tensor=key,
                weight=self.key_dense.kernel,
                bias=self.key_dense.bias,
                head_num=self.num_heads,
                from_dim=key.shape.as_list()[-1],
                to_dim=self.key_dense.units,
                transpose_type=0,
            )
            value = lego_modules.linear_transpose_forward(
                input_tensor=value,
                weight=self.value_dense.kernel,
                bias=self.value_dense.bias,
                head_num=self.num_heads,
                from_dim=value.shape.as_list()[-1],
                to_dim=self.value_dense.units,
                transpose_type=0,
            )
        else:
            query = self.query_dense(query)
            key = self.key_dense(key)
            value = self.value_dense(value)
            # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
            query = self.separate_heads(query, batch_size)
            key = self.separate_heads(key, batch_size)
            value = self.separate_heads(value, batch_size)

        # (B, H, S, W) - Dense -> (B, H, S, S) -softmax-> (B, H, S, S)
        output, weights = self.attention(
            query, key, value, mask, use_mask=self.use_mask, use_causal_mask=self.use_causal_mask
        )
        if self.enable_vis_score:
            for i in list(range(self.num_heads)):
                weights_head = weights[:, i, :, :]
                self.add_attn_summary("attn_weights_head" + str(i), weights_head)
        if self.switch_gpu_optimize:
            lego_modules = graph_transform.g_lego_module
            output = lego_modules.transpose_forward(output, 0)
        else:
            output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_dim))
        output = self.out_dense(output)
        if need_weights:
            output = (output, weights)
        return output

    def call(self, x, k=None, mask=None, need_weights=False):
        if k is None:
            # do self attention
            output = self._do_attention(x, x, x, mask, need_weights)
        else:
            # do cross attention
            output = self._do_attention(x, k, k, mask, need_weights)
        return output


class ResidualAttentionBlock(Layer):
    def __init__(
            self,
            d_model,
            n_head,
            dropout=0.0,
            layernorm_eps=1.0e-5,
            use_mask=False,
            use_causal_mask=False,
            mixed_precision=False,
            attn_mask=None,
            enable_pre_norm=False,
            use_trick=False,
            mix_dim=3,
            hidden_emb_divide_ratio=None,
            optimizer=None,
            final_layer_output_cls=False,
            experts_num=None,
            use_adaptiveffn=False,
            **kwargs
    ):
        super(ResidualAttentionBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.use_mask = use_mask
        self.layernorm_eps = layernorm_eps
        self.mixed_precision = mixed_precision
        self.attn_mask = attn_mask
        self.use_causal_mask = use_causal_mask
        self.enable_pre_norm = enable_pre_norm
        self.use_trick = use_trick
        self.mix_dim = mix_dim
        self.hidden_emb_divide_ratio = hidden_emb_divide_ratio
        self.final_layer_output_cls = final_layer_output_cls
        self.experts_num = experts_num
        self.use_adaptiveffn = use_adaptiveffn
        self.optimizer = optimizer

    def build(self, in_shape):
        # attention
        self.attn = Attention(
            embed_dim=self.d_model,
            num_heads=self.n_head,
            drop_out_rate=self.dropout,
            use_mask=self.use_mask,
            use_causal_mask=self.use_causal_mask,
            enable_adp_tao=True,
            enable_vis_score=True,
            optimizer=self.optimizer,
            name="attn",
            mixed_precision=self.mixed_precision,
        )

        # in LayerNorm
        self.ln_1 = LayerNorm(
            name="ResidualAttentionBlock_in_norm",
            epsilon=self.layernorm_eps,
            mixed_precision=self.mixed_precision,
            scale_factor=10,
        )

        # FFN
        if self.hidden_emb_divide_ratio is not None:
            print("hidden_setting, mlp_name:", self.hidden_emb_divide_ratio, "PositionWiseFeedForward")
            self.mlp = PositionWiseFeedForward(hidden_size=self.d_model,
                                               dropout_p=self.dropout,
                                               name="pwff",
                                               optimizer=self.optimizer,
                                               mixed_precision=self.mixed_precision)
        else:
            print("hidden_setting, mlp_name:", self.hidden_emb_divide_ratio, "PerTokenAFFN")
            self.mlp = PerTokensAFFN(hidden_size=self.d_model,
                                     dropout=self.dropout, name="adapt_pwff",
                                     mixed_precision=self.mixed_precision,
                                     optimizer=self.optimizer,
                                     hidden_emb_divide_ratio=self.hidden_emb_divide_ratio,
                                     use_adaptiveffn=self.use_adaptiveffn,
                                     )

        # out LayerNorm
        self.ln_2 = LayerNorm(
            name="ResidualAttentionBlock_out_norm",
            epsilon=self.layernorm_eps,
            mixed_precision=self.mixed_precision,
            scale_factor=10,
        )

    def attention(self, x, k=None):
        if self.use_mask:
            attn_mask = tf.cast(self.attn_mask, tf.float32)
            return self.attn(x, k=k, mask=attn_mask)
        else:
            return self.attn(x, k=k)

    def mixup(self, inputs, new_tokens_num=8):
        _, token_nums, dims = inputs.get_shape().as_list()
        new_dims = dims * token_nums // new_tokens_num
        output = tf.reshape(inputs, [-1, token_nums, new_tokens_num, dims // new_tokens_num])
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [-1, new_tokens_num, new_dims])
        return output

    def mixup_4D(self, inputs, new_tokens_num=8):
        _, H_nums, token_nums, dims = inputs.get_shape().as_list()
        new_dims = dims * token_nums // new_tokens_num
        output = tf.reshape(inputs, [-1, H_nums, token_nums, new_tokens_num, dims // new_tokens_num])
        output = tf.transpose(output, [0, 1, 3, 2, 4])
        output = tf.reshape(output, [-1, H_nums, new_tokens_num, new_dims])
        return output

    def _call_preLN(self, x, k=None):
        if self.use_trick:
            if self.mix_dim == 4:
                attn_out = self.mixup_4D(self.ln_1(x), new_tokens_num=self.experts_num)
            else:
                attn_out = self.mixup(self.ln_1(x), new_tokens_num=self.experts_num)
        else:
            attn_out = self.attention(self.ln_1(x), k=k)
        tf.summary.histogram("attention_out", attn_out)
        # Note: different from fex, fex doesn't have dropout here
        x = x + tf.nn.dropout(attn_out, rate=self.dropout)
        mlp_out = self.mlp(self.ln_2(x))
        tf.summary.histogram("mlp_out", mlp_out)
        x = x + mlp_out

        return x

    def _call_postLN(self, x, k=None):
        if self.use_trick:
            if self.mix_dim == 4:
                attn_out = self.mixup_4D(x, new_tokens_num=self.experts_num)
            else:
                attn_out = self.mixup(x, new_tokens_num=self.experts_num)
        else:
            attn_out = self.attention(x, k=k)
        tf.summary.histogram("attention_out", attn_out)
        # Note: different from fex, fex doesn't have dropout here
        x = self.ln_1(x + tf.nn.dropout(attn_out, rate=self.dropout))
        mlp_out = self.mlp(x)
        tf.summary.histogram("mlp_out", mlp_out)
        x = self.ln_2(x + tf.nn.dropout(mlp_out, rate=self.dropout))

        return x

    def _call_wrapper(self, x, k=None):
        if self.enable_pre_norm:
            return self._call_preLN(x, k=k)
        else:
            return self._call_postLN(x, k=k)

    def call(self, x, k=None):
        # if self.use_trick:
        #     x = self.mixup()
        return self._call_wrapper(x, k=k)


class Transformer(Layer):
    def __init__(
            self,
            width,
            layers,
            heads,
            name,
            attn_mask=None,
            dropout=0.0,
            layernorm_eps=1.0e-5,
            use_mask=False,
            use_causal_mask=False,
            mixed_precision=False,
            enable_pre_norm=False,
            use_trick=False,
            mix_dim=3,
            experts_num=None,
            hidden_emb_divide_ratio=None,
            use_adaptiveffn=False,
            optimizers=None,
            **kwargs
    ):
        super(Transformer, self).__init__(**kwargs)
        self.width = width
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.resblocks = []
        self.layernorm_eps = layernorm_eps
        self.use_mask = use_mask
        self.name = name
        self.mixed_precision = mixed_precision
        self.attn_mask = attn_mask
        self.enable_pre_norm = enable_pre_norm
        self.use_causal_mask = use_causal_mask
        self.use_trick = use_trick
        self.mix_dim = mix_dim
        self.experts_num = experts_num
        self.hidden_emb_divide_ratio = hidden_emb_divide_ratio
        self.use_adaptiveffn = use_adaptiveffn
        self.optimizers = optimizers
        print("Transformer hidden_emb:", self.hidden_emb_divide_ratio)

    def build(self, in_shape):
        for i in list(range(self.layers)):
            self.resblocks.append(
                ResidualAttentionBlock(
                    self.width,
                    self.heads,
                    dropout=self.dropout,
                    layernorm_eps=self.layernorm_eps,
                    use_mask=self.use_mask,
                    use_causal_mask=self.use_causal_mask,
                    name="block.{}".format(i),
                    mixed_precision=self.mixed_precision,
                    attn_mask=self.attn_mask,
                    enable_pre_norm=self.enable_pre_norm,
                    use_trick=self.use_trick,
                    mix_dim=self.mix_dim,
                    experts_num=self.experts_num,
                    use_adaptiveffn=self.use_adaptiveffn,
                    optimizer=self.optimizers[i],
                    hidden_emb_divide_ratio=self.hidden_emb_divide_ratio,
                )
            )

    def call(self, x, k=None):
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, k=k)
            tf.summary.histogram("resblock_{}".format(i), x)
        return x
