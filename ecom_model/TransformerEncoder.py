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