#! -*- coding: utf-8 -*-

from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import initializers, activations

def to_mask(x, mask, mode='mul'):
    """通用mask函数
    这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
    """
    if mask is None:
        return x
    else:
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10


def extract_seq_patches(x, kernel_size, rate):
    """x.shape = [None, seq_len, seq_dim]
    滑动地把每个窗口的x取出来，为做局部attention作准备。
    """
    seq_dim = K.int_shape(x)[-1]
    seq_len = K.shape(x)[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    x = K.temporal_padding(x, (p_left, p_right))
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = K.concatenate(xs, 2)
    return K.reshape(x, (-1, seq_len, kernel_size, seq_dim))

def sequence_masking(x, mask, value=0.0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    else:
        if K.dtype(mask) != K.dtype(x):
            mask = K.cast(mask, K.dtype(x))
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = K.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        return x * mask + value * (1 - mask)

class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        for u in layer.updates:
            if not hasattr(self, '_updates'):
                self._updates = []
            if u not in self._updates:
                self._updates.append(u)
        return outputs

class MultiHeadAttention(Layer):
    """多头注意力机制
    """
    def __init__(
        self,
        heads,
        head_size,
        out_dim=None,
        key_size=None,
        use_bias=True,
        attention_scale=True,
        return_attention_scores=False,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs, mask=None, **kwargs):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        """
        q, k, v = inputs[:3]
        q_mask, v_mask = None, None
        if mask is not None:
            q_mask, v_mask = mask[0], mask[2]
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        qkv_inputs = [qw, kw, vw] + inputs[3:]
        qv_masks = [q_mask, v_mask]
        o, a = self.pay_attention_to(qkv_inputs, qv_masks, **kwargs)
        # 完成输出
        o = K.reshape(o, (-1, K.shape(o)[1], self.head_size * self.heads))
        o = self.o_dense(o)
        # 返回结果
        if self.return_attention_scores:
            return [o, a]
        else:
            return o

    def pay_attention_to(self, inputs, mask=None, **kwargs):
        """实现标准的乘性多头注意力
        a_bias: 对attention矩阵的bias。
                不同的attention bias对应不同的应用。
        p_bias: 在attention里的位置偏置。
                一般用来指定相对位置编码的种类。
        说明: 这里单独分离出pay_attention_to函数，是为了方便
              继承此类来定义不同形式的atttention；此处要求
              返回o.shape=(batch_size, seq_len, heads, head_size)。
        """
        (qw, kw, vw), n = inputs[:3], 3
        q_mask, v_mask = mask
        a_bias, p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        if a_bias:
            a_bias = inputs[n]
            n += 1
        if p_bias == 'rotary':
            cos_pos = K.repeat_elements(inputs[n][..., None, 1::2], 2, -1)
            sin_pos = K.repeat_elements(inputs[n][..., None, ::2], 2, -1)
            qw2 = K.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = K.reshape(qw2, K.shape(qw))
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = K.reshape(kw2, K.shape(kw))
            kw = kw * cos_pos + kw2 * sin_pos
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 处理位置编码
        if p_bias == 'typical_relative':
            position_bias = inputs[n]
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, position_bias)
        elif p_bias == 't5_relative':
            position_bias = K.permute_dimensions(inputs[n], (2, 0, 1))
            a = a + K.expand_dims(position_bias, 0)
        # Attention（续）
        if self.attention_scale:
            a = a / self.key_size**0.5
        if a_bias is not None:
            a = a + a_bias
        a = sequence_masking(a, v_mask, '-inf', -1)
        A = K.softmax(a)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', A, vw)
        if p_bias == 'typical_relative':
            o = o + tf.einsum('bhjk,jkd->bjhd', A, position_bias)
        return o, a

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        if self.return_attention_scores:
            a_shape = (
                input_shape[0][0], self.heads, input_shape[0][1],
                input_shape[1][1]
            )
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                return [mask[0], None]
            else:
                return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

class SelfAttention(OurLayer):
    """多头自注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)
        self.attention = Attention(
            self.heads,
            self.size_per_head,
            self.key_size,
            self.mask_right
        )
    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
            o = self.reuse(self.attention, [x, x, x, x_mask, x_mask])
        else:
            x = inputs
            o = self.reuse(self.attention, [x, x, x])
        return o
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)


class AtrousSelfAttention(OurLayer):
    """空洞多头自注意力机制
    说明：每个元素只跟相对距离为rate的倍数的元素有关联。
    """
    def __init__(self, heads, size_per_head, rate=1,
                 key_size=None, mask_right=False, **kwargs):
        super(AtrousSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.rate = rate
        self.mask_right = mask_right
    def build(self, input_shape):
        super(AtrousSelfAttention, self).build(input_shape)
        self.attention = Attention(
            self.heads,
            self.size_per_head,
            self.key_size,
            self.mask_right
        )
    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        seq_dim = K.int_shape(x)[-1]
        # 补足长度，保证可以reshape
        seq_len = K.shape(x)[1]
        pad_len = self.rate - seq_len % self.rate
        x = K.temporal_padding(x, (0, pad_len))
        if x_mask is not None:
            x_mask = K.temporal_padding(x_mask, (0, pad_len))
        new_seq_len = K.shape(x)[1]
        # 变换shape
        x = K.reshape(x, (-1, new_seq_len // self.rate, self.rate, seq_dim))
        x = K.permute_dimensions(x, (0, 2, 1, 3))
        x = K.reshape(x, (-1, new_seq_len // self.rate, seq_dim))
        if x_mask is not None:
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1))
            x_mask = K.permute_dimensions(x_mask, (0, 2, 1, 3))
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, 1))
        # 做attention
        if x_mask is not None:
            x = self.reuse(self.attention, [x, x, x, x_mask, x_mask])
        else:
            x = self.reuse(self.attention, [x, x, x])
        # 恢复shape
        x = K.reshape(x, (-1, self.rate, new_seq_len // self.rate, self.out_dim))
        x = K.permute_dimensions(x, (0, 2, 1, 3))
        x = K.reshape(x, (-1, new_seq_len, self.out_dim))
        x = x[:, : - pad_len]
        return x
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)


class LocalSelfAttention(OurLayer):
    """局部多头自注意力机制
    说明：每个元素只跟相对距离不超过neighbors的元素有关联，这里的rate
    是真正的膨胀率（跟膨胀卷积一样），如果不了解可以忽略，默认为1就好。
    """
    def __init__(self, heads, size_per_head, neighbors=1, rate=1,
                 key_size=None, mask_right=False, **kwargs):
        super(LocalSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.neighbors = neighbors
        self.rate = rate
        self.mask_right = mask_right
    def build(self, input_shape):
        super(LocalSelfAttention, self).build(input_shape)
        if self.mask_right:
            mask_right = np.ones((1, 1 + 2 * self.neighbors))
            mask_right[:, - self.neighbors : ] = 0
        else:
            mask_right = self.mask_right
        self.attention = Attention(
            self.heads,
            self.size_per_head,
            self.key_size,
            mask_right
        )
    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        # 提取局部特征
        kernel_size = 1 + 2 * self.neighbors
        xp = extract_seq_patches(x, kernel_size, self.rate)
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)
        # 变换shape
        seq_len = K.shape(x)[1]
        seq_dim = K.int_shape(x)[-1]
        x = K.reshape(x, (-1, 1, seq_dim))
        xp = K.reshape(xp, (-1, kernel_size, seq_dim))
        if x_mask is not None:
            xp_mask = K.reshape(xp_mask, (-1, kernel_size, 1))
        # 做attention
        if x_mask is not None:
            x = self.reuse(self.attention, [x, xp, xp, xp_mask])
        else:
            x = self.reuse(self.attention, [x, xp, xp])
        # 恢复shape
        x = K.reshape(x, (-1, seq_len, self.out_dim))
        x = to_mask(x, x_mask, 'mul')
        return x
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)


class SparseSelfAttention(OurLayer):
    """稀疏多头自注意力机制
    来自文章《Generating Long Sequences with Sparse Transformers》
    说明：每个元素只跟相对距离为rate的倍数的元素、以及相对距离不超过rate的元素有关联。
    """
    def __init__(self, heads, size_per_head, rate=2,
                 key_size=None, mask_right=False, **kwargs):
        super(SparseSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        assert rate != 1, u'if rate=1, please use SelfAttention directly'
        self.rate = rate
        self.neighbors = rate - 1
        self.mask_right = mask_right
    def build(self, input_shape):
        super(SparseSelfAttention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        seq_dim = K.int_shape(x)[-1]
        # 补足长度，保证可以reshape
        seq_len = K.shape(x)[1]
        pad_len = self.rate - seq_len % self.rate
        x = K.temporal_padding(x, (0, pad_len))
        if x_mask is not None:
            x_mask = K.temporal_padding(x_mask, (0, pad_len))
        new_seq_len = K.shape(x)[1]
        x = K.reshape(x, (-1, new_seq_len, seq_dim)) # 经过padding后shape可能变为None，所以重新声明一下shape
        # 线性变换
        qw = self.reuse(self.q_dense, x)
        kw = self.reuse(self.k_dense, x)
        vw = self.reuse(self.v_dense, x)
        # 提取局部特征
        kernel_size = 1 + 2 * self.neighbors
        kwp = extract_seq_patches(kw, kernel_size, self.rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, self.rate) # shape=[None, seq_len, kernel_size, out_dim]
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)
        # 形状变换
        qw = K.reshape(qw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.key_size))
        kw = K.reshape(kw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.key_size))
        vw = K.reshape(vw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.size_per_head))
        kwp = K.reshape(kwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.key_size))
        vwp = K.reshape(vwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.size_per_head))
        if x_mask is not None:
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1, 1))
            xp_mask = K.reshape(xp_mask, (-1, new_seq_len // self.rate, self.rate, kernel_size, 1, 1))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 3, 2, 1, 4)) # shape=[None, heads, r, seq_len // r, size]
        kw = K.permute_dimensions(kw, (0, 3, 2, 1, 4))
        vw = K.permute_dimensions(vw, (0, 3, 2, 1, 4))
        qwp = K.expand_dims(qw, 4)
        kwp = K.permute_dimensions(kwp, (0, 4, 2, 1, 3, 5)) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = K.permute_dimensions(vwp, (0, 4, 2, 1, 3, 5))
        if x_mask is not None:
            x_mask = K.permute_dimensions(x_mask, (0, 3, 2, 1, 4))
            xp_mask = K.permute_dimensions(xp_mask, (0, 4, 2, 1, 3, 5))
        # Attention1
        a = K.batch_dot(qw, kw, [4, 4]) / self.key_size**0.5
        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))
        a = to_mask(a, x_mask, 'add')
        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))
        if self.mask_right:
            ones = K.ones_like(a[: 1, : 1, : 1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        # Attention2
        ap = K.batch_dot(qwp, kwp, [5, 5]) / self.key_size**0.5
        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))
        if x_mask is not None:
            ap = to_mask(ap, xp_mask, 'add')
        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))
        if self.mask_right:
            mask = np.ones((1, kernel_size))
            mask[:, - self.neighbors : ] = 0
            mask = (1 - K.constant(mask)) * 1e10
            for _ in range(4):
                mask = K.expand_dims(mask, 0)
            ap = ap - mask
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = K.concatenate([a, ap], -1)
        A = K.softmax(A)
        a, ap = A[..., : K.shape(a)[-1]], A[..., K.shape(a)[-1] : ]
        # 完成输出1
        o1 = K.batch_dot(a, vw, [4, 3])
        # 完成输出2
        ap = K.expand_dims(ap, -2)
        o2 = K.batch_dot(ap, vwp, [5, 4])
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        o = to_mask(o, x_mask, 'mul')
        o = K.permute_dimensions(o, (0, 3, 2, 1, 4))
        o = K.reshape(o, (-1, new_seq_len, self.out_dim))
        o = o[:, : - pad_len]
        return o
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)


class TrainablePositionEmbedding(OurLayer):
    """定义位置Embedding，直接训练出来
    """
    def __init__(self, maxlen, v_dim,
                 merge_mode='add', **kwargs):
        super(TrainablePositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.v_dim = v_dim
        self.merge_mode = merge_mode
    def build(self, input_shape):
        super(TrainablePositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.maxlen, self.v_dim),
            initializer='zeros'
        )
    def call(self, inputs):
        """允许传入r（当前位置id）来得到相对位置向量
        """
        if isinstance(inputs, list):
            x, r = inputs
        else:
            x, r = inputs, 0
        pid = K.arange(K.shape(x)[1])
        pid = K.expand_dims(pid, 0)
        pid = K.tile(pid, [K.shape(x)[0], 1])
        pid = K.abs(pid - K.cast(r, 'int32'))
        pv = K.gather(self.embeddings, pid)
        if self.merge_mode == 'add':
            return pv + x
        else:
            return K.concatenate([x, pv])
    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape
        else:
            return (input_shape[0], input_shape[1], input_shape[2] + self.v_dim)


class SinCosPositionEmbedding(Layer):
    """Google提出来的Sin-Cos形式的位置Embedding
    """
    def __init__(self, v_dim,
                 merge_mode='add', **kwargs):
        super(SinCosPositionEmbedding, self).__init__(**kwargs)
        self.v_dim = v_dim
        self.merge_mode = merge_mode
    def call(self, inputs):
        """允许传入r（当前位置id）来得到相对位置向量
        """
        if isinstance(inputs, list):
            x, r = inputs
        else:
            x, r = inputs, 0
        pid = K.arange(K.shape(x)[1])
        pid = K.expand_dims(pid, 0)
        pid = K.tile(pid, [K.shape(x)[0], 1])
        pid = K.abs(pid - K.cast(r, 'int32'))
        pv = self.idx2pos(pid)
        if self.merge_mode == 'add':
            return pv + x
        else:
            return K.concatenate([x, pv])
    def idx2pos(self, pid):
        pid = K.cast(pid, 'float32')
        pid = K.expand_dims(pid, 2)
        pj = 1. / K.pow(10000., 2. / self.v_dim * K.arange(self.v_dim // 2, dtype='float32'))
        pj = K.expand_dims(pj, 0)
        pv = K.dot(pid, pj)
        pv1, pv2 = K.sin(pv), K.cos(pv)
        pv1, pv2 = K.expand_dims(pv1, 3), K.expand_dims(pv2, 3)
        pv = K.concatenate([pv1, pv2], 3)
        return K.reshape(pv, (K.shape(pv)[0], K.shape(pv)[1], self.v_dim))
    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:-1] + (input_shape[-1] + self.v_dim,)
        
class PositionEmbedding(Layer):
    """定义可训练的位置Embedding
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        hierarchical=None,
        embeddings_initializer='zeros',
        custom_position_ids=False,
        **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if 'int' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, 'int32')
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype='int32')[None]

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = K.gather(embeddings, position_ids // self.input_dim)
            embeddings_y = K.gather(embeddings, position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = K.gather(self.embeddings, position_ids)
            else:
                embeddings = self.embeddings[None, :seq_len]

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])
    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class SinusoidalPositionEmbedding(Layer):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self,
        output_dim,
        merge_mode='add',
        custom_position_ids=False,
        **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            seq_len = K.shape(inputs)[1]
            inputs, position_ids = inputs
            if 'float' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, K.floatx())
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]

        indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
        indices = K.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = tf.einsum('bn,d->bnd', position_ids, indices)
        embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
        embeddings = K.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class ZeroOnePositionEmbedding(Layer):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self,
        output_dim,
        scope,
        merge_mode='add',
        custom_position_ids=False,
        **kwargs
    ):
        super(ZeroOnePositionEmbedding, self).__init__(**kwargs)
        self.scope=scope
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            seq_len = K.shape(inputs)[1]
            inputs, position_ids = inputs
            if 'float' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, K.floatx())
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]

        mean=K.mean(K.abs(inputs))
        embeddings = K.ones_like(inputs)*mean
        #embeddings_1 = Lambda(lambda embeddings: embeddings[:int(seq_len/2)-self, :])(embeddings)
        #embeddings_2 = Lambda(lambda embeddings: embeddings[int(seq_len/2)-self:int(seq_len/2)+self, :])(embeddings)
        #embeddings_3 = Lambda(lambda embeddings: embeddings[int(seq_len/2)+self:, :])(embeddings)
        embeddings_1 = embeddings[:,:int(seq_len/2)-self.scope,:]*0
        embeddings_2 = embeddings[:,int(seq_len/2)-self.scope:int(seq_len/2)+self.scope,:]*0.5
        embeddings_3 = embeddings[:,int(seq_len/2)+self.scope:,:]
        embeddings = K.concatenate([embeddings_1, embeddings_2, embeddings_3], axis=1)

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(ZeroOnePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))