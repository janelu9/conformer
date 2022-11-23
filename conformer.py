import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import initializer as I
from align import Linear,LayerNorm,BatchNorm1D,Conv1D,Conv2D

class GLU(nn.Layer):
    def __init__(self, dim: int=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.glu(x, axis=self.dim)

def get_activation(act):
    activation_funcs = {
        "hardshrink": paddle.nn.Hardshrink,
        "hardswish": paddle.nn.Hardswish,
        "hardtanh": paddle.nn.Hardtanh,
        "tanh": paddle.nn.Tanh,
        "relu": paddle.nn.ReLU,
        "relu6": paddle.nn.ReLU6,
        "leakyrelu": paddle.nn.LeakyReLU,
        "selu": paddle.nn.SELU,
        "swish": paddle.nn.Swish,
        "gelu": paddle.nn.GELU,
        "glu": GLU,
        "elu": paddle.nn.ELU,
    }

    return activation_funcs[act]()


def make_non_pad_mask(lengths):
    return  paddle.arange(0, lengths.max().item()).unsqueeze(0)< lengths.unsqueeze(-1)
    
def subsequent_chunk_mask(size,chunk_size, num_left_chunks =-1 ):
    ret = paddle.zeros([size, size], dtype=paddle.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max(0, (i // chunk_size - num_left_chunks) * chunk_size)
        ending = min(size, (i // chunk_size + 1) * chunk_size)
        ret[i, start:ending] = True
    return ret
    
def add_optional_chunk_mask(xs,
                            masks,
                            use_dynamic_chunk,
                            use_dynamic_left_chunk,
                            decoding_chunk_size,
                            static_chunk_size,
                            num_decoding_left_chunks):
    if use_dynamic_chunk:
        max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            chunk_size = int(paddle.randint(1, max_len, (1, )))
            num_left_chunks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = int(
                        paddle.randint(0, max_left_chunks, (1, )))
        chunk_masks = subsequent_chunk_mask(xs.shape[1], chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks.logical_and(chunk_masks)  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.shape[1], static_chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks.logical_and(chunk_masks)  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks
    
class PositionalEncoding(nn.Layer):
    def __init__(self, emb_dim, dropout_rate,max_length,reverse=False):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        self.max_length = max_length
        self.xscale = emb_dim ** 0.5
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = paddle.zeros((1,max_length, emb_dim),paddle.float32)
        pos = paddle.arange(0, max_length, dtype=paddle.float32).unsqueeze(1)
        div = (-paddle.arange(0, emb_dim, 2,dtype=paddle.float32)/emb_dim * paddle.to_tensor(10000,paddle.float32).log()).exp()
        self.pe[:,:, 0::2] = paddle.sin(pos * div)
        self.pe[:,:, 1::2] = paddle.cos(pos * div)
            
    def forward(self, x,offset=0):
        pos_emb = self.pe[:, offset:offset + x.shape[1]]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)
        
    def position_encoding(self, offset, size):
        return self.dropout(self.pe[:, offset:offset + size])
        
class RelPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x,offset=0):
        x = x * self.xscale
        pos_emb = self.pe[:, offset:offset + x.shape[1]]
        return self.dropout(x), self.dropout(pos_emb)
        
class Conv2dSubsampling4(nn.Layer):
    def __init__(self,idim,odim,dropout_rate,pos_enc_class):
        super().__init__()
        self.pos_enc=pos_enc_class
        self.conv = nn.Sequential(
            Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            Conv2D(odim, odim, 3, 2),
            nn.ReLU(), )
        self.out = Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        self.subsampling_rate = 4
        self.right_context = 6
        self.least_len=(1*2+1)*2+1
    def forward(self, x, x_mask, offset=0):
        pad_len=self.least_len-x.shape[1]
        if pad_len>0:
            x=F.pad(x,[0,0,0,pad_len,0,0])
            x_mask=F.pad(x_mask.astype("int64"),[0,0,0,0,0,pad_len])>0
        x = x.unsqueeze(1)  
        x = self.conv(x)
        x = self.out(x.transpose([0, 2, 1, 3]).reshape([0, 0, -1]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]

class Conv2dSubsampling6(nn.Layer):
    def __init__(self,idim,odim,dropout_rate,pos_enc_class):
        super().__init__()
        self.pos_enc=pos_enc_class
        self.conv = nn.Sequential(
            Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            Conv2D(odim, odim, 5, 3),
            nn.ReLU(), )
        self.linear = Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        self.subsampling_rate = 6
        self.right_context = 10
        self.least_len=(1*3+2)*2+1

    def forward(self, x, x_mask, offset=0):
        pad_len=self.least_len-x.shape[1]
        if pad_len>0:
            x=F.pad(x,[0,0,0,pad_len,0,0])
            x_mask=F.pad(x_mask.astype("int64"),[0,0,0,0,0,pad_len])>0
        x = x.unsqueeze(1) 
        x = self.conv(x)
        x = self.linear(x.transpose([0, 2, 1, 3]).reshape([0, 0, -1]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]      

class Conv2dSubsampling8(nn.Layer):
    def __init__(self,idim,odim,dropout_rate,pos_enc_class):
        super().__init__()
        self.pos_enc=pos_enc_class
        self.conv = nn.Sequential(
            Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            Conv2D(odim, odim, 3, 2),
            nn.ReLU(),
            Conv2D(odim, odim, 3, 2),
            nn.ReLU())
        self.linear = Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2),odim)
        self.subsampling_rate = 8
        self.right_context = 14
        self.least_len=((1*2+1)*2+1)*2+1

    def forward(self, x, x_mask, offset=0):
        pad_len=self.least_len-x.shape[1]
        if pad_len>0:
            x=F.pad(x,[0,0,0,pad_len,0,0])
            x_mask=F.pad(x_mask.astype("int64"),[0,0,0,0,0,pad_len])>0
        x = x.unsqueeze(1) 
        x = self.conv(x)
        x = self.linear(x.transpose([0, 2, 1, 3]).reshape([0, 0, -1]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]

class MultiHeadedAttention(nn.Layer):
    def __init__(self,n_head: int, n_feat: int, dropatt: float,**kwargs):
        super().__init__()
        self.n_feat = n_feat
        self.n_head = n_head
        self.dropatt = nn.Dropout(dropatt, mode="upscale_in_train")
        self.d_head = n_feat // n_head
        assert self.d_head * n_head == self.n_feat, "n_feat must be divisible by n_head"
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        
        self.scale = self.d_head ** -0.5
    
    def compute_kv(self, key, value):
        k = paddle.transpose(paddle.reshape(self.linear_k(key), [0, 0, self.n_head, self.d_head]), [0, 2, 3, 1])
        v = paddle.transpose(paddle.reshape(self.linear_v(value), [0, 0, self.n_head, self.d_head]), [0, 2, 1, 3])
        return k, v    

    def attention(self,q,k,v,attn_mask=0):
        product = paddle.matmul(q,k) * self.scale + attn_mask
        weights = self.dropatt(F.softmax(product))
        out = paddle.reshape(paddle.transpose(paddle.matmul(weights,v),[0, 2, 1, 3]),[0, 0, -1])
        out = self.linear_out(out)
        return out
    
        
    def forward(self, query, key, value, attn_mask= 0, cache=None):
        q =paddle.transpose(paddle.reshape(self.linear_q(query), [0, 0, self.n_head, self.d_head]), [0, 2, 1, 3])  
        k,v = self.compute_kv(key,value)
        if cache is not None:
            k = paddle.concat([cache[0], k], 1)
            v = paddle.concat([cache[1], v], 1)
        new_cache = (k, v)
        out =self.attention(q,k,v,attn_mask)
        return out,new_cache
        
class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding."""

    def __init__(self, n_head, n_feat, dropout_rate):

        super().__init__(n_head, n_feat, dropout_rate)
        self.linear_pos = Linear(n_feat, n_feat, bias_attr=False)
        pos_bias_u = self.create_parameter(
            [self.n_head, self.d_head], default_initializer=I.XavierUniform())
        self.add_parameter('pos_bias_u', pos_bias_u)
        pos_bias_v = self.create_parameter(
            (self.n_head, self.d_head), default_initializer=I.XavierUniform())
        self.add_parameter('pos_bias_v', pos_bias_v)
        
    def _rel_shift(self, x, zero_triu=False):
        zero_pad = paddle.zeros((*x.shape[:3],1), dtype=x.dtype)
        x_padded = paddle.concat([zero_pad, x], axis=-1)
        x_padded = x_padded.reshape([0, 0,x.shape[-1]+1,-1])
        x = x_padded[:,:,1:].reshape(x.shape)
        if zero_triu:
            ones = paddle.ones((x.shape[2], x.shape[3]))
            x = x * paddle.tril(ones, x.shape[3] - x.shape[2])[None,None,:,:]
        return x
        
    def forward(self, query, key, value, pos_emb, attn_mask= 0, cache=None):
        q =paddle.transpose(paddle.reshape(self.linear_q(query), [0, 0, self.n_head, self.d_head]), [0, 2, 1, 3])  
        k,v = self.compute_kv(key,value)
        if cache is not None:
            k = paddle.concat([cache[0], k], 1)
            v = paddle.concat([cache[1], v], 1)
        new_cache = (k, v)
        p = self.linear_pos(pos_emb).reshape([0, 0, self.n_head, self.d_head]).transpose([0, 2, 3, 1])
        q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)
        product =  paddle.matmul(q_with_bias_u, k) + paddle.matmul(q_with_bias_v, p)
        product = product* self.scale + attn_mask
        weights = self.dropatt(F.softmax(product))
        out = paddle.reshape(paddle.transpose(paddle.matmul(weights,v),[0, 2, 1, 3]),[0, 0, -1])
        out = self.linear_out(out)
        return out,new_cache

class ConvolutionModule(nn.Layer):
    def __init__(self,
                 channels,
                 kernel_size=15,
                 activation=nn.ReLU(),
                 norm="batch_norm",
                 causal=False,
                 bias=True):

        super().__init__()
        self.pointwise_conv1 = Conv1D(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=None if bias else False)

        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        self.depthwise_conv = Conv1D(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias_attr=None if bias else False)

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = BatchNorm1D(channels)
        else:
            self.use_layer_norm = True
            self.norm = LayerNorm(channels)

        self.pointwise_conv2 = Conv1D(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=None if bias else False,)
        self.activation = activation

    def forward(self,x,mask_pad=None,cache=None):
        x = x.transpose([0, 2, 1]) 
        if mask_pad is not None: 
            x = x*mask_pad.astype("float32")
        if self.lorder > 0:
            if cache is None:
                x = F.pad(
                    x, [self.lorder, 0], 'constant', 0.0, data_format='NCL')
            else:
                assert cache.shape[0] == x.shape[0]  # B
                assert cache.shape[1] == x.shape[1]  # C
                x = paddle.concat((cache, x), axis=2)

            assert (x.shape[2] > self.lorder)
            new_cache = x[:, :, -self.lorder:]  #[B, C, T]
        else:

            new_cache = None

        x = self.pointwise_conv1(x)  
        x = F.glu(x, axis=1)  


        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose([0, 2, 1])  # [B, T, C]
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose([0, 2, 1])  # [B, C, T]
        x = self.pointwise_conv2(x)


        if mask_pad is not None:  
            x = x*mask_pad.astype("float32")
        x = x.transpose([0, 2, 1])  
        return x, new_cache

class PositionwiseFeedForward(nn.Layer):
    def __init__(self,
                 idim,
                 hidden_units,
                 dropout_rate,
                 activation=nn.ReLU()):
        super().__init__()
        self.w_1 = Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = Linear(hidden_units, idim)

    def forward(self, xs) :
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))

class ConformerEncoderLayer(nn.Layer):
    def __init__(
            self,
            size,
            self_attn,
            feed_forward=None,
            feed_forward_macaron=None,
            conv_module=None,
            dropout_rate=0.1,
            normalize_before=True,
            concat_after=False, ):
  
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size, epsilon=1e-12)  # for the FNN module
        self.norm_mha = LayerNorm(size, epsilon=1e-12)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size, epsilon=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(
                size, epsilon=1e-12)  # for the CNN module
            self.norm_final = LayerNorm(
                size, epsilon=1e-12)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(
            self,
            x,
            mask,
            pos_emb,
            mask_pad=None,
            att_cache=None,
            cnn_cache=None
    ):
        
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)


        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x_att, new_att_cache = self.self_attn(
            x, x, x, pos_emb, mask, cache=att_cache)

        if self.concat_after:
            x_concat = paddle.concat((x, x_att), axis=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)

        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = None
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)

            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache

class ConformerEncoder(nn.Layer):
    def __init__(self,
                 input_size,
                 output_size=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 input_layer="conv2d",
                 pos_enc_layer_type="rel_pos",
                 normalize_before=True,
                 concat_after=False,
                 static_chunk_size=0,
                 use_dynamic_chunk=False,
                 global_cmvn=None,
                 use_dynamic_left_chunk=False,
                 positionwise_conv_kernel_size=1,
                 macaron_style=True,
                 selfattention_layer_type="rel_selfattn",
                 activation_type="swish",
                 use_cnn_module=True,
                 cnn_module_kernel=15,
                 causal=False,
                 cnn_module_norm="batch_norm",
                 max_len=5000):
                 
        super().__init__()
        activation = get_activation(activation_type)
        
        self._output_size = output_size
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding


        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            idim=input_size,
            odim=output_size,
            dropout_rate=dropout_rate,
            pos_enc_class=pos_enc_class(
                output_size,
                positional_dropout_rate,
                max_len), )

        self.normalize_before = normalize_before
        self.after_norm = LayerNorm(output_size, epsilon=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        # self-attention module definition
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size,
                                       attention_dropout_rate)
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate,
                                   activation)
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        self.encoders = nn.LayerList([
            ConformerEncoderLayer(
                size=output_size,
                self_attn=encoder_selfattn_layer(*encoder_selfattn_layer_args),
                feed_forward=positionwise_layer(*positionwise_layer_args),
                feed_forward_macaron=positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                conv_module=convolution_layer(*convolution_layer_args)
                if use_cnn_module else None,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after) for _ in range(num_blocks)
        ])
        
    def output_size(self):
        return self._output_size

    def forward(
            self,
            xs,
            xs_lens,
            decoding_chunk_size=0,
            num_decoding_left_chunks=-1):

        masks = make_non_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, L)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks, offset=0)
        chunk_masks = add_optional_chunk_mask(
            xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk,
            decoding_chunk_size, self.static_chunk_size,
            num_decoding_left_chunks)
        chunk_masks=paddle.where(chunk_masks.unsqueeze(1),0.,-float('inf')).astype("float32")
        for layer in self.encoders:    
            xs, _, _, _ = layer(xs, chunk_masks, pos_emb, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks