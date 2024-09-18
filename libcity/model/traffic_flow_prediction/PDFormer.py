import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class DataEmbedding(nn.Module):
    def __init__(
            self, input_window, feature_dim, embed_dim, lape_dim, adj_mx, drop=0.,
            add_time_in_day=False, add_day_in_week=False, device=torch.device('cpu'),
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)

        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)

        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lap_mx):   # 16, 12, 170, 9   9: flow day week  self.feature_dim = 1
        origin_x = x
        x_ts = x[..., :1].permute(0, 2, 3, 1).squeeze()  # 16, 170, 12
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        if self.add_time_in_day:
            daytime_embed = self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
            x += daytime_embed
        if self.add_day_in_week:
            weekday_embed = self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
            x += weekday_embed
        x += self.spatial_embedding(lap_mx)
        x = self.dropout(x)
        return x, x_ts, daytime_embed, weekday_embed


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :x.shape[2] - self.chomp_size, :].contiguous()


class GraphChannelAttentionLayer(nn.Module):

    def __init__(self, input_window, num_channel, weights=None, device=torch.device('cpu'), sem_delta=5):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_window, num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)
        self.device = device
        self.sem_delta = sem_delta
        if weights is not None:
            self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))

    def forward(self, graphs):  # batch, input_window, num_graph, n_node, n_node
        graphs = F.normalize(graphs, dim=-1, p=1)
        weights = F.softmax(self.weight, dim=1).to(self.device)  # dim =1 or 0  12 2 1 1
        # print(weights.squeeze())
        agg = torch.sum(graphs * weights, dim=2)  # 16, 12, 170, 170
        mask = torch.zeros_like(agg).to(self.device).bool()
        _, indices = torch.topk(agg, k=self.sem_delta, dim=-1, sorted=False)
        mask.scatter_(-1, indices, True)
        # mask.register_hook(lambda grad: print(grad))
        return F.normalize(mask * agg, dim=-1, p=1)


class GraphFusionLayer(nn.Module):
    def __init__(self, input_window, num_channel, device=torch.device('cpu'), sem_delta=5):
        super().__init__()
        self.input_window = input_window
        self.channel_attn = GraphChannelAttentionLayer(input_window=input_window,
                                                       num_channel=num_channel,
                                                       device=device, sem_delta=sem_delta)

    def forward(self, sim_st, period_matrix):  # 16, 12, 170, 170; 16, 12, 170, 170; 170, 170
        graphs = torch.stack([sim_st, period_matrix], dim=2)
        fused_graph = self.channel_attn(graphs)
        return fused_graph

class GraphMetricLearning(nn.Module):
    def __init__(self, input_window, in_dim, hid_dim, num_head, device=torch.device('cpu'), gl_drop=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_head = num_head
        self.weighted_sim = nn.ParameterList()
        self.input_window = input_window
        self.device = device
        self.proj = nn.Linear(in_dim, hid_dim)
        self.drop_out = nn.Dropout(gl_drop)

        for i in range(num_head):
            self.dim_weight = nn.Parameter(torch.FloatTensor(input_window, 1, hid_dim))
            nn.init.xavier_uniform_(self.dim_weight)
            self.weighted_sim.append(self.dim_weight)

    def forward(self, x):  # 16, 170, 12     12 * 1 * 12
        # need softmax ?
        x = self.drop_out(self.proj(x))
        x = x.unsqueeze(dim=1)
        batch_size = x.shape[0]
        num_node = x.shape[2]  # 170
        sim_mtx = torch.zeros((batch_size, self.input_window, num_node, num_node)).to(self.device)
        for i in range(self.num_head):
            x_ = x * self.weighted_sim[i]
            x_ = F.normalize(x_, dim=-1, p=2)
            sim_mtx += x_ @ x_.transpose(-2, -1)  # 16, 12, 170, 170
        sim_mtx /= self.num_head
        return sim_mtx


class STSelfAttention(nn.Module):
    def __init__(
            self, input_window, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2,
            qkv_bias=False, attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1, sem_delta=5,
            gl_drop=0.1, sem_clus_proj=None, geo_clus_proj=None
    ):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.sem_clus_proj = sem_clus_proj
        self.geo_clus_proj = geo_clus_proj
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio
        self.output_dim = output_dim
        self.sem_delta = sem_delta

        # hyper : dimensions of sem, geo, tem, period, metric_dim
        self.period_dim = 32  # 64
        self.metric_dim = 32
        self.metric_head_num = 2
        self.day_proj = nn.Linear(dim, self.period_dim // 2)
        self.week_proj = nn.Linear(dim, self.period_dim // 2)
        self.ts_metric = GraphMetricLearning(input_window, input_window, self.metric_dim, self.metric_head_num,
                                             device=device, gl_drop=gl_drop)
        self.period_metric = GraphMetricLearning(1, self.period_dim, self.metric_dim, self.metric_head_num,
                                                 device=device, gl_drop=gl_drop)
        self.fusion_layer = GraphFusionLayer(input_window, 2, device=device, sem_delta=sem_delta)

        self.geo_clus_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_clus_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_clus_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.scale_geo_q_proj = nn.Linear(2 * int(dim * self.geo_ratio), int(dim * self.geo_ratio))
        self.scale_geo_k_proj = nn.Linear(2 * int(dim * self.geo_ratio), int(dim * self.geo_ratio))
        self.scale_geo_v_proj = nn.Linear(2 * int(dim * self.geo_ratio), int(dim * self.geo_ratio))

        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        self.sem_clus_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_clus_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_clus_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.scale_sem_q_proj = nn.Linear(2 * int(dim * self.sem_ratio), int(dim * self.sem_ratio))
        self.scale_sem_k_proj = nn.Linear(2 * int(dim * self.sem_ratio), int(dim * self.sem_ratio))
        self.scale_sem_v_proj = nn.Linear(2 * int(dim * self.sem_ratio), int(dim * self.sem_ratio))

        self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)

        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_ts, daytime_embed, weekday_embed, x_patterns, pattern_keys, geo_mask=None):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)

        geo_clus_x = torch.stack([torch.mean(x[..., v, :], dim=-2) for k, v in self.geo_clus_proj['c2n'].items()], dim=-2)
        geo_clus_q = self.geo_clus_q_conv(geo_clus_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_clus_k = self.geo_clus_k_conv(geo_clus_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_clus_v = self.geo_clus_v_conv(geo_clus_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        geo_q = self.scale_geo_q_proj(torch.cat([geo_clus_q[..., self.geo_clus_proj['n2c'], :], geo_q], dim=-1))
        geo_k = self.scale_geo_k_proj(torch.cat([geo_clus_k[..., self.geo_clus_proj['n2c'], :], geo_k], dim=-1))
        geo_v = self.scale_geo_v_proj(torch.cat([geo_clus_v[..., self.geo_clus_proj['n2c'], :], geo_v], dim=-1))

        geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        if geo_mask is not None:
            geo_attn.masked_fill_(geo_mask, float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio))

        '''
            pooling x base the sem_clusters to generate embeddings from different scales.
            x: 16 12 170 64
            sem_clus_x: 16 12 sem_clus_num 64
        '''
        sem_clus_x = torch.stack([torch.mean(x[..., v, :], dim=-2) for k, v in self.sem_clus_proj['c2n'].items()], dim=-2)
        sem_clus_q = self.sem_clus_q_conv(sem_clus_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_clus_k = self.sem_clus_k_conv(sem_clus_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_clus_v = self.sem_clus_v_conv(sem_clus_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        '''
            sem q k v concatenate 
        '''
        sem_q = self.scale_sem_q_proj(torch.cat([sem_clus_q[..., self.sem_clus_proj['n2c'], :], sem_q], dim=-1))
        sem_k = self.scale_sem_k_proj(torch.cat([sem_clus_k[..., self.sem_clus_proj['n2c'], :], sem_k], dim=-1))
        sem_v = self.scale_sem_v_proj(torch.cat([sem_clus_v[..., self.sem_clus_proj['n2c'], :], sem_v], dim=-1))

        sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale  # 16 12 2 170 170

        '''
            x_ts: time series vector for each node, to capture the short range relation. shape: 16, 170, 12
        '''
        sim_ts = self.ts_metric(x_ts)  # 16 12 170 170   # need negative weights?
        day_proj = self.day_proj(daytime_embed)
        week_proj = self.week_proj(weekday_embed)
        time_proj = torch.concatenate([day_proj, week_proj], dim=-1)[:, :, 0, :]  # 16 12 64
        # sim_time = self.period_metric(time_proj).squeeze()  # 16 12 12
        sim_time = F.normalize(self.period_metric(time_proj).squeeze(), dim=-1, p=1)
        # 16 12 12 1 1 * 16 1 12 170 170 sum-> 16 12 170 170
        period_matrix = (sim_time.unsqueeze(3).unsqueeze(4) * sim_ts.unsqueeze(1)).sum(2)
        final_mask = self.fusion_layer(sim_ts, period_matrix).unsqueeze(2)  # 16 12 1 170 170

        # sem_attn.masked_fill_(final_mask, float('-inf'))
        sem_attn = sem_attn * final_mask
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, int(D * self.sem_ratio))

        x = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1))
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalSelfAttention(nn.Module):
    def __init__(
            self, dim, dim_out, t_attn_size, t_num_heads=6, qkv_bias=False,
            attn_drop=0., proj_drop=0., device=torch.device('cpu'),
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.t_attn_size = t_attn_size

        self.t_q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale

        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)

        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        x = self.proj(t_x)
        x = self.proj_drop(x)
        return x


class STEncoderBlock(nn.Module):

    def __init__(
            self, input_window, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="pre",
            output_dim=1, sem_delta=5, gl_drop=0.1, sem_clus_proj=None, geo_clus_proj=None
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(input_window,
                                       dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads,
                                       sem_num_heads=sem_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim,
                                       sem_delta=sem_delta, gl_drop=gl_drop, sem_clus_proj=sem_clus_proj,
                                       geo_clus_proj=geo_clus_proj
                                       )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_ts, daytime_embed, weekday_embed, x_patterns, pattern_keys, geo_mask=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(
                self.st_attn(self.norm1(x), x_ts, daytime_embed, weekday_embed, x_patterns, pattern_keys,
                             geo_mask=geo_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(
                self.st_attn(x, x_ts, x_patterns, pattern_keys, geo_mask=geo_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class PDFormer(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.ext_dim = self.data_feature.get("ext_dim", 0)
        self.num_batches = self.data_feature.get('num_batches', 1)
        self.sem_clus_num = self.data_feature.get('sem_clus_num')
        self.sem_clus_proj = self.data_feature.get('sem_clus_proj')
        self.geo_clus_proj = self.data_feature.get('geo_clus_proj')
        self.adj_mx = data_feature.get('adj_mx')
        sd_mx = data_feature.get('sd_mx')
        sh_mx = data_feature.get('sh_mx')
        self._logger = getLogger()
        self.dataset = config.get('dataset')

        self.embed_dim = config.get('embed_dim', 64)
        self.skip_dim = config.get("skip_dim", 256)
        lape_dim = config.get('lape_dim', 8)
        geo_num_heads = config.get('geo_num_heads', 4)
        sem_num_heads = config.get('sem_num_heads', 2)
        t_num_heads = config.get('t_num_heads', 2)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.)
        attn_drop = config.get("attn_drop", 0.)
        drop_path = config.get("drop_path", 0.3)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.t_attn_size = config.get("t_attn_size", 3)
        enc_depth = config.get("enc_depth", 6)
        type_ln = config.get("type_ln", "pre")
        self.type_short_path = config.get("type_short_path", "hop")
        self.gl_drop = config.get("gl_drop", 0.1)

        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)
        self.device = config.get('device', torch.device('cpu'))
        self.world_size = config.get('world_size', 1)
        self.huber_delta = config.get('huber_delta', 1)
        self.quan_delta = config.get('quan_delta', 0.25)
        self.far_mask_delta = config.get('far_mask_delta', 5)
        self.sem_mask_delta = config.get('sem_mask_delta', 5)

        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)
        self.max_epoch = config.get('max_epoch', 200)
        self.task_level = config.get('task_level', 0)
        if self.max_epoch * self.num_batches * self.world_size < self.step_size * self.output_window:
            self._logger.warning('Parameter `step_size` is too big with {} epochs and '
                                 'the model cannot be trained for all time steps.'.format(self.max_epoch))
        if self.use_curriculum_learning:
            self._logger.info('Use use_curriculum_learning!')

        c2n, n2c = self.sem_clus_proj['c2n'], self.sem_clus_proj['n2c']
        c2n = {k: torch.LongTensor(v).to(self.device) for k, v in c2n.items()}
        n2c = torch.LongTensor(n2c).to(self.device)
        self.sem_clus_proj = {'c2n': c2n,
                              'n2c': n2c}

        c2n, n2c = self.geo_clus_proj['c2n'], self.geo_clus_proj['n2c']
        c2n = {k: torch.LongTensor(v).to(self.device) for k, v in c2n.items()}
        n2c = torch.LongTensor(n2c).to(self.device)
        self.geo_clus_proj = {'c2n': c2n,
                              'n2c': n2c}

        if self.type_short_path == "dist":
            distances = sd_mx[~np.isinf(sd_mx)].flatten()
            std = distances.std()
            sd_mx = np.exp(-np.square(sd_mx / std))
            self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.far_mask[sd_mx < self.far_mask_delta] = 1
            self.far_mask = self.far_mask.bool()
        else:
            sh_mx = sh_mx.T
            self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.geo_mask[sh_mx >= self.far_mask_delta] = 1
            self.geo_mask = self.geo_mask.bool()

        self.pattern_keys = torch.from_numpy(data_feature.get('pattern_keys')).float().to(self.device)
        self.pattern_embeddings = nn.ModuleList([
            TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)
        ])

        self.enc_embed_layer = DataEmbedding(self.input_window,
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx, drop=drop,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
        )

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(input_window=self.input_window,
                           dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size,
                           geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads,
                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=enc_dpr[i],
                           act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device,
                           type_ln=type_ln, output_dim=self.output_dim, sem_delta=self.sem_mask_delta,
                           gl_drop=self.gl_drop, sem_clus_proj=self.sem_clus_proj, geo_clus_proj=self.geo_clus_proj
                           ) for i in range(enc_depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )

    def forward(self, batch, lap_mx=None):
        x = batch['X']     # 16 12 170 9
        T = x.shape[1]
        x_pattern_list = []
        for i in range(self.s_attn_size):
            x_pattern = F.pad(
                x[:, :T + i + 1 - self.s_attn_size, :, :self.output_dim],
                (0, 0, 0, 0, self.s_attn_size - 1 - i, 0),
                "constant", 0,
            ).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)  # (B, T, N, s_attn_size, output_dim)

        x_pattern_list = []
        pattern_key_list = []
        for i in range(self.output_dim):
            x_pattern_list.append(self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1))
            pattern_key_list.append(self.pattern_embeddings[i](self.pattern_keys[..., i]).unsqueeze(-1))
        x_patterns = torch.cat(x_pattern_list, dim=-1)
        pattern_keys = torch.cat(pattern_key_list, dim=-1)

        enc, x_ts, daytime_embed, weekday_embed = self.enc_embed_layer(x, lap_mx)

        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, x_ts, daytime_embed, weekday_embed, x_patterns, pattern_keys, self.geo_mask)
            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1))

        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        return skip.permute(0, 3, 2, 1)

    def get_loss_func(self, set_loss):
        if set_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                    'masked_mse', 'masked_rmse', 'masked_mape', 'masked_huber', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        if set_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif set_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif set_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif set_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        elif set_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif set_loss.lower() == 'huber':
            lf = partial(loss.huber_loss, delta=self.huber_delta)
        elif set_loss.lower() == 'quantile':
            lf = partial(loss.quantile_loss, delta=self.quan_delta)
        elif set_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif set_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif set_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif set_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        elif set_loss.lower() == 'masked_huber':
            lf = partial(loss.masked_huber_loss, delta=self.huber_delta, null_val=0)
        elif set_loss.lower() == 'r2':
            lf = loss.r2_score_torch
        elif set_loss.lower() == 'evar':
            lf = loss.explained_variance_score_torch
        else:
            lf = loss.masked_mae_torch
        return lf

    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None, set_loss='masked_mae'):
        lf = self.get_loss_func(set_loss=set_loss)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        if self.training:
            if batches_seen % self.step_size == 0 and self.task_level < self.output_window:
                self.task_level += 1
                self._logger.info('Training: task_level increase from {} to {}'.format(
                    self.task_level - 1, self.task_level))
                self._logger.info('Current batches_seen is {}'.format(batches_seen))
            if self.use_curriculum_learning:
                return lf(y_predicted[:, :self.task_level, :, :], y_true[:, :self.task_level, :, :])
            else:
                return lf(y_predicted, y_true)
        else:
            return lf(y_predicted, y_true)

    def calculate_loss(self, batch, batches_seen=None, lap_mx=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, lap_mx)
        return self.calculate_loss_without_predict(y_true, y_predicted, batches_seen)

    def predict(self, batch, lap_mx=None):
        return self.forward(batch, lap_mx)