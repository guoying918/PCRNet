
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .backbone_micd import MICD
from .metric import MD_distance
import numpy as np

EMBEND_DIM = 128 # 64, 96, 128, 256
NUM_LAYERS = 3
####################################################

class Query_inter(nn.Module):
    """
    input: (171,9,128)
    无参数，保留最显著类别信息
    """
    def __init__(self, in_channel):
        super(Query_inter, self).__init__()

    def forward(self, x):
        max_sim, _ = x.max(dim=1)
        return max_sim.mean(dim=-1, keepdim=True).unsqueeze(-1)  # (171, 1, 1)

class Class_inter(nn.Module):
    """
    input: (171,9,128)
    注意力权重计算：对每个类别生成权重
    """
    def __init__(self, in_channel):
        super(Class_inter, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_channel, in_channel // 4),  
            nn.ReLU(),
            nn.Linear(in_channel // 4, 1)   
        )
    def forward(self, x):
        attn_weights = F.softmax(self.attn(x), dim=1)  # (171, 9, 1)
        aggregated = (x * attn_weights).sum(dim=0, keepdim=True)  # (1, 9, 128)
        return aggregated.mean(dim=-1, keepdim=True)  # (1, 9, 1)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.125):
        super(FeedForward, self).__init__()
        hidden_dim = int(dim*ffn_expansion_factor)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CosineSimilarity(nn.Module):
    def __init__(self, class_nums, in_channel = EMBEND_DIM):
        super(CosineSimilarity, self).__init__()
        self.Nc = class_nums
        self.proj = nn.Sequential(
            nn.Linear(2*in_channel + 1, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, EMBEND_DIM),
        )

    def proto_compute(self, support_features, support_labels):
        C = support_features.shape[1] 
        prototypes = torch.zeros(self.Nc, C, device=support_features.device)  # (self.Nc, C)
        
        for c in range(self.Nc):
            mask = (support_labels == c)
            if mask.sum() > 0:
                prototypes[c] = support_features[mask].mean(dim=0)  # 计算均值mean得到类原型
        
        return prototypes  # (Nc, C) 

    def forward(self, support_features, support_labels, query_features):

        query_f = query_features.unsqueeze(1)  # (Nq, 1, C)
        support_proto = self.proto_compute(support_features, support_labels)  # (Nc, C)
        support_proto = support_proto.unsqueeze(0)  # (1, Nc, C)

        cos_similarity = MD_distance(support_features, support_labels, query_features)
        cos_similarity = cos_similarity.unsqueeze(-1)

        query_f_repeat = query_f.expand(-1, self.Nc, -1)
        support_proto_repeat = support_proto.expand(query_f_repeat.size(0), -1, -1)
        s_q_cat = torch.cat([query_f_repeat, support_proto_repeat], dim=-1) # (171,9,128*2)
        sim_cat = torch.cat([s_q_cat, cos_similarity], dim=-1) # (171,9,128*2+1)
        similarity = self.proj(sim_cat) # (171, 9, 128)

        return similarity, cos_similarity

class QueryWiseAttention(nn.Module):
    """
    input: (171, 9, 128)
    output: (171, 9, 128)
    """
    def __init__(self, class_nums, in_dim=EMBEND_DIM, num_heads=4, dropout=0.3): #2
        super(QueryWiseAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.ffn = FeedForward(in_dim)
    def forward(self, x):
        b, c, d = x.size() # (171,9,128)
        x_q = rearrange(x, "b c d -> c b d") # (9,171,128)
        x_q_norm = self.norm1(x_q)
        attn_output, attn_weights = self.attention(
            query=x_q_norm,
            key=x_q_norm,
            value=x_q_norm,
            need_weights=True
        ) # attn_weights: (9, 171, 171), attn_output:(9, 171, 128)
        x_1 = x_q + attn_output
        x_2 = x_1 + self.ffn(self.norm2(x_1))
        out = x_2.transpose(0,1)
        return out

class ClassWiseAttention(nn.Module):
    """
    input: (171,9,128)
    output: (171,9,128)
    """
    def __init__(self, num_classes, in_dim=EMBEND_DIM, num_heads=4, dropout=0.3):
        super(ClassWiseAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.ffn = FeedForward(in_dim)

    def forward(self, x, gms, gmt, domain):
        b, c, d = x.size() # (171,9,128)
        x_c = x
        if domain == 'source':
            global_class_matrix = gms
        elif domain == 'target':
            global_class_matrix = gmt
        else:
            return "Error! NO DOMAIN SET!"
        x_c_norm = self.norm1(x_c)
        attn_output, attn_weights = self.attention(
            query=x_c_norm,
            key=global_class_matrix,
            value=global_class_matrix,
            need_weights=True
        ) # attn_weights: (9, 9)
        x_1 = x_c_norm + attn_output
        x_2 = x_1 + self.ffn(self.norm2(x_1))

        return x_2

class Proj(nn.Module):
    def __init__(self, in_channel, drop = 0.0):
        super(Proj, self).__init__()
        self.lin1 = nn.Linear(in_channel, in_channel // 2)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(in_channel // 2, in_channel)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x_1 = self.lin1(x)
        x_2 = self.act(x_1)
        x_2 = self.drop(x_2)
        out = self.lin2(x_2)
        return out

class AttentionStack(nn.Module):
    def __init__(self, class_nums, in_dim = EMBEND_DIM, num_layers=NUM_LAYERS):
        super(AttentionStack, self).__init__()
        self.num_layers = num_layers
        self.query_layers = nn.ModuleList([QueryWiseAttention(class_nums) for _ in range(num_layers)])
        self.class_layers = nn.ModuleList([ClassWiseAttention(class_nums) for _ in range(num_layers)])
        self.linears = nn.ModuleList([Proj(in_dim) for _ in range(num_layers)])

        self.qi = Query_inter(in_dim)
        self.ci = nn.ModuleList([Class_inter(in_dim) for _ in range(num_layers)])

    def forward(self, x, gms, gmt, domain, mode, return_intermediate=True):
        assert self.num_layers >= 1
        intermediate_outputs = []
        # query 和 class 分别处理后叠加
        for i in range(self.num_layers):
            # shortskip
            temp = x
            #
            query_corre = self.query_layers[i](x)  # (N_q, N_c, C)
            class_corre = self.class_layers[i](x, gms, gmt, domain)  # (N_q, N_c, C)
            x_q = self.qi(query_corre)
            x_q_sig = torch.sigmoid(x_q)
            x_c = self.ci[i](class_corre)
            x_c_sig = torch.sigmoid(x_c)
            query_out = query_corre * x_c_sig
            class_out = class_corre * x_q_sig
            combined = query_out + class_out
            x = self.linears[i](combined) # add & mean
            x = x + temp
            #
            if return_intermediate:
                intermediate_outputs.append(x.clone().detach())
        out = x
        # return out
        if return_intermediate:
            return out, intermediate_outputs
        else:
            return out

class Mapping(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dim, out_dim, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x) # 100->128
        return x

class feature_encode(nn.Module):
    def __init__(self, class_nums, src_dim, tar_dim, in_dim = EMBEND_DIM):
        super(feature_encode, self).__init__()
        self.class_nums = class_nums
        self.in_dim = in_dim
        self.source_mapping = Mapping(src_dim, self.in_dim)
        self.target_mapping = Mapping(tar_dim, self.in_dim)
        self.backbone = MICD(self.in_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.in_dim, out_features=self.in_dim),
            nn.BatchNorm1d(self.in_dim),
        )
        self.gms = torch.nn.Parameter(
            torch.normal(0, 1e-1, size=[1, self.class_nums, self.in_dim], dtype=torch.float, device='cuda'),
            requires_grad=True)
        self.gmt = torch.nn.Parameter(
            torch.normal(0, 1e-1, size=[1, self.class_nums, self.in_dim], dtype=torch.float, device='cuda'),
            requires_grad=True)

        self.cosine_similarity = CosineSimilarity(class_nums)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.3),
            nn.Linear(self.in_dim // 4, 1, bias=False)
        )
        self.attention_stack = AttentionStack(class_nums)

    def forward(self, support, query, support_lables, domain, md = 'qandc', state = 'train'): # qtoc,ctoq,qandc, onlyc,onlyq
        """
        support: 
        query:
        """
        if domain == 'source':
            s_feature_map = self.source_mapping(support) # (45,128,9,9)
            q_feature_map = self.source_mapping(query) # (171,128,9,9)
        elif domain == 'target':
            s_feature_map = self.target_mapping(support) 
            q_feature_map = self.target_mapping(query)
        s_feature_ex = self.backbone(s_feature_map)
        q_feature_ex = self.backbone(q_feature_map)
        x_s = (torch.flatten(s_feature_ex, 2)).mean(dim = 2)
        x_q = (torch.flatten(q_feature_ex, 2)).mean(dim = 2)
        s_feature = self.fc(x_s) # (45, 128)
        q_feature = self.fc(x_q) # (171, 128)
        similarity_matrix, cos_similarity = self.cosine_similarity(s_feature, support_lables, q_feature) # torch.Size([N_q, N_c, C]) (171, 9, 128)
        out1, return_intermediate = self.attention_stack(similarity_matrix, self.gms, self.gmt, domain, md) # torch.Size([171, 9, 128])
        out = self.mlp(out1).squeeze(-1)

        # return out, rcrg_stage
        return out, return_intermediate[1]

# def get_parameter_number(net):
#     total_num = sum(p.numel() for p in net.parameters())
#     trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}

# from thop import profile
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     supports = torch.rand(45, 100, 9, 9).to(device)
#     supports = torch.rand(1, 100, 9, 9).to(device)
#     querys = torch.rand(171, 100, 9, 9).to(device)
#     support_labels = torch.randint(0, 9, (45,)).to(device)

#     feature_encoder = feature_encode(9, 100, 100).to(device)
#     print(get_parameter_number(feature_encoder))
#     flops, params = profile(feature_encoder, inputs=(supports, querys, support_labels,'source','train'))
#     print(f"FLOPs: {flops / 1e9} GFLOPs")