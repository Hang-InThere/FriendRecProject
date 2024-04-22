import numpy as np
import scipy.sparse as sp
import torch
from dgl.nn.pytorch import GraphConv, SAGEConv, GATConv
from torch import nn
import torch.nn.functional as F
import world
from dgl.nn.pytorch import DeepWalk



class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self._init_weight()

    def _init_weight(self):
        self.num_users = self.dataset.n_users

        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['layer']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)


        nn.init.normal_(self.embedding_user.weight, std=0.1)

        self.f = nn.Sigmoid()
        self.SocialGraph = self.dataset.get_social_D_adj_D()
        self.gcn = GCN()
        self.sage = GraphSage()
        self.gat = GAT(self.latent_dim,self.latent_dim,self.latent_dim,1)


    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        embs = [users_emb]
        G = self.SocialGraph

        for layer in range(self.n_layers):
            users_emb = torch.sparse.mm(G, users_emb)
            embs.append(users_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users = light_out
        return users





class FriendRec(LightGCN):

    def _init_weight(self):
        super(FriendRec, self)._init_weight()
        self.num_items = self.dataset.n_items
        self.num_ratings = 6 #[0,1,2,3,4,5]
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_rating = torch.nn.Embedding(
            num_embeddings=self.num_ratings, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.normal_(self.embedding_rating.weight, std=0.1)
        self.ItemGraph = self.dataset.get_inter_D_adj_D()
        self.user_items = self.dataset.user_items
        self.rating = self.dataset.rating

        self.w_r1 = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.w_r2 = nn.Linear(self.latent_dim, self.latent_dim)
        self.Graph_Comb = Graph_Comb(self.latent_dim)




    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        rating_emb = self.embedding_rating.weight

        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        G = self.ItemGraph
        all_emb = torch.sparse.mm(G, all_emb)
        embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return self.final_user

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def getEmbedding(self, users, pos_u, neg_u):

        all_users = []
        social_user_emb = super().computer()
        all_users.append(social_user_emb)
        #graph = self.dataset.social_graph.to(world.device)
        #social_user_emb = self.sage(graph,self.embedding_user.weight)
        item_user_emb = self.computer()
        all_users.append(item_user_emb)
        all_users = torch.stack(all_users, dim=1)
        all_users = torch.mean(all_users, dim=1)

        #all_users = torch.cat([social_user_emb,item_user_emb],dim=1)
        #all_users = item_user_emb
        self.all_users_emb = all_users
        users_emb = all_users[users]
        pos_emb = all_users[pos_u]
        neg_emb = all_users[neg_u]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_user(pos_u)
        neg_emb_ego = self.embedding_user(neg_u)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getUsersDotUsers(self, users):
        all_users = self.all_users_emb
        users_emb_u = all_users[users.long()]
        items_emb_v = all_users
        dot = self.f(torch.matmul(users_emb_u, items_emb_v.t())).clone()
        return dot

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(64, 64,allow_zero_in_degree=True,weight=True)
        self.conv2 = GraphConv(64, 64,allow_zero_in_degree=True,weight=True)
        self.conv3 = GraphConv(64, 64,allow_zero_in_degree=True,weight=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = F.dropout(h,training=self.training)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = F.dropout(h,training=self.training)
        h = self.conv3(g, h)
        return h

class GraphSage(nn.Module):
    def __init__(self):
        super(GraphSage, self).__init__()
        self.sage1 = SAGEConv(64,64,'pool')
        self.sage2 = SAGEConv(64,64,'pool')
        self.sage3 = SAGEConv(64,64,'pool')

    def forward(self,g,in_feat):
        h = self.sage1(g, in_feat)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.sage2(g, h)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.sage3(g, h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads,activation=torch.tanh)
        self.conv1.set_allow_zero_in_degree(True)
        self.conv2 = GATConv(hidden_feats*num_heads, out_feats, num_heads,activation=torch.tanh)
        self.conv2.set_allow_zero_in_degree(True)
        self.conv3 = GATConv(hidden_feats*num_heads, out_feats, num_heads,activation=torch.tanh)
        self.conv3.set_allow_zero_in_degree(True)
        self.dropout = nn.Dropout(0.5)
    def forward(self, g, x):
        h = self.conv1(g, x)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv3(g,x)
        return h

class Graph_Comb(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))
        h2 = torch.tanh(self.att_y(y))
        output = self.comb(torch.cat((h1, h2), dim=1))
        output = output / output.norm(2)
        return output