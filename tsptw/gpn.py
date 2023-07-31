import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        v = torch.FloatTensor(n_hidden).cuda()
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)

    def forward(self, q, ref):       # query and reference
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)
        q = self.Wq(q)     # (B, dim)
        ref = self.Wref(ref)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)
        
        q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size, dim)
        # v_view: (B, dim, 1)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
        
        # (B, size, dim) * (B, dim, 1)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
        
        return u, ref


class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        
        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whi = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wci = nn.Linear(n_hidden, n_hidden)    # w(ct)
        # self.Wei = nn.Linear(n_hidden, n_hidden)
        
        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)    # w(ct)
        # self.Wef = nn.Linear(n_hidden, n_hidden)
        
        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)    # W(ht)
        # self.Wec = nn.Linear(n_hidden, n_hidden)
        
        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)    # w(ct)
        # self.Weo = nn.Linear(n_hidden, n_hidden)
    
    def forward(self, x, h, c, xe=None):       # query and reference
        if xe is not None:
            # input gate
            i = torch.sigmoid(self.Wxi(x) + self.Wei(xe) + self.Whi(h) + self.wci(c))
            # forget gate
            f = torch.sigmoid(self.Wxf(x) + self.Wef(xe) + self.Whf(h) + self.wcf(c))
            # cell gate
            c = f * c + i * torch.tanh(self.Wxc(x) + self.Wec(xe) + self.Whc(h))
            # output gate
            o = torch.sigmoid(self.Wxo(x) + self.Weo(xe) + self.Who(h) + self.wco(c))

            h = o * torch.tanh(c)
        else:
            # input gate
            i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
            # forget gate
            f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
            # cell gate
            c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
            # output gate
            o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))

            h = o * torch.tanh(c)
        
        return h, c


class GPN(torch.nn.Module):
    
    def __init__(self, n_feature, n_hidden):
        super(GPN, self).__init__()
        self.city_size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        # lstm for first turn
        self.lstm0 = nn.LSTM(n_hidden, n_hidden)
        self.lstm0_cell = nn.LSTMCell(n_hidden, n_hidden)
        
        # pointer layer
        self.pointer = Attention(n_hidden)
        
        # lstm encoder
        self.encoder = LSTM(n_hidden)
        
        # trainable first hidden input
        h0 = torch.FloatTensor(n_hidden).cuda()
        c0 = torch.FloatTensor(n_hidden).cuda()
        
        # trainable latent variable coefficient
        alpha = torch.ones(1).cuda()
        
        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)
        
        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        self.c0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)
        
        # embedding
        self.embedding_x = nn.Linear(n_feature, n_hidden)
        # self.embedding_xe = nn.Linear(n_feature, n_hidden)
        self.embedding_all = nn.Linear(n_feature, n_hidden)
        
        
        # weights for GNN
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)
        
        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x, X_all, mask, h=None, c=None, latent=None, xe=None):
        """
        Inputs (B: batch size, size: city size, dim: hidden dimension)

        x: current city coordinate (B, 2)
        X_all: all cities' cooridnates (B, size, 2)
        mask: mask visited cities
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent: latent pointer vector from previous layer (B, size, dim)

        Outputs

        softmax: probability distribution of next city (B, size)
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent_u: latent pointer vector for next layer
        """
        
        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)
        
        
        # =============================
        # vector context
        # =============================
        
        # x_expand = x.unsqueeze(1).repeat(1, self.city_size, 1)   # (B, size)
        # X_all = X_all - x_expand
        
        # the weights share across all the cities
        x = self.embedding_x(x)
        if xe is not None:
            xe = self.embedding_xe(xe)
        context = self.embedding_all(X_all)  # [B, size, embed_dim]
        context *= X_all.sum(2).bool()[..., None].repeat(1, 1, self.dim)

        city_size_true = torch.zeros((self.batch_size, 1)).cuda()
        for s in range(self.city_size):
            # X_all: [bs, size, 4]
            not_zero = X_all.sum(2)[:, s:s + 1].bool()
            city_size_true = city_size_true + not_zero.float()
        
        # =============================
        # process hidden variable
        # =============================
        
        first_turn = False
        if h is None or c is None:
            first_turn = True
        
        if first_turn:
            # (dim) -> (B, dim)
            
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()
            
            input_context = context.permute(1,0,2).contiguous()  # [size, bs, embed_dim]

            h_enc, c_enc = h0.squeeze(0), c0.squeeze(0)
            for s in range(self.city_size):
                _h_enc, _c_enc = self.lstm0_cell(input_context[s, :, :], (h_enc, c_enc))
                # X_all: [bs, size, 4]
                not_zero = X_all.sum(2)[:, s:s+1].bool()
                h_enc = (not_zero) * _h_enc + (~not_zero) * h_enc
                c_enc = (not_zero) * _c_enc + (~not_zero) * c_enc
            #
            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)
            c = c_enc.squeeze(0)

        # =============================
        # graph neural network encoder
        # =============================

        city_size_true = city_size_true[:, :, None].repeat(1, self.city_size, self.dim)
        city_size_true = city_size_true.view(-1, self.dim)
        # (B, size, dim)
        context = context.view(-1, self.dim)
        not_zero = X_all.sum(2).bool()[..., None].repeat(1, 1, self.dim).view(-1, self.dim)
        
        context = self.r1 * self.W1(context * not_zero)\
            + (1-self.r1) * F.relu(self.agg_1(context * not_zero/(city_size_true-1)))
        context *= not_zero
        context = self.r2 * self.W2(context * not_zero)\
            + (1-self.r2) * F.relu(self.agg_2(context * not_zero/(city_size_true-1)))
        context *= not_zero
        context = self.r3 * self.W3(context * not_zero)\
            + (1-self.r3) * F.relu(self.agg_3(context * not_zero/(city_size_true-1)))
        context *= not_zero
        
        # LSTM encoder
        h, c = self.encoder(x, h, c, xe)
        # query vector
        q = h
        # pointer
        u, _ = self.pointer(q, context)
        latent_u = u.clone()
        u = 10 * torch.tanh(u) + mask
        
        if latent is not None:
            u += self.alpha * latent
    
        return F.softmax(u, dim=1), h, c, latent_u
