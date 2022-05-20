import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.densenet121.features(x)
        x = F.relu(x)
        return x


class Attention(nn.Module):

    def __init__(self, k_size, v_size, affine_size=512):
        super().__init__()
        self.affine_k = nn.Linear(k_size, affine_size, bias=False)
        self.affine_v = nn.Linear(v_size, affine_size, bias=False)
        self.affine = nn.Linear(affine_size, 1, bias=False)

    def forward(self, k, v):
        # k: batch size x hidden size
        # v: batch size x spatial size x hidden size
        # z: batch size x spatial size
        # TODO other ways of attention?
        content_v = self.affine_k(k).unsqueeze(1) + self.affine_v(v)
        z = self.affine(torch.tanh(content_v)).squeeze(2)
        alpha = torch.softmax(z, dim=1)
        context = (v * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class WordSAT(nn.Module):

    def __init__(self, vocab_size, feat_size=1024, embed_size=256, hidden_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.atten = Attention(hidden_size, feat_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.init_h = nn.Linear(2 * feat_size, hidden_size)
        self.init_c = nn.Linear(2 * feat_size, hidden_size)
        self.lstmcell = nn.LSTMCell(embed_size + 2 * feat_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, cnn_feats1, cnn_feats2, captions=None, max_len=100):
        batch_size = cnn_feats1.size(0)
        if captions is not None:
            seq_len = captions.size(1)
        else:
            seq_len = max_len

        cnn_feats1_t = cnn_feats1.view(batch_size, self.feat_size, -1).permute(0, 2, 1)
        cnn_feats2_t = cnn_feats2.view(batch_size, self.feat_size, -1).permute(0, 2, 1)
        global_feats1 = cnn_feats1.mean(dim=(2, 3))
        global_feats2 = cnn_feats2.mean(dim=(2, 3))
        h = self.init_h(torch.cat((global_feats1, global_feats2), dim=1))
        c = self.init_c(torch.cat((global_feats1, global_feats2), dim=1))

        logits = cnn_feats1.new_zeros((batch_size, seq_len, self.vocab_size), dtype=torch.float)

        if captions is not None:
            embeddings = self.embed(captions)
            for t in range(seq_len):
                context1, alpha1 = self.atten(h, cnn_feats1_t)
                context2, alpha2 = self.atten(h, cnn_feats2_t)
                context = torch.cat((context1, context2), dim=1)
                h, c = self.lstmcell(torch.cat((embeddings[:, t], context), dim=1), (h, c))
                logits[:, t] = self.fc(self.dropout(h))

            return logits

        else:
            x_t = cnn_feats1.new_full((batch_size,), 1, dtype=torch.long)
            for t in range(seq_len):
                embedding = self.embed(x_t)
                context1, alpha1 = self.atten(h, cnn_feats1_t)
                context2, alpha2 = self.atten(h, cnn_feats2_t)
                context = torch.cat((context1, context2), dim=1)
                h, c = self.lstmcell(torch.cat((embedding, context), dim=1), (h, c))
                logit  =self.fc(h)
                x_t = logit.argmax(dim=1)
                logits[:, t] = logit

            return logits.argmax(dim=2)


class Encoder2Decoder(nn.Module):

    def __init__(self, num_classes, vocab_size, feat_size=1024, embed_size=256, hidden_size=512):
        super().__init__()
        self.encoder = Encoder(num_classes)
        self.decoder = WordSAT(vocab_size, feat_size, embed_size, hidden_size)

    def forward(self, images1, images2, captions=None, max_len=100):
        cnn_feats1 = self.encoder(images1)
        cnn_feats2 = self.encoder(images2)
        return self.decoder(cnn_feats1, cnn_feats2, captions, max_len)