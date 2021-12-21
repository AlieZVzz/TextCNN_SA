import torch
from d2l import torch as d2l
import torch.nn as nn

batchsize = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size=batchsize)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_size, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.conv = nn.ModuleList()
        for c, k in zip(num_channels, kernel_size):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        encoding = torch.cat([torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1) for conv in self
                             .convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


embed_size, kernel_size, num_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_size, num_channels)


def init_weihts(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weihts)

glove_embedding = d2l.TokenEmbedding('globe.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False

lr = 0.001
num_epochs = 5
optimizer = torch.optim.Adam(net.parameter(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, optimizer, num_epochs, devices)
