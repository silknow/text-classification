"""
Text classification model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList as List


class ConvBlock(nn.Module):
    def __init__(self, input_dim, slen, out_channels, kernel_size,
                 activation='elu'):
        super(ConvBlock, self).__init__()
        # print(f'Conv: k={kernel_size} filters={out_channels}')
        self.conv = nn.Conv1d(input_dim, out_channels, kernel_size)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f'invalid activation f={activation}')

        self.pool = nn.MaxPool1d(slen - kernel_size + 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, embeddings, embeddings_freeze, slen, output_size,
                 dropout_p=0.1,
                 kernel_sizes=(3, 4, 5),
                 channels_outs=(100, 100, 100),
                 hidden_size=100,
                 activation='relu',
                 alpha_dropout=False):
        """
        channels_out[s] is the ouput of the convolution which converts
        channels_in into channels_out. this is commonly called n_filters or
        n_feature_maps and "conceptually corresponds" to the number of
        features extracted by a convolution

        SCNN uses ELUs and AlphaDropout to create a self-normalizing CNN
        The idea is to build smaller (less filters) networks with the same
        performance.
        """
        super(TextCNN, self).__init__()
        self.nk = len(kernel_sizes)
        self.dim_e = embeddings.shape[1]
        self.dim_sum_filter = sum(channels_outs)  # sum of all channels_out
        self.hidden_size = hidden_size

        if activation == 'relu':
            activation_function = F.relu
        elif activation == 'elu':
            activation_function = F.elu
        elif activation == 'gelu':
            activation_function = F.gelu

        # embedding
        self.embed = nn.Embedding.from_pretrained(torch.Tensor(embeddings),
                                                  freeze=embeddings_freeze)

        # Convolution Block
        # by default pytorch uses Lecun Intialization for convolutions
        self.conv_blocks = List([ConvBlock(self.dim_e, slen, f, k, activation)
                                 for k, f in zip(kernel_sizes, channels_outs)])

        # dropout
        self.dropout = None
        self.dropout2 = None
        if alpha_dropout:
            self.dropout = nn.AlphaDropout(dropout_p)
        else:
            self.dropout = nn.Dropout(dropout_p)

        # a fc hidden layer to squeeze into a desired size
        if hidden_size > 0:
            self.fc = nn.Linear(self.dim_sum_filter, self.hidden_size)
            if alpha_dropout:
                self.dropout2 = nn.AlphaDropout(dropout_p)
            else:
                self.dropout2 = nn.Dropout(dropout_p)
            self.fc_act = activation_function
            # output
            self.output = nn.Linear(self.hidden_size, output_size)
        # no squeezing
        else:
            self.output = nn.Linear(self.dim_sum_filter, output_size)

    def forward(self, inputs):
        """Expects fixed length sequences as input. inputs (batch, slen).
        i.e. all batches must have the same fixed length.
        """
        # inputs is (batch, slen)
        x = self.embed(inputs)  # x is (batch, seq, dim_e)
        x = x.transpose(1, 2)   # x is (batch, dim_e, slen)
        # because conv1d requires (batch, channels_in=dim_e, slen)
        # to produce an output (batch, channels_out, slen - k + 1)
        # we then pool1d (kernel=slen-k) over the output of conv1d
        # since 1d works along time (i.e. sequence) this means
        # we get (batch, channels_out=cshannels_outs_k, 1) which we squeeze
        conv_blocks_out = [self.conv_blocks[n](x).squeeze(-1)
                           for n in range(self.nk)]
        # and finally we concatenate all our conv1ds with different kernel
        # sizes together to get (batch, sum_k(channels_outs_k))
        # i.e. we concat the channels_out (i.e. featutres)
        x = torch.cat(conv_blocks_out, dim=1)

        # and do some dropout
        x = self.dropout(x)

        # squeeze into hidden
        if self.hidden_size > 0:
            x = self.fc(x)
            x = self.dropout2(x)
            x = self.fc_act(x)

        # map to classes and return
        x = self.output(x)
        return x


def sn_create_default_model(embeddings, slen, output_size):
    """Create a model with the parameters as defined in D3.3.
    """
    dropout_p = 0.4
    kernel_sizes = (2, 3, 4)
    channels_out = (100, 100, 100)
    hidden_size = 0
    activation = 'gelu'
    alpha_dropout = True

    model = TextCNN(embeddings, True, slen, output_size,
                    dropout_p=dropout_p,
                    kernel_sizes=kernel_sizes,
                    channels_outs=channels_out,
                    hidden_size=hidden_size,
                    activation=activation,
                    alpha_dropout=alpha_dropout)

    return model
