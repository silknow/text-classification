"""
Text classification models for use with nn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList as List
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from awd import WeightDrop, embedded_dropout


class BoV(nn.Module):
    """Bag of Vectors text classifier
    """

    def __init__(self, embeddings, embeddings_freeze, output_size):
        super(BoV, self).__init__()
        self.dim_e = embeddings.shape[1]

        # embedding
        self.bag = nn.EmbeddingBag.from_pretrained(torch.Tensor(embeddings),
                                                   freeze=embeddings_freeze,
                                                   mode='mean')
        self.output = nn.Linear(self.dim_e, output_size)

    def forward(self, inputs):
        """Expects padded sequences as input.
        Different batches can have different lengths.
        inputs (B, S)
        """
        # inputs -> (B, S)
        x = self.bag(inputs)  # x -> (B,)
        out = self.output(x)  # out -> (output_size,)
        return out


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


class BiRNN(nn.Module):
    def __init__(self, embeddings, embeddings_freeze, output_size,
                 hidden_size=100,
                 dropout_p=0.1):
        """
        """
        super(BiRNN, self).__init__()
        self.dim_e = embeddings.shape[1]
        self.dim_o = hidden_size

        # embedding
        self.embed = nn.Embedding.from_pretrained(torch.Tensor(embeddings),
                                                  freeze=embeddings_freeze)

        self.lstm = nn.LSTM(input_size=self.dim_e,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout_p)
        self.output = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs):
        """Expects packed sequences as input.
        Each batch may have a different length.
        """
        batch_size = len(inputs.batch_sizes)
        # unpack
        x, lengths = pad_packed_sequence(inputs, batch_first=True)
        # x is (batch, slen)
        batch_size = x.size(0)
        x = self.embed(x)  # x is (batch, seq, dim_e)
        # but we need x to be (slen, batch, dim_e)
        x = x.transpose(0, 1)

        # rnn
        x = pack_padded_sequence(x, lengths, batch_first=False)
        _, (x, _) = self.lstm(x)  # we want x=$h_(t=slen)$ (last hidden output)
        # we have x as (num_directions=2, batch, hidden)
        # we transpose it to (batch, num_direction, hidden)
        # than we view it as (batch, 1, hidden * 2) and finally we
        # squeeze it to (batch, hidden * 2) because we wanted to have
        # the directions concatenated  as a single feature vector (per input)
        x = x.transpose(0, 1).contiguous().view(batch_size, 1, -1).squeeze(1)
        # we need to have squeeze(dim=1) because of the batch_size=1 case

        # do some dropout
        x = self.dropout(x)
        # map to classes
        x = self.output(x)
        return x


class PooledBiRNN(BiRNN):
    """Same as BiRNN but instead of using just the last hidden state of
    the RNN. I
    """

    def __init__(self, *args, **kwargs):
        super(PooledBiRNN, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        """Expects packed sequences as input.
        Each batch may have a different length.
        """
        # unpack
        x, lengths = pad_packed_sequence(inputs, batch_first=True)
        # x is (batch, slen)
        x = self.embed(x)  # x is (batch, seq, dim_e)
        # but we need x to be (slen, batch, dim_e)
        x = x.transpose(0, 1)

        # rnn
        x = pack_padded_sequence(x, lengths, batch_first=False)
        x, (_, _) = self.lstm(x)
        # x is (slen, batch, num_directions * hidden_size)
        x, lengths = pad_packed_sequence(x, batch_first=False)
        # turn into batch first (batch, slen, num_dirs * hiddden)
        # and then (batch, num_dirs * hidden, slen)
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2))  # (batch, num_dirs * hidden, 1)
        x = x.squeeze(2)

        # do some dropout
        x = self.dropout(x)
        # map to classes
        x = self.output(x)
        return x


class RegLSTM(nn.Module):
    """
    RegLSTM - basically a AWD-LSTM converted to text classification.
    See
    Rethinking Complex Neural Network Architectures for Document Classifcation
    Adhikari et all
    """

    def __init__(self, embeddings, embeddings_freeze, output_size,
                 hidden_size=512,
                 dropout_e=0.1,
                 dropout_w=0.2,
                 dropout_out=0.5):
        super(RegLSTM, self).__init__()
        self.dropout_e = dropout_e
        self.dim_e = embeddings.shape[1]
        self.dim_o = hidden_size

        # embedding
        self.embed = nn.Embedding.from_pretrained(torch.Tensor(embeddings),
                                                  freeze=embeddings_freeze)

        self.lstm = nn.LSTM(input_size=self.dim_e,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=True)
        self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'],
                               dropout=dropout_w)

        self.dropout = nn.Dropout(dropout_out)
        self.output = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs):
        """Expects packed sequences as input.
        Each batch may have a different length.
        """
        # unpack
        x, lengths = pad_packed_sequence(inputs, batch_first=True)
        # x is (batch, slen)
        x = embedded_dropout(self.embed, x,
                             dropout=self.dropout_e if self.training else 0)
        # x is (batch, seq, dim_e)
        # but we need x to be (slen, batch, dim_e)
        x = x.transpose(0, 1)

        # rnn
        x = pack_padded_sequence(x, lengths, batch_first=False)
        x, (_, _) = self.lstm(x)
        # x is (slen, batch, num_directions * hidden_size)
        x, lengths = pad_packed_sequence(x, batch_first=False)
        rnn_out = x
        # turn into batch first (batch, slen, num_dirs * hiddden)
        # and then (batch, num_dirs * hidden, slen)
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2))  # (batch, num_dirs * hidden, 1)
        x = x.squeeze(2)

        # do some dropout
        x = self.dropout(x)
        # map to classes
        x = self.output(x)
        return x, rnn_out


class HANAttention(nn.Module):
    """Implements attention for HAN
    """

    def __init__(self, input_size, att_size):
        super(HANAttention, self).__init__()
        self.dim_i = input_size
        self.dim_a = att_size

        self.linear = nn.Linear(self.dim_i, self.dim_a, bias=True)
        self.u_w = nn.Linear(self.dim_a, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs):
        #  inputs is (slen, batch, dim_i) i.e. cat hidden states of a GRU
        u_t = torch.tanh(self.linear(inputs))   # u_it (slen, batch, dim_a)
        alpha = self.softmax(self.u_w(u_t))  # alpha_it (sln, batch, 1)
        return alpha


class HANSentence(nn.Module):
    """A classifier that implements only the sentence part of the HAN."""

    def __init__(self, embeddings, embeddings_freeze, output_size,
                 hidden_size=100,
                 att_size=200, dropout_p=0):
        super(HANSentence, self).__init__()

        self.dim_e = embeddings.shape[1]
        self.dim_o = hidden_size
        self.dim_a = att_size

        # embedding
        self.embed = nn.Embedding.from_pretrained(torch.Tensor(embeddings),
                                                  freeze=embeddings_freeze)

        self.gru = nn.GRU(input_size=self.dim_e,
                          hidden_size=hidden_size,
                          num_layers=1,
                          bidirectional=True)

        self.att_word = HANAttention(hidden_size * 2, att_size)

        self.dropout = nn.Dropout(dropout_p)

        self.output = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs):
        # unpack
        x, lengths = pad_packed_sequence(inputs, batch_first=True)
        # x is (batch, slen)
        x = self.embed(x)  # x is (batch, seq, dim_e)
        # but we need x to be (slen, batch, dim_e)
        x = x.transpose(0, 1)

        # rnn
        x = pack_padded_sequence(x, lengths, batch_first=False)
        x, _ = self.gru(x)
        x, lengths = pad_packed_sequence(x, batch_first=False)
        # x is (slen, batch, num_directions * hidden_size)
        # x, lengths = pad_packed_sequence(x, batch_first=False)

        # attention
        alpha = self.att_word(x)
        x = torch.sum(alpha * x, dim=0)

        # do some dropout
        x = self.dropout(x)
        # map to classes
        x = self.output(x)
        return x
