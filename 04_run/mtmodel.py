import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList as List
from tqdm import tqdm


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


class SharedTextCNN(nn.Module):
    def __init__(self, embeddings, embeddings_freeze, slen, output_size,
                 dropout_p=0.1,
                 kernel_sizes=(3, 4, 5),
                 channels_outs=(100, 100, 100),
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
        super(SharedTextCNN, self).__init__()
        self.nk = len(kernel_sizes)
        self.dim_e = embeddings.shape[1]
        self.dim_sum_filter = sum(channels_outs)  # sum of all channels_out

        if activation == 'relu':
            self.activation_function = F.relu
        elif activation == 'elu':
            self.activation_function = F.elu
        elif activation == 'gelu':
            self.activation_function = F.gelu

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
            self.dropout2 = nn.AlphaDropout(dropout_p)
        else:
            self.dropout = nn.Dropout(dropout_p)
            self.dropout2 = nn.Dropout(dropout_p)

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

        # map to output size, apply non-lineariy and return
        x = self.output(x)
        x = self.activation_function(x)
        x = self.dropout2(x)
        return x


class MTHead(nn.Module):
    def __init__(self, name, input_size, output_size):
        super(MTHead, self).__init__()
        self.name = name
        self.output = nn.Linear(input_size, output_size)

    def forward(self, x):
        return (self.name, self.output(x))


class MTTextCNN(nn.Module):
    def __init__(self, embeddings, embeddings_freeze, slen, named_outs,
                 dropout_p=0.1,
                 kernel_sizes=(3, 4, 5),
                 channels_outs=(100, 100, 100),
                 hidden_size=300,
                 activation='relu',
                 alpha_dropout=False):
        super(MTTextCNN, self).__init__()
        self.hidden_size = hidden_size
        self.nt = len(named_outs.keys())

        self.shared = SharedTextCNN(embeddings, embeddings_freeze, slen,
                                    hidden_size, dropout_p, kernel_sizes,
                                    channels_outs, activation, alpha_dropout)

        self.heads = List([MTHead(name, hidden_size, named_outs[name])
                           for name in named_outs])

    def forward(self, x):
        shared_out = self.shared(x)
        named_outs = [self.heads[n](shared_out)
                      for n in range(self.nt)]
        named_outs = {k: v for k, v in named_outs}
        return named_outs


def mttrain(model, optim, named_iters, epochs, clip=0):
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    iters = named_iters.items()
    tasks = [task for task, _ in iters]
    iters = [iterator for _, iterator in iters]

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_epoch_loss = 0.0
        steps = 0
        for examples in zip(*iters):
            x = None
            y = None

            # select a random task
            r = torch.randint(0, len(tasks), (1,))[0].item()
            task = tasks[r]
            x, y = examples[r]

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # forward
            outs = model(x)
            output = outs[task]

            # backward
            optim.zero_grad()
            loss = criterion(output, y)
            loss.backward()

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()

            total_epoch_loss += loss.item() / len(y)  # len(y) = batch_size
            steps += 1
        # display epoch stats
        total_epoch_loss /= steps


def mteval(model, named_iters):
    import numpy as np
    from torchutil import output_to_multiclass
    from sklearn.metrics import accuracy_score

    model.eval()

    for task in named_iters:
        y_prd = []
        y_tru = []
        iterator = named_iters[task]
        with torch.no_grad():
            for x, y in iterator:
                if torch.cuda.is_available():
                    x = x.cuda()

                # forward
                output = model(x)
                output = output[task]

                output = output_to_multiclass(output, dim=1)
                output = output.item()
                y_prd.append(output)
                y_tru.append(y)

        y_prd = np.array(y_prd)
        y_tru = np.array(y_tru)
        acc = accuracy_score(y_tru, y_prd)
        print(f'Task = {task}\tacc = {acc}')


def get_datafiles_iid():
    datafiles = [
        ('material',
         '/data/euprojects/silknow/tasks/material/iid.trn.csv',
         '/data/euprojects/silknow/tasks/material/iid.tst.csv'),
        ('timespan',
         '/data/euprojects/silknow/tasks/timespan/iid.trn.csv',
         '/data/euprojects/silknow/tasks/timespan/iid.tst.csv'),
        ('place',
         '/data/euprojects/silknow/tasks/place/iid.trn.csv',
         '/data/euprojects/silknow/tasks/place/iid.tst.csv'),
    ]
    return datafiles


def get_datafiles_xling():
    datafiles = [
        ('material',
         '/data/euprojects/silknow/tasks/material/xling.trn.csv',
         '/data/euprojects/silknow/tasks/material/xling.tst.csv'),
        ('timespan',
         '/data/euprojects/silknow/tasks/timespan/xling.trn.csv',
         '/data/euprojects/silknow/tasks/timespan/xling.tst.csv'),
        ('place',
         '/data/euprojects/silknow/tasks/place/xling.trn.csv',
         '/data/euprojects/silknow/tasks/place/xling.tst.csv'),
    ]
    return datafiles


def get_datafiles_xling_alt():
    datafiles = [
        ('material',
         '/data/euprojects/silknow/tasks/material/xling.alt.csv',
         '/data/euprojects/silknow/tasks/material/xling.tst.csv'),
        ('timespan',
         '/data/euprojects/silknow/tasks/timespan/xling.alt.csv',
         '/data/euprojects/silknow/tasks/timespan/xling.tst.csv'),
        ('place',
         '/data/euprojects/silknow/tasks/place/xling.alt.csv',
         '/data/euprojects/silknow/tasks/place/xling.tst.csv'),
    ]
    return datafiles


def get_datafiles_group():
    datafiles = [
        ('material',
         '/data/euprojects/silknow/tasks/material/group.trn.csv',
         '/data/euprojects/silknow/tasks/material/group.tst.csv'),
        ('timespan',
         '/data/euprojects/silknow/tasks/timespan/group.trn.csv',
         '/data/euprojects/silknow/tasks/timespan/group.tst.csv'),
        ('place',
         '/data/euprojects/silknow/tasks/place/group.trn.csv',
         '/data/euprojects/silknow/tasks/place/group.tst.csv'),
    ]
    return datafiles


def get_named_iters_iid(vocab, seqlen, batch_size):
    from csvtxtdataset import CSVDatasetMultilingualMulticlass
    from csvtxtdataset import seq_collate_pad
    from torch.utils.data import DataLoader
    pad = True
    collate_fn = seq_collate_pad

    # datafiles = get_datafiles_iid()
    # datafiles = get_datafiles_xling()
    # datafiles = get_datafiles_xling_alt()
    datafiles = get_datafiles_group()
    named_trn_iters = {}
    named_tst_iters = {}
    named_labels = {}

    for task, trn_file, tst_file in datafiles:
        trn = CSVDatasetMultilingualMulticlass(vocab, trn_file, '\t',
                                               'txt', 'lang',
                                               task, seqlen, pad)
        labels = trn.get_labels()
        tst = CSVDatasetMultilingualMulticlass(vocab, tst_file, '\t',
                                               'txt', 'lang',
                                               task, seqlen, pad,
                                               labels)
        print(f'Task = {task}\ttrain steps = {len(trn) // batch_size}')

        trn_loader = DataLoader(trn, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_fn)
        tst_loader = DataLoader(tst, batch_size=1, shuffle=False,
                                collate_fn=collate_fn)

        named_trn_iters[task] = trn_loader
        named_tst_iters[task] = tst_loader
        named_labels[task] = labels

    return named_trn_iters, named_tst_iters, named_labels


def get_vocab(vocab_file):
    from vocabulary import VocabMultiLingual
    vocab = VocabMultiLingual(sos=None, eos=None, unk=None)
    vocab.load(vocab_file)
    return vocab


def get_embeddings(emb_file):
    return torch.load(emb_file)


def main():
    # set params
    slen = 300
    epochs = 200 * 3 * 3
    filters = 300
    hidden_size = 300
    dropout_p = 0.2
    clip = 0
    alpha_dropout = True
    activation = 'gelu'
    lr = 3e-4
    wdecay = 0.0
    bsize = 64

    # load vocab and embeddings
    VOCAB_FILE = '/data/euprojects/silknow/vocab.txt'
    EMB_FILE = '/data/euprojects/silknow/embedding_matrix.pt'
    vocab = get_vocab(VOCAB_FILE)
    embeddings = get_embeddings(EMB_FILE)

    # load data
    named_trn_iters, named_tst_iters, named_labels = get_named_iters_iid(vocab,
                                                                         slen,
                                                                         bsize)
    named_outs = {k: len(named_labels[k]) for k in named_labels}
    # create model and optimizer
    channels_outs = (filters, filters, filters)
    # kernel_sizes = (3, 4, 5)
    kernel_sizes = (2, 3, 4)
    model = MTTextCNN(embeddings, True, slen, named_outs, dropout_p,
                      kernel_sizes, channels_outs, hidden_size,
                      activation, alpha_dropout)

    if torch.cuda.is_available():
        model = model.cuda()
    # train
    from torch.optim import Adam
    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=lr, weight_decay=wdecay)
    mttrain(model, optim, named_trn_iters, epochs, clip)
    # evals
    mteval(model, named_tst_iters)

    return


if __name__ == '__main__':
    main()
