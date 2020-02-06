import torch
from torch.optim import Adam
from models import BoV, TextCNN, BiRNN, PooledBiRNN, RegLSTM, HANSentence
from csvtxtdataset import seq_collate_pad, seq_collate_pack
from vocabulary import VocabMultiLingual


def get_vocab(args):
    vocab = VocabMultiLingual(sos=None, eos=None, unk=None)
    vocab.load(args.vocab)
    return vocab


def get_embeddings(args):
    return torch.load(args.embeddings)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(args, embs, n_classes):  # noqa: C901
    # create model, optimizer, loss
    model = None
    optim = None
    criterion = None
    collate_fn = None

    slen = args.seqlen
    freeze = not args.no_freeze

    ks = args.kernel_start
    kernel_sizes = (ks, ks + 1, ks + 2)
    channels_outs = (args.filters, args.filters, args.filters)

    if args.model == 'BoV':
        print('Model: BoV')
        model = BoV(embs, n_classes)
        collate_fn = seq_collate_pad
    elif args.model == 'TextCNN':
        print('Model: TextCNN')
        model = TextCNN(embs, freeze, slen, n_classes, dropout_p=args.dropout,
                        kernel_sizes=kernel_sizes, channels_outs=channels_outs,
                        hidden_size=args.hidden_size, activation='relu')
        collate_fn = seq_collate_pad
    elif args.model == 'TextSCNN':
        print('Model: TextSCNN')
        model = TextCNN(embs, freeze, slen, n_classes, dropout_p=args.dropout,
                        kernel_sizes=kernel_sizes, channels_outs=channels_outs,
                        hidden_size=args.hidden_size, activation='elu',
                        alpha_dropout=True)
        collate_fn = seq_collate_pad
    elif args.model == 'TextGCNN':
        print('Model: TextGCNN')
        model = TextCNN(embs, freeze, slen, n_classes, dropout_p=args.dropout,
                        kernel_sizes=kernel_sizes, channels_outs=channels_outs,
                        hidden_size=args.hidden_size, activation='gelu',
                        alpha_dropout=True)
        collate_fn = seq_collate_pad
    elif args.model == 'BiLSTM':
        print('Model: BiLSTM')
        model = BiRNN(embs, freeze, n_classes, hidden_size=args.hidden_size,
                      dropout_p=args.dropout)
        collate_fn = seq_collate_pack
    elif args.model == 'BiLSTMpool':
        print('Model: Pooled BiLSTM')
        model = PooledBiRNN(embs, freeze, n_classes,
                            hidden_size=args.hidden_size,
                            dropout_p=args.dropout)
        collate_fn = seq_collate_pack
    elif args.model == 'RegLSTM':
        print('Model: Pooled BiLSTM')
        model = RegLSTM(embs, freeze, n_classes, hidden_size=args.hidden_size,
                        dropout_e=args.edrop, dropout_w=args.wdrop,
                        dropout_out=args.dropout)
        collate_fn = seq_collate_pack
    elif args.model == 'HANSentence':
        print('Model: HANSentence')
        model = HANSentence(embs, freeze, n_classes,
                            hidden_size=args.hidden_size,
                            att_size=args.hidden_size,
                            dropout_p=args.dropout)
        collate_fn = seq_collate_pack
    print(f'Freeze={freeze}')
    print(f'Parameters: {count_parameters(model)}')

    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=args.lr, weight_decay=args.wdecay)

    multilabel = False
    if multilabel:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    return model, optim, criterion, collate_fn


def get_cnn_model(args, embs, n_classes):  # noqa: C901
    # create model, optimizer, loss
    model = None
    optim = None
    criterion = None
    collate_fn = None

    slen = args.seqlen
    freeze = not args.no_freeze

    channels_outs = (args.filters, args.filters, args.filters)
    model = TextCNN(embs, freeze, slen, n_classes, dropout_p=args.dropout,
                    channels_outs=channels_outs,
                    hidden_size=0, activation=args.activation,
                    alpha_dropout=args.alpha_dropout)
    collate_fn = seq_collate_pad
    print(f'Freeze={freeze}')
    print(f'Parameters: {count_parameters(model)}')

    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=args.lr, weight_decay=args.wdecay)

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    return model, optim, criterion, collate_fn
