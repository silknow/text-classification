#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >

import argparse

from dnnhelper import (train_model,
                       create_embeddings,
                       eval_model,
                       classify_csv)


def parse_args():
    parser = argparse.ArgumentParser(description='SilkNOW Text Classifier',
                                     usage='''sntxtclassify <command> [<args>]

The available commands are:
   train        Train a text classification model
   embeddings   Create embeddings file
   evaluate     Evaluate a text classification model
   classify     Classify text samples
''')
    subparsers = parser.add_subparsers(required=True, dest='command')

    #
    # TRAIN
    #
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data-train', type=str, help='train CSV file',
                              required=True)

    train_parser.add_argument('--target', type=str,
                              help='CSV column target',
                              required=True)

    train_parser.add_argument('--pretrained-embeddings', type=str,
                              help='pretrained embeddings path',
                              default='embeddings')

    train_parser.add_argument('--all-embeddings', action='store_true',
                              help='Keep all Keep all word vectors')

    train_parser.add_argument('--model-save', type=str, help='save model path',
                              required=True)
    #
    # Embeddings
    #
    emb_parser = subparsers.add_parser('embeddings')
    emb_parser.add_argument('--data-train', type=str, help='train CSV file',
                            required=True)
    emb_parser.add_argument('--pretrained-embeddings', type=str,
                            help='pretrained embeddings path',
                            required=True)
    emb_parser.add_argument('--save', type=str, help='save embeddings path',
                            required=True)

    #
    # Evaluate
    #
    eval_parser = subparsers.add_parser('evaluate')
    eval_parser.add_argument('--model-load', type=str, help='model path',
                             required=True)
    eval_parser.add_argument('--data-test', type=str, help='test CSV path',
                             required=True)
    eval_parser.add_argument('--target', type=str, help='CSV column target',
                             required=True)

    #
    # Classify
    #
    classify_parser = subparsers.add_parser('classify')
    classify_parser.add_argument('--model-load', type=str, help='model path')
    classify_parser.add_argument('--data-input', type=str,
                                 help='input CSV path', required=True)
    classify_parser.add_argument('--data-output', type=str,
                                 help='output CSV path', required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.command == 'train':
        train_model(args.data_train,
                    args.pretrained_embeddings,
                    args.target,
                    args.model_save,
                    args.all_embeddings)
    elif args.command == 'embeddings':
        create_embeddings(args.data_train,
                          args.pretrained_embeddings,
                          args.save)
    elif args.command == 'evaluate':
        eval_model(args.model_load, args.data_test, args.target)
    elif args.command == 'classify':
        classify_csv(args.model_load, args.data_input, args.data_output)


if __name__ == '__main__':
    main()
