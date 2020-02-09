# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >


import dnnhelper


class SilkNOWmodel():
    def __init__(self):
        self.loaded = False
        self.model = None
        self.labels = None
        self.vocab = None

    def load(self, model_path):
        """Load a SilkNOW text classification model."""
        self.model, self.labels, self.vocab = dnnhelper.load_model(model_path)
        self.loaded = True

    def classify_text(self, text, lang):
        """Classify a single text-lang pair."""
        ds = [{'txt': text, 'lang': lang}]
        r = dnnhelper.classify_text(self.model,
                                    self.labels,
                                    self.vocab,
                                    ds)
        return r[0]

    def classify_texts(self, text_lang_pairs):
        """Classify a list of text-lang pairs."""
        ds = [{'txt': p[0], 'lang': p[1]} for p in text_lang_pairs]
        r = dnnhelper.classify_text(self.model,
                                    self.labels,
                                    self.vocab,
                                    ds)
        return r


def load_model(model_load_path):
    """Load a SilkNOW text classification model."""
    model = SilkNOWmodel()
    model.load(model_load_path)
    return model
