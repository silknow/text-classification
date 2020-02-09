#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >

import sys
import os
import sdnn

models_dir = '/data/euprojects/silknow/models/'
model_names = ['timespan', 'place', 'material', 'technique']

models = {name: sdnn.load_model(os.path.join(models_dir, name))
          for name in model_names}

res = {}
for name in models:
    val = models[name].classify_text(sys.argv[2], sys.argv[1])
    res[name] = val

for name in res:
    val = res[name]['label']
    print(f'{name}: {val}')
