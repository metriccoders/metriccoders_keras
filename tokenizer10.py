# -*- coding: utf-8 -*-
"""Tokenizer10.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NtJjNeKv0vJLTbWUm8LSBuVX2JlkuzWp
"""

!pip install keras keras_nlp tensorflow tensorflow-text

import keras, keras_nlp

tokenizer = keras_nlp.models.Tokenizer.from_preset("albert_base_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("albert_large_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("albert_extra_large_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("albert_extra_extra_large_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bart_base_en")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bart_large_en")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bart_large_en_cnn")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bert_tiny_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bert_small_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bert_medium_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bert_base_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bert_base_en")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bert_base_zh")

tokenizer("班加罗尔是卡纳塔克邦的首府")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bert_base_multi")

tokenizer("ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ನಗರ")

tokenizer = keras_nlp.models.Tokenizer.from_preset("bert_large_en_uncased")

tokenizer("Bengaluru is the garden city of India")

