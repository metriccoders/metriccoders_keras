import keras, keras_nlp

tokenizer = keras_nlp.models.AlbertTokenizer.from_preset("albert_base_en_uncased")

tokenizer("Bengaluru is called the garden city of India")

tokenizer(["Bengaluru is called the garden city of India", "Bengaluru is the capital city of Karnataka"])

tokenizer.detokenize(tokenizer("Bengaluru is the capital city of Karnataka"))

import io

bytes_io = io.BytesIO()

import tensorflow as tf

ds = tf.data.Dataset.from_tensor_slices(["Bengaluru is the capital city of Karnataka"])

import sentencepiece

sentencepiece.SentencePieceTrainer.train(
    sentence_iterator=ds.as_numpy_iterator(),
    model_writer=bytes_io,
    vocab_size=10,
    model_type="WORD",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="[CLS]",
    eos_piece="[SEP]",
    user_defined_symbols="[MASK]",
)

tokenizer = keras_nlp.models.AlbertTokenizer(proto=bytes_io.getvalue())

tokenizer("Bengaluru is the capital city of Karnataka")

