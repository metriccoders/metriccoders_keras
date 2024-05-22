
import keras, keras_nlp

tokenizer = keras_nlp.models.Tokenizer.from_preset("albert_base_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("albert_large_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("albert_extra_large_en_uncased")

tokenizer("Bengaluru is the garden city of India")

tokenizer = keras_nlp.models.Tokenizer.from_preset("albert_extra_extra_large_en_uncased")

tokenizer("Bengaluru is the garden city of India")

