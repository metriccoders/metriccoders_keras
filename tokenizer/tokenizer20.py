
import keras, keras_nlp

string_text = "Bengaluru is the garden city of India"

tokenizer = keras_nlp.models.BartTokenizer.from_preset("bart_base_en")

tokenizer(string_text)

tokenizer = keras_nlp.models.BartTokenizer.from_preset("bart_large_en")

tokenizer(string_text)

tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_base_en")

tokenizer(string_text)

tokenizer = keras_nlp.models.BloomTokenizer.from_preset("bloom_560m_multi")

tokenizer(string_text)

tokenizer = keras_nlp.models.BloomTokenizer.from_preset("bloom_1.1b_multi")

tokenizer(string_text)

tokenizer = keras_nlp.models.BloomTokenizer.from_preset("bloomz_560m_multi")

tokenizer(string_text)

tokenizer = keras_nlp.models.DebertaV3Tokenizer.from_preset("deberta_v3_extra_small_en")

tokenizer(string_text)

