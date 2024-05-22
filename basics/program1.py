# -*- coding: utf-8 -*-


import keras, keras_nlp
features = ["The quick brown fox jumped.", "I forgot my homework."]
labels = [0, 3]

# Pretrained classifier.
classifier = keras_nlp.models.AlbertClassifier.from_preset(
    "albert_base_en_uncased",
    num_classes=4,
)
classifier.fit(x=features, y=labels, batch_size=2)
classifier.predict(x=features, batch_size=2)

# Re-compile (e.g., with a new learning rate).
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
    jit_compile=True,
)
# Access backbone programmatically (e.g., to change `trainable`).
classifier.backbone.trainable = False
# Fit again.
classifier.fit(x=features, y=labels, batch_size=2)

