
import tensorflow as tf
import keras_nlp

def predict(text):
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
    encoding = preprocessor(text)

    # prepping the preprocessed
    encoding['padding_mask'] = tf.cast(encoding['padding_mask'], tf.int32)
    encoding['padding_mask'] = tf.constant(encoding['padding_mask'].numpy().reshape(1, -1))
    encoding['token_ids'] = tf.constant(encoding['token_ids'].numpy().reshape(1, -1))
    encoding['segment_ids'] = tf.constant(encoding['segment_ids'].numpy().reshape(1, -1))

    # model defination and signature function
    model = tf.saved_model.load('api\model')
    predict = model.signatures['serving_default']
    score_tensor = predict(**encoding)

    score = score_tensor['logits'].numpy()[0, 0]

    return score