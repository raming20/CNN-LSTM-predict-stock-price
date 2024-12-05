import tensorflow as tf
import keras


class Encoder(keras.layers.Layer):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units

        # The RNN layer processes those vectors sequentially.
        self.rnn = keras.layers.Bidirectional(
            merge_mode='sum',
            layer=keras.layers.GRU(units,
                                # Return the sequence and state
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform', return_state=True)) # (batch_size, sequence_length=number_prices, units)

    def call(self, x):

        x, final_state_fwd1, final_state_bwd1 = self.rnn(x) # (batch_size, sequence_length=number_prices, units)

        self.final_state_fwd1 = final_state_fwd1
        self.final_state_bwd1 = final_state_bwd1

        return x


class CrossAttention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

    def call(self, x, context):
        
        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)


        attn_scores = tf.reduce_mean(attn_scores, axis=1)

        self.last_attention_weights = attn_scores
        self.attn_output = attn_output

        x = self.add([x, attn_output]) # (batch_size, number_prices, n_dims)
        x = self.layernorm(x)

        return x


class Decoder(keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units):
        super(Decoder, self).__init__()

        self.units = units
        self.start_price = 0
        self.end_price = 0

        # 2. The RNN keeps track of what's been generated so far.
        self.rnn = keras.layers.GRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

        self.attention = CrossAttention(units)

        self.output_layer = keras.layers.Dense(2)


    def call(self,
            context, x,
            state=None,
            return_state=False):

        x, state = self.rnn(x, initial_state=state) # (batch_size, number_prices, n_dims)

        x = self.attention(x, context) # (batch_size, number_prices, n_dims)
        self.last_attention_weights = self.attention.last_attention_weights

        logits = self.output_layer(x) # (batch_size, number_prices, vocab_size)

        if return_state:
            return logits, state
        else:
            return logits


    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_vector = tf.fill([batch_size, 1], self.start_price)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        return start_vector, done, self.rnn.get_initial_state(start_vector)[0]


    def get_next_price(self, context, next_price, done, state, temperature = 0.0):
        logits, state = self(
            context, next_price,
            state = state,
            return_state=True)

        if temperature == 0.0:
            next_price = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :]/temperature
            next_price = tf.random.categorical(logits, num_samples=1)

        done = done | (next_price == self.end_price)
        next_price = tf.where(done, tf.constant(0, dtype=tf.int64), next_price)

        return next_price, done, state


class Translator(keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units,
                context_text_processor,
                target_text_processor):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(context_text_processor, units)
        decoder = Decoder(target_text_processor, units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        #TODO(b/250038731): remove this
        try:
        # Delete the keras mask, so keras doesn't scale the loss+accuracy.
            del logits._keras_mask 
        except AttributeError:
            pass

        return logits

