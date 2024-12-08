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
        """
            x: (batch_size, sequence_length, n_dims_x)
            
            (batch_size, sequence_length, self.units)
        """
        x, final_state_fwd1, final_state_bwd1 = self.rnn(x) 

        self.final_state_fwd1 = final_state_fwd1 # (batch_size, self.units)
        self.final_state_bwd1 = final_state_bwd1 # (batch_size, self.units)

        return x # (batch_size, sequence_length=number_prices, self.units)


class CrossAttention(keras.layers.Layer):
    def __init__(self, units, num_heads=1, **kwargs):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(key_dim=int(units / num_heads), num_heads=num_heads, **kwargs)
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

    def call(self, context, decoder_input):
        """
            context: (batch_size, sequence_length_context, n_dims)
            decoder_input: (batch_size, sequence_length_decoder_input, n_dims)
            
            assert n_dims of context and decoder_input is same
            
            (batch_size, sequence_length_decoder_input, n_dims)
        """
        
        attn_output, attn_scores = self.mha(
            query=decoder_input,
            key=context,
            value=context,
            return_attention_scores=True
        )
        
        # attn_output: (batch_size, sequence_length_decoder_input, n_dims)
        # attn_scores: (batch_size, num_heads, sequence_length_decoder_input, sequence_length_context) -> giá trị softmax của từng key với query

        self.attn_output = attn_output

        attn_scores = tf.reduce_mean(attn_scores, axis=1) # (batch_size, num_heads, sequence_length_decoder_input, sequence_length_context)
        self.attn_scores = attn_scores

        decoder_output = self.add([decoder_input, attn_output]) # (batch_size, sequence_length_decoder_input, n_dims)
        decoder_output = self.layernorm(decoder_output) # (batch_size, sequence_length_decoder_input, n_dims)

        return decoder_output # (batch_size, sequence_length_decoder_input, n_dims)


class Decoder(keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units):
        super(Decoder, self).__init__()

        self.units = units

        # 2. The RNN keeps track of what's been generated so far.
        self.rnn = keras.layers.GRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

        self.attention = CrossAttention(units)

        self.output_layer = keras.layers.Dense(2)


    def call(self,
            context, decoder_input,
            state=None):
        """
            context: (batch_size, sequence_length_context, self.units)
            decoder_input: (batch_size, sequence_length_decoder_input, n_dims)
            
            (
                (batch_size, sequence_length_decoder_input, 2),
                (batch_size, self.units)
            )
        """

        decoder_output_rnn, state = self.rnn(decoder_input, initial_state=state) 
        
        # decoder_output_rnn: (batch_size, sequence_length_decoder_input, self.units)
        # state: (batch_size, self.units)

        decoder_output = self.attention(context, decoder_output_rnn) # (batch_size, sequence_length_decoder_input, self.units)

        open_and_close = self.output_layer(decoder_output) # (batch_size, sequence_length_decoder_input, 2)

        return open_and_close, state


    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_vector = tf.fill([batch_size, 1], self.start_price)  # (batch_size, 1)
        done = tf.zeros([batch_size, 1], dtype=tf.bool) # (batch_size, 1)
        
        initial_state = self.rnn.get_initial_state(start_vector)[0] # (batch_size, self.units)
        return start_vector, done, initial_state


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

    def __init__(self, units):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(units)
        decoder = Decoder(units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, decoder_input = inputs
        context = self.encoder(context)
        logits = self.decoder(context, decoder_input)

        # #TODO(b/250038731): remove this
        # try:
        # # Delete the keras mask, so keras doesn't scale the loss+accuracy.
        #     del logits._keras_mask 
        # except AttributeError:
        #     pass

        return logits

