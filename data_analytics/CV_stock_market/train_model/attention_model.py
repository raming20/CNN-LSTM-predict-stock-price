import tensorflow as tf
import keras


class Encoder(keras.layers.Layer):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units

        # The RNN layer processes those vectors sequentially.
        self.rnn = keras.layers.Bidirectional(
            merge_mode="sum",
            layer=keras.layers.GRU(units,
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform')) # (batch_size, sequence_length, units)

    def call(self, x):
        """
            x: (batch_size, sequence_length, n_dims_x)
            
            (batch_size, sequence_length, self.units), (batch_size, self.units)
        """
        x = tf.cast(x, dtype=tf.float32)
        x = self.rnn(x) # (batch_size, sequence_length, self.units)
        last_state = x[:, -1, :]
        self.last_state = last_state # (batch_size, self.units)

        return x, last_state 


class CrossAttention(keras.layers.Layer):
    def __init__(self, units, num_heads=1, **kwargs):
        super().__init__()
        self.units = units
        
        self.query = keras.layers.Dense(units)
        self.key = keras.layers.Dense(units)
        self.value = keras.layers.Dense(units)
        
        self.mha = keras.layers.MultiHeadAttention(key_dim=int(units / num_heads), num_heads=num_heads, **kwargs)
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

    def call(self, query_input, key_input):
        """
            query_input: (batch_size, sequence_length_query_input, n_dims)
            key_input: (batch_size, sequence_length_key_input, n_dims)
            
            assert n_dims of context and decoder_input is same
            
            (batch_size, sequence_length_key_input, n_dims)
        """
        query = self.query(query_input)
        key = self.key(key_input)
        value = self.value(key_input)
        
        attn_output, attn_scores = self.mha(
            query=query,
            key=key,
            value=value,
            return_attention_scores=True
        )
        
        # attn_output: (batch_size, sequence_length_key_input, n_dims)
        # attn_scores: (batch_size, num_heads, sequence_length_key_input, sequence_length_query_input) -> giá trị softmax của từng key với query

        self.attn_output = attn_output

        attn_scores = tf.reduce_mean(attn_scores, axis=1) # (batch_size, num_heads, sequence_length_key_input, sequence_length_query_input)
        self.attn_scores = attn_scores

        decoder_output = self.add([query_input, attn_output]) # (batch_size, sequence_length_key_input, n_dims)
        decoder_output = self.layernorm(decoder_output) # (batch_size, sequence_length_key_input, n_dims)

        return decoder_output # (batch_size, sequence_length_key_input, n_dims)


class Decoder(keras.layers.Layer):

    def __init__(self, units):
        super(Decoder, self).__init__()

        self.units = units

        # 2. The RNN keeps track of what's been generated so far.
        self.rnn = keras.layers.GRU(units,
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform')

        self.attention = CrossAttention(units)

        self.output_layer = keras.layers.Dense(2)


    def call(self,
            context, 
            decoder_input,
            before_state):
        """
            context: (batch_size, sequence_length_context, self.units)
            decoder_input: (batch_size, sequence_length_decoder_input, n_dims)
            before_state: (batch_size, self.units)
            
            (
                (batch_size, sequence_length_decoder_input, 2),
                (batch_size, self.units)
            )
        """

        decoder_input = tf.cast(decoder_input, dtype=tf.float32)
        before_state = tf.cast(before_state, dtype=tf.float32)
        decoder_output_rnn = self.rnn(decoder_input, initial_state=before_state)
        before_state = decoder_output_rnn[:, -1, :]
        
        # decoder_output_rnn: (batch_size, sequence_length_decoder_input, self.units)
        # before_state: (batch_size, self.units)

        decoder_output = self.attention(context, decoder_output_rnn) # (batch_size, sequence_length_decoder_input, self.units)

        open_and_close = self.output_layer(decoder_output) # (batch_size, sequence_length_decoder_input, 2)

        return open_and_close, before_state


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


class Attention(keras.Layer):

    def __init__(self, units):
        super().__init__()
        # Build the encoder and decoder
        self.units = units
        
        self.encoder = Encoder(units)
        self.decoder = Decoder(units)

    def call(self, encoder_input, decoder_input):
        
        # encoder_input: (batch_size, sequence_length_encoder_input, n_dims_encoder_input)
        # decoder_input: (batch_size, sequence_length_decoder_input, n_dims_decoder_input)
        
        context, encoder_last_state = self.encoder(encoder_input)
        
        # context: (batch_size, sequence_length_encoder_input, self.units)
        # encoder_last_state: (batch_size, self.units)
        
        open_and_close, decoder_last_state = self.decoder(context, decoder_input, encoder_last_state)
        
        # open_and_close: (batch_size, sequence_length_decoder_input, 2)
        # decoder_last_state: (batch_size, self.units)
        

        return open_and_close, decoder_last_state


class ImageTimeSeries(keras.Model):
    def __init__(self, units):
        super().__init__()
        # Build the encoder and decoder
        self.units = units
        
        self.flatten_image_30_days = keras.layers.Flatten()
        self.flatten_image_7_days = keras.layers.Flatten()
        self.flatten_image_3_days = keras.layers.Flatten()
        
        self.attention = Attention(units)
    
    def call(self, inputs):
        """
            list_images_30_days: (batch_size, 287, 287, 3)
            list_images_7_days: (batch_size, 287, 287, 3)
            list_images_3_days: (batch_size, 287, 287, 3)
            
            percent_change_of_open_close: (batch_size, 3, 2)
        """
        
        (
            list_images_30_days,
            list_images_7_days,
            list_images_3_days,
            percent_change_of_open_close,
        ) = inputs
        
        list_images_30_days = self.flatten_image_30_days(list_images_30_days)
        list_images_7_days = self.flatten_image_30_days(list_images_7_days)
        list_images_3_days = self.flatten_image_30_days(list_images_3_days)
        
        encoder_input = tf.stack(
            [
                list_images_30_days, 
                list_images_7_days, 
                list_images_3_days
            ],
            axis=1
        ) # (batch_size, 3, n_dims)
        
        open_and_close, decoder_last_state = self.attention(encoder_input, percent_change_of_open_close)
        
        return open_and_close
