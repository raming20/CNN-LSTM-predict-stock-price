import tensorflow as tf
import keras

@tf.keras.utils.register_keras_serializable()
class CNNEncoder(keras.layers.Layer):
    def __init__(self, units, *args, **kwargs):
        super(CNNEncoder, self).__init__(*args, **kwargs)
        self.units = units
        
        self.convolutions = keras.layers.TimeDistributed(
            keras.Sequential([
                keras.layers.Conv2D(8, (2, 2), strides=1, activation='relu'),
                # keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
                # keras.layers.Conv2D(8, (2, 2), activation='relu'),
                # keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            ])
        )
        
        self.full_connected = keras.layers.TimeDistributed(
            keras.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dense(256),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(256),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(256),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(256),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(256),
                keras.layers.Dropout(0.2),
            ])
        )
        
        self.rnn = keras.Sequential([
            keras.layers.Bidirectional(merge_mode="sum", layer=keras.layers.GRU(units, return_sequences=True)),
            keras.layers.Dropout(0.1),
            keras.layers.Bidirectional(merge_mode="sum", layer=keras.layers.GRU(units, return_sequences=True)),
            keras.layers.Dropout(0.1),
        ])

    def call(
            self, 
            encoder_input_dict):
        """
            encoder_input_dict: {
                list_images_30_days: (batch_size, 287, 287, 3)
                list_images_7_days: (batch_size, 287, 287, 3)
                list_images_3_days: (batch_size, 287, 287, 3)
            }
            
            (batch_size, sequence_length=3, self.units), (batch_size, self.units)
        """
        
        list_images_30_days = encoder_input_dict["list_images_30_days"]
        list_images_7_days = encoder_input_dict["list_images_7_days"]
        list_images_3_days = encoder_input_dict["list_images_3_days"]
        
        input_rnn = tf.stack(
            [
                # list_images_30_days, 
                list_images_7_days, 
                list_images_3_days
            ],
            axis=1
        ) # (batch_size, sequence_length=3, 287, 287, 3)
        
        input_rnn = self.convolutions(input_rnn)
        input_rnn = self.full_connected(input_rnn)
                
        input_rnn = tf.cast(input_rnn, dtype=tf.float32)
        output_rnn = self.rnn(input_rnn) # (batch_size, sequence_length=3, self.units)
        self.last_state = output_rnn[:, -1, :] # (batch_size, self.units)

        return output_rnn, self.last_state 

@tf.keras.utils.register_keras_serializable()
class CrossAttention(keras.layers.Layer):
    def __init__(self, units, num_heads=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

@tf.keras.utils.register_keras_serializable()
class Decoder(keras.layers.Layer):

    def __init__(self, units, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

        self.units = units

        # 2. The RNN keeps track of what's been generated so far.
        
        self.first_gru = keras.layers.GRU(self.units, return_sequences=True)
        self.last_gru = keras.Sequential([
            keras.layers.Dropout(0.1),
            keras.layers.GRU(self.units, return_sequences=True),
            keras.layers.Dropout(0.1),
        ])

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
        decoder_output_rnn = self.first_gru(decoder_input, initial_state=before_state)
        decoder_output_rnn = self.last_gru(decoder_output_rnn)
        before_state = decoder_output_rnn[:, -1, :]
        
        # decoder_output_rnn: (batch_size, sequence_length_decoder_input, self.units)
        # before_state: (batch_size, self.units)

        decoder_output = self.attention(
            query_input=decoder_output_rnn, 
            key_input=context) # (batch_size, sequence_length_decoder_input, self.units)

        open_and_close = self.output_layer(decoder_output) # (batch_size, sequence_length_decoder_input, 2)

        return open_and_close, before_state


    def get_initial_state(self, decoder_start_input, before_state):
        open_and_close, before_state = self.rnn(decoder_start_input, initial_state=before_state) # (batch_size, self.units)
        return before_state


    def get_next_price(self, context, open_and_close_before, state):
        open_and_close, new_state = self(
            context = context, 
            decoder_input = open_and_close_before,
            state = state
        )

        return open_and_close, new_state


@tf.keras.utils.register_keras_serializable()
class CNNAttention(keras.Model):

    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build the encoder and decoder
        self.units = units
        
        self.encoder = CNNEncoder(units)
        self.decoder = Decoder(units)

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
        
        encoder_input = {
            "list_images_30_days": list_images_30_days,
            "list_images_7_days": list_images_7_days,
            "list_images_3_days": list_images_3_days,
        }
        context, encoder_last_state = self.encoder(encoder_input)
        
        # context: (batch_size, sequence_length_encoder_input, self.units)
        # encoder_last_state: (batch_size, self.units)
        
        open_and_close, decoder_last_state = self.decoder(
            context=context,
            decoder_input=percent_change_of_open_close,
            before_state=encoder_last_state,
        )
        
        # open_and_close: (batch_size, sequence_length_decoder_input, 2)
        # decoder_last_state: (batch_size, self.units)

        return open_and_close
    
    def predict_next_3_days_prices(
        self, 
        list_images_30_days,
            list_images_7_days,
            list_images_3_days,
            start_percent_change_of_open_close
        ):
        """
            
        """
        
        # list_images_30_days: (batch_size, 287, 287, 3)
        # list_images_7_days: (batch_size, 287, 287, 3)
        # list_images_3_days: (batch_size, 287, 287, 3)
        # start_percent_change_of_open_close: (batch_size, 1, 2)
        
        
        encoder_input = {
            "list_images_30_days": list_images_30_days,
            "list_images_7_days": list_images_7_days,
            "list_images_3_days": list_images_3_days,
        }
        
        context, encoder_last_state = self.encoder(encoder_input) 
        # context: (batch_size, sequence_length=3, self.units)
        # encoder_last_state: (batch_size, self.units)
        
        list_open_and_close = []
        open_and_close = start_percent_change_of_open_close # (batch_size, 1, 2)
        last_state = encoder_last_state # (batch_size, self.units)
        for _ in range(3):
            open_and_close, last_state = self.decoder(
                context=context, 
                decoder_input=open_and_close, 
                before_state=last_state
            )
            # open_and_close: (batch_size, 1, 2)
            # last_state: (batch_size, self.units)
            
            list_open_and_close.append(open_and_close)
        
        list_open_and_close = tf.concat(list_open_and_close, axis=1) # list_open_and_close: (batch_size, 3, 2)
        
        return list_open_and_close
    
    def get_config(self):
        # Include 'units' in the config
        config = super(CNNAttention, self).get_config()
        config.update({
            'units': self.units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Ensure 'units' is included in the config
        return cls(**config)
        


    
