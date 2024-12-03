def channels_to_time_step_attention_ema_macd_trend(image_shape, days_result, days_of_ema_9=2, days_of_macd_history=2):

    # Hàm Attention
    def attention_3d_block(inputs):
        # inputs.shape = (batch_sizes, time_steps=days_result, features=lstm_units)
        time_steps = inputs.shape[1]
        
        a = keras.layers.Permute((2, 1))(inputs)  # (batch_sizes, features=lstm_units, time_steps=days_result)
        
        # Tính attention score: (batch_sizes, features=lstm_units, time_steps=days_result)
        # Softmax áp dụng trên axis cuối
        a = keras.layers.Dense(time_steps, activation='softmax')(a)  
        
        # Trả về attention score theo thứ tự: (batch_sizes, time_steps=days_result, features=lstm_units)
        a_probs = keras.layers.Permute((2, 1))(a)  
        
        # Tính trọng số của inputs với attention score: (batch_sizes, time_steps=days_result, features=lstm_units)
        output_attention_mul = keras.layers.Multiply()([inputs, a_probs])  
        
        return output_attention_mul

    trend_type_input = keras.layers.Input(shape=(1,), name="trend_type_input")
    x1 = keras.layers.Dense(8, activation='relu')(trend_type_input)
    x1 = keras.layers.RepeatVector(days_result)(x1) # (batch_size, days_result, 8)
    
    ema_9_input = keras.layers.Input(shape=(days_of_ema_9,), name="ema_9_input")
    x3 = keras.layers.Dense(8, activation='relu')(ema_9_input)
    x3 = keras.layers.RepeatVector(days_result)(x3) # (batch_size, days_result, 8)
    
    macd_history_input = keras.layers.Input(shape=(days_of_macd_history,), name="macd_history_input")
    x4 = keras.layers.Dense(8, activation='relu')(macd_history_input)
    x4 = keras.layers.RepeatVector(days_result)(x4) # (batch_size, days_result, 8)
    
    image_input = keras.layers.Input(shape=image_shape, name="image_input") # (batch_size, height, width, channels)
    x2 = keras.layers.Conv2D(8, (2, 2), strides=1, activation='relu')(image_input) # (batch_size, height, width, channels=8)
    x2 = keras.layers.Conv2D(days_result, (2, 2), activation='relu')(x2) # (batch_size, height, width, channels=days_result)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x2)

    x2 = keras.layers.Permute((3, 1, 2))(x2) # (batch_size, channels, height, width)
    x2 = keras.layers.TimeDistributed(keras.layers.Flatten())(x2)
    x2 = keras.layers.TimeDistributed(keras.layers.Dense(64, activation='relu'))(x2)
    x2 = keras.layers.TimeDistributed(keras.layers.Dropout(0.1))(x2) # (batch_size, days_result, 64)
    
    x = keras.layers.concatenate([x3, x4, x1, x2], axis=-1)
    x = keras.layers.TimeDistributed(keras.layers.Dense(64, activation='relu'))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dropout(0.1))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dense(64, activation='relu'))(x)
    
    # LSTM với Attention
    lstm = keras.layers.LSTM(128, activation="relu", return_sequences=True)(x)
    lstm = keras.layers.Dropout(0.1)(lstm)
    lstm = keras.layers.LSTM(128, activation="relu", return_sequences=True)(lstm)
    lstm_out = keras.layers.Dropout(0.1)(lstm) # (batch_sizes, time_steps=days_result, features=lstm_units)
    
    # Áp dụng Attention: (batch_sizes, time_steps=days_result, features=lstm_units)
    attention_mul = attention_3d_block(lstm_out)
    
    # Lớp đầu ra
    output = keras.layers.TimeDistributed(keras.layers.Dense(2, activation='linear'))(attention_mul)
    
    # Xây dựng mô hình
    model = keras.models.Model(inputs=[ema_9_input, macd_history_input, trend_type_input, image_input], outputs=output)
    
    return model, "channels_to_time_step_attention_ema_macd_trend"
