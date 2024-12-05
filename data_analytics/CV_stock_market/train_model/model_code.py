import keras


relu = keras.activations.relu

def model_1(image_shape, days_result):
    model = keras.Sequential([
        keras.layers.Input(image_shape),
        keras.layers.Conv2D(16, (2, 2), activation='sigmoid'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        # keras.layers.Conv2D(8, (3, 3), activation='sigmoid'),
        # keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L1()),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L1()),
        keras.layers.Dropout(0.3),
        keras.layers.Reshape((1, 32)),
        keras.layers.LSTM(64, activation='relu', return_sequences=False, kernel_regularizer=keras.regularizers.L1()),
        keras.layers.RepeatVector(days_result),
        keras.layers.LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.L1()),
        keras.layers.LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.L1()),
        keras.layers.TimeDistributed(keras.layers.Dense(2))
    ])
    
    return model, "model_1"


def model_2(image_shape, days_result):
    model = keras.Sequential([
        keras.layers.Input(image_shape),
        # keras.layers.Conv2D(8, (2, 2), activation='sigmoid'),
        # keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        # keras.layers.Conv2D(8, (3, 3), activation='sigmoid'),
        # keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L1()),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.L1()),
        keras.layers.Dropout(0.3),
        # keras.layers.Reshape((1, 32)),
        keras.layers.RepeatVector(days_result),
        keras.layers.LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.L1()),
        # keras.layers.LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.L1()),
        # keras.layers.LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.L1()),
        keras.layers.TimeDistributed(keras.layers.Dense(2))
    ])
    
    return model, "model_2"


def model_3(image_shape, days_result):
    model = keras.Sequential([
        keras.layers.Input(image_shape),
        # keras.layers.Conv2D(8, (2, 2), activation='sigmoid'),
        # keras.layers.MaxPooling2D((2, 2)),
        # keras.layers.Conv2D(8, (3, 3), activation='sigmoid'),
        # keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='sigmoid', kernel_regularizer=keras.regularizers.L1()),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='sigmoid', kernel_regularizer=keras.regularizers.L1()),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='sigmoid', kernel_regularizer=keras.regularizers.L1()),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='sigmoid', kernel_regularizer=keras.regularizers.L1()),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2)
    ])
    
    return model, "model_3"


def model_4(image_shape, days_result):
    model = keras.Sequential([
        keras.layers.Input(image_shape),
        keras.layers.Conv2D(8, (2, 2), activation='sigmoid'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(8, (3, 3), activation='sigmoid'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(68, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(68, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(68, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(68, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2)
    ])
    
    return model, "model_4"


def model_5(image_shape, days_result):
    model = keras.Sequential([
        keras.layers.Input(image_shape),
        keras.layers.Conv2D(8, (2, 2), activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),  # Giảm tỷ lệ Dropout
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.RepeatVector(days_result),
        keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(2))
    ])
    
    return model, "model_5"


def model_5_with_trend_type(image_shape, days_result):
    trend_type_input = keras.layers.Input(shape=(1,), name="trend_type_input")
    x1 = keras.layers.Dense(8, activation='relu')(trend_type_input)
    
    image_input = keras.layers.Input(shape=image_shape, name="image_input")
    
    x2 = keras.layers.Conv2D(8, (2, 2), activation='relu')(image_input)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x2)
    x2 = keras.layers.Conv2D(8, (3, 3), activation='relu')(x2)
    x2 = keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = keras.layers.Flatten()(x2)
    x2 = keras.layers.Dense(32, activation='relu')(x2)
    
    combined = keras.layers.concatenate([x1, x2])
    
    z = keras.layers.Dense(32, activation='relu')(combined)
    z = keras.layers.Dense(32, activation='relu')(z)
    
    input_lstm = keras.layers.RepeatVector(days_result)(z)
    lstm = keras.layers.LSTM(64, activation='tanh', return_sequences=True)(input_lstm)
    output_lstm = keras.layers.TimeDistributed(keras.layers.Dense(2))(lstm)
    
    model = keras.models.Model(inputs=[trend_type_input, image_input], outputs=output_lstm)
    
    return model, "model_5_with_trend_type"


def model_5_with_trend_type_1(image_shape, days_result):
    trend_type_input = keras.layers.Input(shape=(1,), name="trend_type_input")
    x1 = keras.layers.Dense(32, activation='relu')(trend_type_input) # (batch_size, features)
    x1 = keras.layers.Reshape((1, 32))(x1) # (batch_size, 1, features)
    
    image_input = keras.layers.Input(shape=image_shape, name="image_input")
    
    x2 = keras.layers.Conv2D(8, (2, 2), activation='relu')(image_input)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x2)
    x2 = keras.layers.Conv2D(8, (3, 3), activation='relu')(x2)
    x2 = keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = keras.layers.Flatten()(x2)
    x2 = keras.layers.Dense(32, activation='relu')(x2)
    
    z = keras.layers.Dense(32, activation='relu')(x2)
    z = keras.layers.Dropout(0.1)(z)
    z = keras.layers.Dense(32, activation='relu')(z)
    
    input_lstm = keras.layers.RepeatVector(days_result-1)(z) # (batch_size, time_steps, height, width, channels)
    input_lstm = keras.layers.concatenate([x1, input_lstm], axis=1)
    lstm = keras.layers.LSTM(64, activation='tanh', return_sequences=True)(input_lstm)
    lstm = keras.layers.Dropout(0.1)(lstm)
    output_lstm = keras.layers.TimeDistributed(keras.layers.Dense(2))(lstm)
    
    model = keras.models.Model(inputs=[trend_type_input, image_input], outputs=output_lstm)
    
    return model, "model_5_with_trend_type_1"


def model_5_biLSTM(image_shape, days_result):
    model = keras.Sequential([
        keras.layers.Input(image_shape),
        keras.layers.Conv2D(8, (2, 2), activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),  # Giảm tỷ lệ Dropout
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.RepeatVector(days_result),
        keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        keras.layers.Dropout(0.1),
        keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(2))
    ])
    
    return model, "model_5_biLSTM"


def model_6(image_shape, days_result):
    model = keras.Sequential([
        keras.layers.Input(image_shape),
        keras.layers.Conv2D(8, (2, 2), activation='sigmoid'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Conv2D(8, (3, 3), activation='sigmoid'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(68, activation='relu'),
        keras.layers.Dropout(0.1),  # Giảm tỷ lệ Dropout
        keras.layers.Dense(68, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.RepeatVector(days_result),
        keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(2))
    ])
    
    return model, "model_6"


def model_7(image_shape, days_result):
    model = keras.Sequential([
        keras.layers.Input(image_shape),
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2)  # Số lượng đầu ra
    ])
    
    return model, "model_7"


def model_8(image_shape, days_result):
    model = keras.Sequential([
        keras.layers.Input(image_shape),

        keras.layers.TimeDistributed(keras.layers.Conv2D(8, (2, 2), activation='sigmoid')),
        keras.layers.RepeatVector(days_result),
        keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2), strides=(2, 2))),
        keras.layers.TimeDistributed(keras.layers.Flatten()),
        keras.layers.TimeDistributed(keras.layers.Dense(68, activation='relu')),
        keras.layers.TimeDistributed(keras.layers.Dense(8, activation='relu')),
        keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(2)),
    ])
    
    return model, "model_8"


def model_9(image_shape, days_result):
    # 1. Mô hình CNN để trích xuất đặc trưng từ ảnh
    cnn_encoder = keras.models.Sequential([
        keras.layers.Input(shape=image_shape),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu')
    ])

    # 2. Tạo mô hình seq2seq với LSTM
    model = keras.models.Sequential([
        cnn_encoder,
        keras.layers.RepeatVector(days_result),  # Nhân bản đầu ra của CNN thành chuỗi cho từng bước thời gian
        keras.layers.LSTM(64, activation='relu', return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(2))  # 2 giá trị đầu ra cho mỗi ngày (mở cửa, đóng cửa)
    ])

    return model, "model_9"


def model_10(image_shape, days_result):
    # get_max_in_open_close_prices_percent_of_last_days_result_for_multiple_days_result
    
    model = keras.Sequential([
        keras.layers.Input(image_shape),
        keras.layers.Conv2D(8, (2, 2), activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),  # Giảm tỷ lệ Dropout
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.RepeatVector(days_result),
        keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(2))
    ])
    
    return model, "model_10"


def model_11(image_shape, days_result):
    # get_max_in_open_close_prices_percent_of_last_days_result_for_multiple_days_result
    
    model = keras.Sequential([
        keras.layers.Input(image_shape),
        keras.layers.Conv2D(8, (2, 2), activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),  # Giảm tỷ lệ Dropout
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(2)
    ])
    
    return model, "model_11"


def attention_cnn(image_shape, lstm_units=64, days_result=3):

    # Hàm Attention
    def attention_3d_block(inputs, single_attention_vector=False):
        # inputs.shape = (batch_size, time_steps, input_dim)
        time_steps = inputs.shape[1]
        print(inputs.shape)
        
        a = keras.layers.Permute((2, 1))(inputs)  # Chuyển vị ma trận
        a = keras.layers.Dense(time_steps, activation='softmax')(a)  # Tính attention score
        
        a_probs = keras.layers.Permute((2, 1))(a)  # Trả về attention score theo thứ tự
        output_attention_mul = keras.layers.Multiply()([inputs, a_probs])  # Tính trọng số của inputs với attention score
        
        return output_attention_mul

    # Mô hình CNN-LSTM với Attention
    def attention_model(image_shape, lstm_units=64, days_result=3):
        inputs = keras.layers.Input(shape=image_shape)
        
        # Các lớp CNN để trích xuất đặc trưng từ ảnh
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        # Làm phẳng output từ CNN
        x = keras.layers.Flatten()(x)
        
        # Dense layer sau CNN
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        # Lặp lại đặc trưng cho các bước thời gian (RepeatVector)
        x = keras.layers.RepeatVector(days_result)(x)  # days_result là số ngày bạn muốn dự đoán
        
        # LSTM với Attention
        lstm_out = keras.layers.LSTM(lstm_units, return_sequences=True)(x)
        lstm_out = keras.layers.Dropout(0.3)(lstm_out)
        
        # Áp dụng Attention
        attention_mul = attention_3d_block(lstm_out)
        
        # Làm phẳng lại output
        attention_mul = keras.layers.Flatten()(attention_mul)
        
        # Output của mô hình (giá đóng cửa và xu hướng)
        output = keras.layers.Dense(3, activation='sigmoid')(attention_mul)
        
        # Xây dựng mô hình
        model = keras.models.Model(inputs=[inputs], outputs=output)
        
        return model, "attention_model"
    
    return attention_model(image_shape, lstm_units, days_result)


def attention_cnn_2(image_shape, lstm_units=64, days_result=3):

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

    # Mô hình CNN-LSTM với Attention
    inputs = keras.layers.Input(shape=image_shape)
    
    # Các lớp CNN để trích xuất đặc trưng từ ảnh
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    
    # Làm phẳng output từ CNN
    x = keras.layers.Flatten()(x)
    
    # Dense layer sau CNN
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Lặp lại đặc trưng cho các bước thời gian (RepeatVector)
    x = keras.layers.RepeatVector(days_result)(x)  # days_result là số ngày bạn muốn dự đoán
    
    # LSTM với Attention
    lstm_out = keras.layers.LSTM(lstm_units, return_sequences=True)(x)
    lstm_out = keras.layers.Dropout(0.3)(lstm_out) # (batch_sizes, time_steps=days_result, features=lstm_units)
    
    # Áp dụng Attention: (batch_sizes, time_steps=days_result, features=lstm_units)
    attention_mul = attention_3d_block(lstm_out)
    
    # Lớp đầu ra
    output = keras.layers.TimeDistributed(keras.layers.Dense(2, activation='linear'))(attention_mul)
    
    # Xây dựng mô hình
    model = keras.models.Model(inputs=[inputs], outputs=output)
    
    return model, "attention_cnn_2"


def split_cnn(image_shape, days_result):
    inputs = keras.layers.Input(shape=image_shape)
        
    # Các lớp CNN để trích xuất đặc trưng từ ảnh
    x = keras.layers.Conv2D(days_result, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((3, 3))(x)

    a = keras.layers.Permute((3, 1, 2))(x)
    
    flatten = keras.layers.TimeDistributed(keras.layers.Flatten())(a)
    flatten = keras.layers.TimeDistributed(keras.layers.Dense(32))(flatten)
    flatten = keras.layers.TimeDistributed(keras.layers.Dense(32))(flatten)
    lstm_1 = keras.layers.LSTM(64, return_sequences=True)(flatten)
    lstm_1 = keras.layers.TimeDistributed(keras.layers.Dense(2))(lstm_1)
     # Chỉ lấy 3 bước thời gian đầu tiên
    output = keras.layers.Lambda(lambda x: x[:, :days_result, :])(lstm_1)  # Lấy 3 bước đầu
    
    # Xây dựng mô hình
    model = keras.models.Model(inputs=[inputs], outputs=output)
    return model, "split_cnn"


def channels_to_time_step_attention(image_shape, days_result=3, lstm_units=64):

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

    # Mô hình CNN-LSTM với Attention
    inputs = keras.layers.Input(shape=image_shape)
    
    # Các lớp CNN để trích xuất đặc trưng từ ảnh    
    def cnn_layers(inputs):
        inputs = keras.layers.Conv2D(days_result, (3, 3), activation='relu')(inputs)
        # inputs = keras.layers.MaxPooling2D((2, 2))(inputs)
        return inputs
    
    x = cnn_layers(inputs)
    x = keras.layers.Permute((3, 1, 2))(x)
    
    # Làm phẳng output từ CNN
    x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
    x = keras.layers.TimeDistributed(keras.layers.Dense(64, activation='relu'))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dropout(0.1))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dense(64, activation='relu'))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dropout(0.1))(x)
    
    # LSTM với Attention
    lstm_out = keras.layers.LSTM(lstm_units, activation="tanh", return_sequences=True)(x)
    lstm_out = keras.layers.Dropout(0.1)(lstm_out) # (batch_sizes, time_steps=days_result, features=lstm_units)
    
    # Áp dụng Attention: (batch_sizes, time_steps=days_result, features=lstm_units)
    attention_mul = attention_3d_block(lstm_out)
    
    # Lớp đầu ra
    output = keras.layers.TimeDistributed(keras.layers.Dense(2, activation='relu'))(attention_mul)
    
    # Xây dựng mô hình
    model = keras.models.Model(inputs=[inputs], outputs=output)
    
    return model, "channels_to_time_step_attention"


def attention_cnn_ema_macd_trend(image_shape, days_result, days_of_ema_9=2, days_of_macd_history=2, lstm_units=128):

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
    
    image_input = keras.layers.Input(shape=image_shape, name="image_input")
    x2 = keras.layers.Conv2D(8, (2, 2), activation='relu')(image_input)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x2)
    x2 = keras.layers.Flatten()(x2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)
    x2 = keras.layers.Dropout(0.1)(x2)
    
    ema_9_input = keras.layers.Input(shape=(days_of_ema_9,), name="ema_9_input")
    x3 = keras.layers.Dense(8, activation='relu')(ema_9_input)
    
    macd_history_input = keras.layers.Input(shape=(days_of_macd_history,), name="macd_history_input")
    x4 = keras.layers.Dense(8, activation='relu')(macd_history_input)
    
    combined = keras.layers.concatenate([x3, x4, x1, x2])
    
    z = keras.layers.Dense(64, activation='relu')(combined)
    z = keras.layers.Dropout(0.1)(z)
    z = keras.layers.Dense(64, activation='relu')(z)
    
    # Lặp lại đặc trưng cho các bước thời gian (RepeatVector)
    x = keras.layers.RepeatVector(days_result)(z)  # days_result là số ngày bạn muốn dự đoán
    
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, activation="tanh", return_sequences=True))(x)
    lstm = keras.layers.Dropout(0.1)(lstm)
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, activation="tanh", return_sequences=True))(lstm)
    lstm_out = keras.layers.Dropout(0.1)(lstm) # (batch_sizes, time_steps=days_result, features=lstm_units)
    
    # Áp dụng Attention: (batch_sizes, time_steps=days_result, features=lstm_units)
    attention_mul = attention_3d_block(lstm_out)
    
    # Lớp đầu ra
    output = keras.layers.TimeDistributed(keras.layers.Dense(2, activation='linear'))(attention_mul)
    
    # Xây dựng mô hình
    model = keras.models.Model(inputs=[ema_9_input, macd_history_input, trend_type_input, image_input], outputs=output)
    
    return model, "attention_cnn_ema_macd_trend"


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
    x2 = keras.layers.Conv2D(days_result, (2, 2), activation='relu')(image_input) # (batch_size, height, width, channels=days_result)
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
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, activation="relu", return_sequences=True))(x)
    lstm = keras.layers.Dropout(0.1)(lstm)
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, activation="relu", return_sequences=True))(lstm)
    lstm_out = keras.layers.Dropout(0.1)(lstm) # (batch_sizes, time_steps=days_result, features=lstm_units)
    
    # Áp dụng Attention: (batch_sizes, time_steps=days_result, features=lstm_units)
    attention_mul = attention_3d_block(lstm_out)
    
    # Lớp đầu ra
    output = keras.layers.TimeDistributed(keras.layers.Dense(2, activation='linear'))(attention_mul)
    
    # Xây dựng mô hình
    model = keras.models.Model(inputs=[ema_9_input, macd_history_input, trend_type_input, image_input], outputs=output)
    
    return model, "channels_to_time_step_attention_ema_macd_trend"

def channels_to_time_step_attention_v2_ema_macd_trend(image_shape, days_result, days_of_ema_9=2, days_of_macd_history=2):

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
    x2 = keras.layers.Conv2D(days_result, (2, 2), activation='relu')(image_input) # (batch_size, height, width, channels=days_result)
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
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, activation="relu", return_sequences=True))(x)
    lstm = keras.layers.Dropout(0.1)(lstm)
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, activation="relu", return_sequences=True))(lstm)
    lstm_out = keras.layers.Dropout(0.1)(lstm) # (batch_sizes, time_steps=days_result, features=lstm_units)
    
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, activation="relu", return_sequences=True))
    
    # Áp dụng Attention: (batch_sizes, time_steps=days_result, features=lstm_units)
    attention_mul = attention_3d_block(lstm_out)
    
    # Lớp đầu ra
    output = keras.layers.TimeDistributed(keras.layers.Dense(2, activation='linear'))(attention_mul)
    
    # Xây dựng mô hình
    model = keras.models.Model(inputs=[ema_9_input, macd_history_input, trend_type_input, image_input], outputs=output)
    
    return model, "channels_to_time_step_attention_v2_ema_macd_trend"


def channels_to_time_step_attention_ema_macd_trend(image_shape, days_result, days_of_ema_9=2, days_of_macd_history=2):

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
    
    attention_output = keras.layers.Attention()([x, x])
    
    
    
    # Xây dựng mô hình
    model = keras.models.Model(inputs=[ema_9_input, macd_history_input, trend_type_input, image_input], outputs=output)
    
    return model, "channels_to_time_step_attention_ema_macd_trend"
