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



