import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, 
    TimeDistributed, LSTM, Dense, Dropout, Flatten,
    GlobalAveragePooling2D, Reshape, Bidirectional,
    ConvLSTM2D, RepeatVector, AveragePooling2D, Activation,
    GlobalMaxPooling2D, concatenate, Add, Multiply
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.constraints import MaxNorm
import warnings
warnings.filterwarnings('ignore')

# Set memory growth untuk GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



class StandardCNNModel:
    """
    SIMPLIFIED CNN Architecture for SMALL DATASET (683 samples)
    - Only 2 Convolutional Blocks (was 4 - TOO DEEP!)
    - Lighter regularization for small data
    - Reduced parameters to prevent overfitting
    """
    
    def __init__(self, input_shape=(64, 64, 1), num_classes=4, 
                 num_filters=32, dropout_rate=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self):
        """Build SHALLOW CNN Model - 2 blocks only for small dataset"""
        inputs = Input(shape=self.input_shape, name='spectrogram_input')
        
        # === BLOCK 1: Low-level features ===
        x1 = Conv2D(self.num_filters, (5, 5), padding='same', 
                   kernel_regularizer=l1_l2(l1=0.00001, l2=0.00001),
                   kernel_initializer=HeNormal(),
                   kernel_constraint=MaxNorm(3))(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(self.num_filters, (5, 5), padding='same',
                   kernel_regularizer=l1_l2(l1=0.00001, l2=0.00001),
                   kernel_constraint=MaxNorm(3))(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D((2, 2))(x1)  # 64x64 -> 32x32
        x1 = Dropout(0.2)(x1)
        
        # === BLOCK 2: High-level features ===
        x2 = Conv2D(self.num_filters * 2, (3, 3), padding='same',
                   kernel_regularizer=l1_l2(l1=0.00001, l2=0.00001),
                   kernel_constraint=MaxNorm(3))(x1)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv2D(self.num_filters * 2, (3, 3), padding='same',
                   kernel_regularizer=l1_l2(l1=0.00001, l2=0.00001),
                   kernel_constraint=MaxNorm(3))(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling2D((2, 2))(x2)  # 32x32 -> 16x16
        x2 = Dropout(0.25)(x2)
        
        # Convolutional Block 3 (128 filters)
        x3 = Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=l1_l2(l1=0.00001, l2=0.00001),
                   kernel_constraint=MaxNorm(3))(x2)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = MaxPooling2D((2, 2))(x3)  # 16x16 -> 8x8
        x3 = Dropout(0.3)(x3)
        
        # === GLOBAL POOLING (no more convolutions) ===
        x_global = GlobalAveragePooling2D()(x3)
        
        # === SIMPLIFIED DENSE LAYERS ===
        # Dense layer 1
        d1 = Dense(128, kernel_regularizer=l1_l2(l1=0.00001, l2=0.00001),
                  kernel_initializer=HeNormal(),
                  kernel_constraint=MaxNorm(3))(x_global)
        d1 = BatchNormalization()(d1)
        d1 = Activation('relu')(d1)
        d1 = Dropout(0.35)(d1)
        
        # Dense layer 2
        d2 = Dense(64, kernel_regularizer=l1_l2(l1=0.00001, l2=0.00001),
                  kernel_initializer=HeNormal(),
                  kernel_constraint=MaxNorm(3))(d1)
        d2 = BatchNormalization()(d2)
        d2 = Activation('relu')(d2)
        d2 = Dropout(0.3)(d2)
        
        # === OUTPUT LAYER ===
        outputs = Dense(self.num_classes, activation='softmax')(d2)
        
        self.model = Model(inputs, outputs, name='ImprovedCNN_3Blocks')
        return self.model
    
    def compile_model(self, learning_rate=0.001, optimizer_type='adam'):
        """Compile model dengan learning rate HIGHER untuk small dataset convergence"""
        # For small dataset (683 samples), use HIGHER lr for faster convergence
        if optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, 
                           momentum=0.9, 
                           nesterov=True,
                           clipnorm=1.0)  # Gradient clipping
        elif optimizer_type == 'adam':
            optimizer = Adam(learning_rate=learning_rate, 
                            beta_1=0.9, 
                            beta_2=0.999,
                            clipnorm=1.0)
        else:
            optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )



class StandardLSTMModel:
    """
    OPTIMIZED LSTM Architecture untuk EMG Sequence Classification
    - 2 LSTM layers dengan residual connections
    - BatchNorm untuk stabil training
    """
    
    def __init__(self, input_shape, num_classes=4, lstm_units=128, dropout_rate=0.4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self):
        """Build Optimized LSTM Model"""
        # Handle input shape
        if isinstance(self.input_shape, (list, tuple)):
            if len(self.input_shape) == 2:  # (64, 64) spectrogram
                actual_input_shape = (64 * 64,)  # Flatten untuk LSTM
            elif len(self.input_shape) == 1:  # Already flattened
                actual_input_shape = self.input_shape
            else:
                actual_input_shape = (64 * 64,)
        else:
            actual_input_shape = (64 * 64,)
        
        inputs = Input(shape=actual_input_shape, name='sequence_input')
        
        # Reshape untuk LSTM: (batch, timesteps, features)
        # Spectrogram flattened -> reshape menjadi 64 timesteps dengan 64 features
        x = Reshape((64, 64))(inputs)
        
        # === CONV1D untuk feature extraction awal ===
        x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # === LSTM BLOCK 1 ===
        lstm1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                                 recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                                 kernel_constraint=MaxNorm(3),
                                 recurrent_constraint=MaxNorm(3),
                                 name='lstm1'))(x)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        
        # === LSTM BLOCK 2 ===
        lstm2 = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                                 recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                                 kernel_constraint=MaxNorm(3),
                                 recurrent_constraint=MaxNorm(3),
                                 name='lstm2'))(lstm1)
        lstm2 = Dropout(self.dropout_rate)(lstm2)
        
        # === LSTM BLOCK 3 (final) ===
        lstm3 = Bidirectional(LSTM(self.lstm_units, return_sequences=False,
                                 kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                                 recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                                 kernel_constraint=MaxNorm(3),
                                 recurrent_constraint=MaxNorm(3),
                                 name='lstm3'))(lstm2)
        lstm3 = Dropout(self.dropout_rate)(lstm3)
        
        # === MULTI-SCALE FEATURE EXTRACTION ===
        # Average pooling dari intermediate layers
        avg1 = tf.keras.layers.GlobalAveragePooling1D()(lstm1)
        avg2 = tf.keras.layers.GlobalAveragePooling1D()(lstm2)
        
        # Gabungkan semua features
        fused = concatenate([avg1, avg2, lstm3])
        
        # === DENSE LAYERS ===
        # Dense layer 1
        d1 = Dense(256, activation='relu', 
                  kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                  kernel_initializer=HeNormal(),
                  kernel_constraint=MaxNorm(3))(fused)
        d1 = BatchNormalization()(d1)
        d1 = Dropout(0.5)(d1)
        
        # Dense layer 2
        d2 = Dense(128, activation='relu',
                  kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                  kernel_constraint=MaxNorm(3))(d1)
        d2 = BatchNormalization()(d2)
        d2 = Dropout(0.4)(d2)
        
        # Dense layer 3
        d3 = Dense(64, activation='relu',
                  kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                  kernel_constraint=MaxNorm(3))(d2)
        d3 = Dropout(0.3)(d3)
        
        # === OUTPUT LAYER ===
        outputs = Dense(self.num_classes, activation='softmax', name='output')(d3)
        
        self.model = Model(inputs, outputs, name='OptimizedLSTM_3Blocks')
        return self.model
    
    def compile_model(self, learning_rate=0.005, optimizer_type='adam'):
        """Compile model dengan LR tinggi"""
        if optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, 
                           momentum=0.9, 
                           nesterov=True,
                           clipnorm=1.0)
        elif optimizer_type == 'adam':
            optimizer = Adam(learning_rate=learning_rate, 
                            beta_1=0.9, 
                            beta_2=0.999,
                            clipnorm=1.0)
        else:
            optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )



class StandardCNNLSTMModel:
    """
    OPTIMIZED CNN-LSTM Hybrid Architecture
    - 3 CNN blocks untuk spatial feature extraction
    - 2 LSTM blocks untuk temporal modeling
    """
    
    def __init__(self, input_shape=(64, 64, 1), num_classes=4,
                 num_filters=32, lstm_units=64, dropout_rate=0.5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self):
        """Build Optimized CNN-LSTM Model"""
        inputs = Input(shape=self.input_shape, name='spectrogram_input')
        
        # === CNN BLOCK 1 ===
        c1 = Conv2D(self.num_filters, (5, 5), padding='same',
                   kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                   kernel_initializer=HeNormal(),
                   kernel_constraint=MaxNorm(3))(inputs)
        c1 = BatchNormalization()(c1)
        c1 = Activation('relu')(c1)
        c1 = Conv2D(self.num_filters, (5, 5), padding='same',
                   kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                   kernel_constraint=MaxNorm(3))(c1)
        c1 = BatchNormalization()(c1)
        c1 = Activation('relu')(c1)
        c1 = MaxPooling2D((2, 2))(c1)  # 64x64 -> 32x32
        c1 = Dropout(0.3)(c1)
        
        # === CNN BLOCK 2 ===
        c2 = Conv2D(self.num_filters * 2, (3, 3), padding='same',
                   kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                   kernel_constraint=MaxNorm(3))(c1)
        c2 = BatchNormalization()(c2)
        c2 = Activation('relu')(c2)
        c2 = Conv2D(self.num_filters * 2, (3, 3), padding='same',
                   kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                   kernel_constraint=MaxNorm(3))(c2)
        c2 = BatchNormalization()(c2)
        c2 = Activation('relu')(c2)
        c2 = MaxPooling2D((2, 2))(c2)  # 128x128 -> 64x64
        c2 = Dropout(0.4)(c2)
        
        # === CNN BLOCK 3 ===
        c3 = Conv2D(self.num_filters * 4, (3, 3), padding='same',
                   kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                   kernel_constraint=MaxNorm(3))(c2)
        c3 = BatchNormalization()(c3)
        c3 = Activation('relu')(c3)
        c3 = Conv2D(self.num_filters * 4, (3, 3), padding='same',
                   kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                   kernel_constraint=MaxNorm(3))(c3)
        c3 = BatchNormalization()(c3)
        c3 = Activation('relu')(c3)
        c3 = GlobalAveragePooling2D()(c3)  # 64x64 -> vector
        c3 = Dropout(0.5)(c3)
        
        # Reshape untuk LSTM: (batch, timesteps, features)
        # CNN features sebagai sequence dengan 1 timestep
        x = Reshape((1, -1))(c3)
        
        # === LSTM BLOCK 1 ===
        lstm1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                                 recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                                 kernel_constraint=MaxNorm(3),
                                 recurrent_constraint=MaxNorm(3)))(x)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        
        # === LSTM BLOCK 2 ===
        lstm2 = Bidirectional(LSTM(self.lstm_units, return_sequences=False,
                                 kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                                 recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                                 kernel_constraint=MaxNorm(3),
                                 recurrent_constraint=MaxNorm(3)))(lstm1)
        lstm2 = Dropout(self.dropout_rate)(lstm2)
        
        # === DENSE LAYERS ===
        # Dense layer 1
        d1 = Dense(256, activation='relu', 
                  kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                  kernel_initializer=HeNormal(),
                  kernel_constraint=MaxNorm(3))(lstm2)
        d1 = BatchNormalization()(d1)
        d1 = Dropout(0.5)(d1)
        
        # Dense layer 2
        d2 = Dense(128, activation='relu',
                  kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
                  kernel_constraint=MaxNorm(3))(d1)
        d2 = BatchNormalization()(d2)
        d2 = Dropout(0.4)(d2)
        
        # Output
        outputs = Dense(self.num_classes, activation='softmax')(d2)
        
        self.model = Model(inputs, outputs, name='OptimizedCNNLSTM_3CNN_2LSTM')
        return self.model
    
    def compile_model(self, learning_rate=0.005, optimizer_type='adam'):
        """Compile model"""
        if optimizer_type == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, 
                           momentum=0.9, 
                           nesterov=True,
                           clipnorm=1.0)
        elif optimizer_type == 'adam':
            optimizer = Adam(learning_rate=learning_rate, 
                            beta_1=0.9, 
                            beta_2=0.999,
                            clipnorm=1.0)
        else:
            optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )



class ModelBuilder:
    """Factory untuk membuat model - OPTIMIZED untuk jurnal"""
    
    @staticmethod
    def build_standard_cnn(input_shape=(64, 64, 1), num_classes=4, **kwargs):
        """Build Standard CNN dengan 4 blocks"""
        model = StandardCNNModel(input_shape, num_classes, **kwargs)
        model.build_model()
        model.compile_model()
        return model
    
    @staticmethod
    def build_standard_lstm(input_shape, num_classes=4, **kwargs):
        """Build Standard LSTM dengan 3 blocks"""
        model = StandardLSTMModel(input_shape, num_classes, **kwargs)
        model.build_model()
        model.compile_model()
        return model
    
    @staticmethod
    def build_standard_cnn_lstm(input_shape=(64, 64, 1), num_classes=4, **kwargs):
        """Build CNN-LSTM Hybrid dengan 3 CNN + 2 LSTM"""
        model = StandardCNNLSTMModel(input_shape, num_classes, **kwargs)
        model.build_model()
        model.compile_model()
        return model