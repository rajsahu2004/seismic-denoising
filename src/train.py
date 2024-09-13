import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D # type: ignore
import matplotlib.pyplot as plt
print(tf.__version__)
# Check if GPUs are available
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

# ---------------- Helper Functions ---------------- #
def load_and_preprocess_data(img_path, noise_path):
    img = np.load(img_path, allow_pickle=True)
    noise = np.load(noise_path, allow_pickle=True)
    
    # Process shapes and rescale
    img = check_shape(img)
    noise = check_shape(noise)
    img = rescale_volume(img, 1, 99)
    noise = rescale_volume(noise, 1, 99)
    
    # Cast to float32
    img = img.astype(np.float32)
    noise = noise.astype(np.float32)
    
    return img, noise

def check_shape(npy_file):
    if npy_file.shape != (1259, 300, 300):
        npy_file = npy_file.T
    return npy_file

def rescale_volume(seismic, low=0, high=100):
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255
    return seismic

# ---------------- Dataset Creation ---------------- #
def create_dataset(pairs, batch_size=8, shuffle_buffer_size=100):
    img_paths, noise_paths = zip(*pairs)
    
    dataset = tf.data.Dataset.from_tensor_slices((list(img_paths), list(noise_paths)))
    
    def _load_data(img_path, noise_path):
        img_slices, noise_slices = tf.numpy_function(load_and_preprocess_data, [img_path, noise_path], [tf.float32, tf.float32])
        
        img_slices.set_shape([None, 300, 300])
        noise_slices.set_shape([None, 300, 300])
        
        # Return slices
        img_ds = tf.data.Dataset.from_tensor_slices(img_slices)
        noise_ds = tf.data.Dataset.from_tensor_slices(noise_slices)
        
        return tf.data.Dataset.zip((img_ds, noise_ds))

    # Apply _load_data and split into individual slices
    dataset = dataset.flat_map(_load_data)
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ---------------- Model Definition ---------------- #
def build_autoencoder(input_shape=(300, 300, 1)):
    input_img = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

# ---------------- Training Function ---------------- #
def train_model(train_data, val_data, epochs=50):
    model = build_autoencoder()
    model.summary()
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    # Save the model
    model.save('seismic_denoising_model.h5')

    # Plot loss curves
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.savefig('training_loss_curve.png')

# ---------------- Main Script ---------------- #
if __name__ == "__main__":
    # Define file pairs (img_path, noise_path)
    file_pairs = [
        ('data/training_data/77695365/seismicCubes_RFC_fullstack_2023.77695365.npy', 'data/training_data/77695365/seismic_w_noise_vol_77695365.npy'),
        ('data/training_data/76135802/seismicCubes_RFC_fullstack_2023.76135802.npy', 'data/training_data/76135802/seismic_w_noise_vol_76135802.npy'),
        # Add all your file pairs here...
    ]

    # Split data into training and validation sets
    train_pairs, val_pairs = train_test_split(file_pairs, test_size=0.2)

    # Create datasets
    train_dataset = create_dataset(train_pairs, batch_size=8)
    val_dataset = create_dataset(val_pairs, batch_size=8)
    
    for img, noise in train_dataset.take(1):
        print("Image shape:", img.shape)
        print("Noise shape:", noise.shape)

    # Train the model
    train_model(train_dataset, val_dataset, epochs=50)