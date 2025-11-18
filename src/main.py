# Импорт необходимых библиотек
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Часть 1: Загрузка и подготовка данных Fashion-MNIST (уменьшенный набор)
print("Загрузка данных...")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Берем только часть данных для быстрого обучения
TRAIN_SAMPLES = 3000  # Еще меньше для скорости
TEST_SAMPLES = 500

x_train = x_train[:TRAIN_SAMPLES]
y_train = y_train[:TRAIN_SAMPLES]
x_test = x_test[:TEST_SAMPLES]
y_test = y_test[:TEST_SAMPLES]

# Нормализация и преобразование формы данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Добавление размерности канала
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Форма тренировочных данных: {x_train.shape}")
print(f"Форма тестовых данных: {x_test.shape}")

# Визуализация примеров исходных данных
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 8))
for i in range(12):  # Покажем 12 примеров
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Часть 2: Реализация VAE (исправленная версия)

# Параметры модели
latent_dim = 2

# Слой Sampling с reparameterization trick
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Энкодер
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Декодер
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 32, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 32))(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Полная модель VAE с исправлениями
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Добавляем KL divergence к модели для возможности использования в loss
        self.add_loss(self._calculate_kl_loss(z_mean, z_log_var))
        return reconstructed

    def _calculate_kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        return kl_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            # Прямой проход через encoder и decoder
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Reconstruction loss (MSE вместо binary_crossentropy для стабильности)
            reconstruction_loss = tf.reduce_mean(
                tf.square(data - reconstruction)
            )
            reconstruction_loss *= 28 * 28  # Масштабирование

            # KL divergence loss
            kl_loss = self._calculate_kl_loss(z_mean, z_log_var)

            # Total loss
            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Создание и компиляция модели VAE
vae = VAE(encoder, decoder)
vae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    run_eagerly=True
)

# Вывод архитектуры
print("Архитектура энкодера:")
encoder.summary()
print("\nАрхитектура декодера:")
decoder.summary()

# Часть 3: Обучение модели (быстрое)
print("Начало быстрого обучения...")
history = vae.fit(
    x_train,
    epochs=5,  # Еще меньше эпох
    batch_size=32,  # Еще меньше batch_size
    validation_data=(x_test,),
    verbose=1
)

# Визуализация процесса обучения
plt.figure(figsize=(12, 4))
if 'loss' in history.history:
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.legend()

if 'reconstruction_loss' in history.history:
    plt.subplot(1, 3, 2)
    plt.plot(history.history['reconstruction_loss'], label='Training Recon Loss')
    if 'val_reconstruction_loss' in history.history:
        plt.plot(history.history['val_reconstruction_loss'], label='Validation Recon Loss')
    plt.title('Reconstruction Loss')
    plt.legend()

if 'kl_loss' in history.history:
    plt.subplot(1, 3, 3)
    plt.plot(history.history['kl_loss'], label='Training KL Loss')
    if 'val_kl_loss' in history.history:
        plt.plot(history.history['val_kl_loss'], label='Validation KL Loss')
    plt.title('KL Loss')
    plt.legend()

plt.tight_layout()
plt.show()

# Часть 4: Визуализация латентного пространства

print("Получение латентных представлений...")
z_mean, z_log_var, z = vae.encoder.predict(x_test[:200], verbose=0)  # Только 200 примеров

# Визуализация латентного пространства
plt.figure(figsize=(10, 8))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test[:200], cmap='tab10', alpha=0.7)
plt.colorbar()
plt.xlabel('z[0]')
plt.ylabel('z[1]')
plt.title('Латентное пространство VAE')
plt.show()

# Часть 5: Генерация новых изображений
def plot_latent_space(decoder, n=15, figsize=8):  # Еще меньше для скорости
    digit_size = 28
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n))

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap="Greys_r")
    plt.title("Сгенерированные изображения")
    plt.axis('off')
    plt.show()

print("Генерация изображений...")
plot_latent_space(vae.decoder)

# Часть 6: Интерполяция в латентном пространстве
def interpolate_images(start_idx, end_idx, n_steps=6):  # Уменьшили шаги
    start_z = z_mean[start_idx:start_idx+1]
    end_z = z_mean[end_idx:end_idx+1]

    interpolated_z = []
    for alpha in np.linspace(0, 1, n_steps):
        interpolated_z.append(alpha * end_z + (1 - alpha) * start_z)

    interpolated_z = np.vstack(interpolated_z)
    interpolated_images = vae.decoder.predict(interpolated_z, verbose=0)

    plt.figure(figsize=(12, 2))
    for i, img in enumerate(interpolated_images):
        plt.subplot(1, n_steps, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')

    start_class = class_names[y_test[start_idx]]
    end_class = class_names[y_test[end_idx]]
    plt.suptitle(f'Интерполяция: {start_class} → {end_class}')
    plt.tight_layout()
    plt.show()

print("Демонстрация интерполяции...")
interpolate_images(0, 3)
interpolate_images(5, 8)

# Часть 7: Сравнение реконструкций
print("Сравнение оригиналов и реконструкций...")
test_samples = x_test[:6]
reconstructions = vae.predict(test_samples, verbose=0)

plt.figure(figsize=(12, 3))
for i in range(6):
    # Оригинал
    plt.subplot(2, 6, i + 1)
    plt.imshow(test_samples[i].squeeze(), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Реконструкция
    plt.subplot(2, 6, i + 7)
    plt.imshow(reconstructions[i].squeeze(), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА!")
print(f"Использовано данных: {TRAIN_SAMPLES} тренировочных")
print(f"Размер латентного пространства: {latent_dim}")
print(f"Количество эпох: 5")
print("="*50)