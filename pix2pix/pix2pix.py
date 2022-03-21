import tensorflow as tf

from matplotlib import pyplot as plt

import os
import time
import datetime

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DATASET_PATH = './images/dataset/pix2pix/'
CHECKPOINT_PATH = './checkpoints/pix2pix'

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_W = 256
IMG_H = 256
OUTPUT_CHANNELS = 3

LAMBDA = 100

EPOCHS = 1000

def loadImg(imageFileName):
    img = tf.io.read_file(imageFileName)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)

    imgW = tf.shape(img)[1] // 2

    inputImg = img[:, imgW:, :]
    truthImg = img[:, :imgW, :]

    return inputImg, truthImg

def resize256(img):
    img = tf.image.resize(img, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img

def resize286(img):
    img = tf.image.resize(img, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img

def normalize(img):
    img = (img / 127.5) - 1
    return img

@tf.function()
def random_jitter(inputImg, truthImg):
    inputImg = resize286(inputImg)
    truthImg = resize286(truthImg)

    stackedImg = tf.stack([inputImg, truthImg], axis=0)
    croppedImg = tf.image.random_crop(stackedImg, size=[2, IMG_H, IMG_W, 3])

    inputImg = croppedImg[0]
    truthImg = croppedImg[1]
 
    if tf.random.uniform(()) > 0.5:
        inputImg = tf.image.flip_left_right(inputImg)
        truthImg = tf.image.flip_left_right(truthImg)
    return inputImg, truthImg

def loadTrainData(image_file):
    inputImg, truthImg = loadImg(image_file)
    input_image, real_image = random_jitter(inputImg, truthImg)
    inputImg = normalize(inputImg)
    truthImg = normalize(truthImg)

    return input_image, real_image

def loadTestData(image_file):
    inputImg, truthImg = loadImg(image_file)
    inputImg = resize256(inputImg)
    truthImg = resize256(truthImg)
    inputImg = normalize(inputImg)
    truthImg = normalize(truthImg)

    return inputImg, truthImg

class Encoder(tf.keras.models.Model):
    def __init__(self):
        def block(filters, size):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=False))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU())
            return model

        def blockNonBatchNorm(filters, size):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=False))
            model.add(tf.keras.layers.LeakyReLU())

            return model

        super().__init__(self)

        self.blocks = []
        self.blocks.append(blockNonBatchNorm(64, 4))
        self.blocks.append(block(128, 4)) 
        self.blocks.append(block(256, 4))
        self.blocks.append(block(512, 4))
        self.blocks.append(block(512, 4))
        self.blocks.append(block(512, 4))
        self.blocks.append(block(512, 4))
        self.blocks.append(block(512, 4))
    
    def call(self, x):
        skips = []
        for encoderLayer in self.blocks:
            x = encoderLayer(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        return skips, x

class Decoder(tf.keras.models.Model):
    def __init__(self):
        def block(filters, size):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.ReLU())

            return model

        def blockNonDropOut(filters, size):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())

            return model

        super().__init__(self)

        self.blocks = []
        self.blocks.append(block(512, 4))
        self.blocks.append(block(512, 4))
        self.blocks.append(block(512, 4))
        self.blocks.append(blockNonDropOut(512, 4))
        self.blocks.append(blockNonDropOut(256, 4))
        self.blocks.append(blockNonDropOut(128, 4))
        self.blocks.append(blockNonDropOut(64, 4))
    
    def call(self, skips, x):
        for decoderLayer, skip in zip(self.blocks, skips):
            x = decoderLayer(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        return x

class Generator(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer = tf.random_normal_initializer(0., 0.02),
                                        activation='tanh')

    def call(self, inputs):
        skips , x = self.enc(inputs)
        outputs = self.dec(skips, x)
        outputs = self.last(outputs)

        return outputs

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

class Discriminator(tf.keras.models.Model):
    def __init__(self):
        def block(filters, size):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=False))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU())
            return model

        def blockNonBatchNorm(filters, size):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=False))
            model.add(tf.keras.layers.LeakyReLU())

            return model

        super().__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.blocks = []
        self.blocks.append(blockNonBatchNorm(64, 4))
        self.blocks.append(block(128, 4))
        self.blocks.append(block(256, 4))
        self.blocks.append(tf.keras.layers.ZeroPadding2D())
        self.blocks.append(tf.keras.layers.Conv2D(512, 4, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False))
        self.blocks.append(tf.keras.layers.BatchNormalization())
        self.blocks.append(tf.keras.layers.LeakyReLU())
        self.blocks.append(tf.keras.layers.ZeroPadding2D())
        self.blocks.append(tf.keras.layers.Conv2D(1, 4, strides=1,
                                        kernel_initializer=initializer))

    def call(self, inputs):
        x = tf.keras.layers.concatenate(inputs)
        for discLayer in self.blocks:
            x = discLayer(x)

        return x

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generate_images(model, test_input, tar, num):
    prediction = model(test_input, training=True)
    resultFig = plt.figure(figsize=(15,15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    resultFig.savefig(DATASET_PATH + 'result/result' + str(num) + '.png')

def generate_image_fromPicture(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def train():
    inputImg, truthImg = loadImg(DATASET_PATH+'test/content0.jpg')

    plt.figure()
    plt.imshow(inputImg/255.0)
    plt.figure()
    plt.imshow(truthImg/255.0)

    train_dataset = tf.data.Dataset.list_files(DATASET_PATH + 'train/*.jpg')
    train_dataset = train_dataset.map(loadTrainData,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(DATASET_PATH + 'test/*.jpg')
    test_dataset = test_dataset.map(loadTestData)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    inputs = tf.keras.layers.Input(shape=[256,256,3])
    outputs = Generator()(inputs)
    generator = tf.keras.models.Model(inputs = inputs, outputs = outputs)

    tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
    gen_output = generator(inputImg[tf.newaxis,...], training=False)
    #plt.imshow(gen_output[0,...])
    
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    outputs = Discriminator()([inp, tar])
    discriminator = tf.keras.Model(inputs=[inp, tar], outputs = outputs)
    tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

    print(gen_output.shape)
    disc_out = discriminator([inp, gen_output], training=False)
    #plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
    #plt.colorbar()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    checkpoint_prefix = os.path.join(CHECKPOINT_PATH, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    for example_input, example_target in test_dataset.take(1):
        generate_images(generator, example_input, example_target, 0)

    @tf.function
    def train_step(input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

    def fit(train_ds, epochs, test_ds):
        for epoch in range(epochs):
            start = time.time()

            #display.clear_output(wait=True)
            print("Epoch: ", epoch)

            # Train
            for n, (input_image, target) in train_ds.enumerate():
                print('.', end='')
                if (n+1) % 100 == 0:
                    print()
                train_step(input_image, target, epoch)
            print()

            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 100 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
                for example_input, example_target in test_dataset.take(1):
                    generate_images(generator, example_input, example_target, epoch)

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                time.time()-start))

        checkpoint.save(file_prefix = checkpoint_prefix)

    fit(train_dataset, EPOCHS, test_dataset)

    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))
    for inp, tar in test_dataset.take(50):
        generate_images(generator, inp, tar)

if __name__ == '__main__':
    train()