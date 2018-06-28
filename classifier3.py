
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Flatten, Dropout, MaxPooling2D, Conv2D, Dense
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,TensorBoard
import h5py
import timeit
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
with K.tf.device('/gpu: 1'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4, \
                            inter_op_parallelism_threads=4, allow_soft_placement=True,\
                            device_count = {'CPU' : 1, 'GPU' : 1 })
    session = tf.Session(config=config)
    K.set_session(session)


# In[22]:


input_shape=(64,64,3)
num_classes=2


# In[25]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()



# In[27]:
model.save_weights('model2.h5')

# Save Model
model.save('classifier2.h5')
checkpoint=ModelCheckpoint('classifier1.h5',
                            monitor='val_acc',
                            verbose= 1,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='auto',
                            period=1)


start = timeit.default_timer()
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
       target_size=(64, 64),
        batch_size=64,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'validation',
        target_size=(64, 64),
        batch_size=64,
        class_mode='binary')


model = model.fit_generator(
        training_set,
        steps_per_epoch=10,
        epochs=1,
        validation_data=test_set,
        validation_steps=100,
        callbacks=[checkpoint])

end = timeit.default_timer()
print("Time Taken to run the model:",end - start, "seconds") 