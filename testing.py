
# coding: utf-8

# In[59]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,TensorBoard
import h5py
import timeit
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
import tensorflow as tf
with K.tf.device('/gpu: 1'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4, \
                            inter_op_parallelism_threads=4, allow_soft_placement=True,\
                            device_count = {'CPU' : 1, 'GPU' : 1 })
    session = tf.Session(config=config)
    K.set_session(session)

# In[60]:


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
#classifier.summary()



# In[71]:


#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.load_weights('model.h5')


# In[72]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[79]:


test_set = test_datagen.flow_from_directory('dataset/dataset/test_set',
                                           target_size = (64, 64),
                                            batch_size = 10,
                                           class_mode = 'binary')

classifier.save_weights('testing_weights.h5')
# In[80]:
# Save Model
classifier.save('classifier1.h5')
checkpoint=ModelCheckpoint('testing.h5',
                            monitor='val_acc',
                            verbose= 1,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='auto',
                            period=1)



X,y=test_set.next()


# In[81]:


classifier.predict_classes(X,batch_size=10,verbose=1)


# In[82]:


test_set = test_datagen.flow_from_directory('dataset/dataset/test_set',
                                           target_size = (64, 64),
                                            batch_size = 10,
                                           class_mode = 'binary')


# In[83]:


X,y=test_set.next()


# In[84]:


classifier.predict_classes(X,batch_size=10,verbose=1)

