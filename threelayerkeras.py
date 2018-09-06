import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
from skimage import data
import dlib

class_names = ['narrow', 'wide']

def displaytrainhistory(history):
    history_dict = history.history
    plt.clf()   # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()    

def displaytestdata(test_images, test_labels, predictions, origin_test_imgs):
    fig = plt.figure(figsize=(12, 48))
    num = len(test_images)
    if num>35:
        num = 35
    for i in range(len(test_images)):
        truth = test_labels[i]
        prediction = np.argmax(predictions[i])
        plt.subplot(6, 6, 1+i)
        plt.axis('off')
        color='green' if truth == prediction else 'red'
        # plt.text(40, 10, "Truth:        {}\nPrediction:    {}\n probability: {:0.2f}%".format(class_names[truth], class_names[prediction], 100*np.max(predictions[i])), 
                # fontsize=12, color=color)
        plt.text(40, -10, "{}({:0.2f}%)".format(class_names[prediction],100*np.max(predictions[i])), 
                fontsize=10, color=color)        
        plt.imshow(origin_test_imgs[i],  cmap="gray")
    plt.show()

def load_data(data_directory):
    print("Loading "+ data_directory +"directory...")
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".jpg")]
        i = 0
        for f in file_names:
            if i%100 == 0:
                print("Loaded "+str(i)+" Images")
            i+=1
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


def main():

    ROOT_PATH = "FaceData2"
    train_data_directory = os.path.join(ROOT_PATH, "Training")
    test_data_directory = os.path.join(ROOT_PATH, "Testing")
    train_images, train_labels = load_data(train_data_directory)
    test_images, test_labels = load_data(test_data_directory)
    origin_test_imgs, origin_test_labels = load_data("FaceData0/Testing")

    #  this is very important
    train_images = np.array(train_images) / 255.0
    test_images = np.array(test_images) / 255.0
    
    model = tf.keras.models.load_model("mymodel.h5py", custom_objects=None, compile=True)
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(50, 50)),
    #     keras.layers.Dense(512,activation=tf.nn.relu),
    #     keras.layers.Dense(2, activation=tf.nn.softmax)
    # ])

    model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    # history = model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)

    # displaytrainhistory(history)

    displaytestdata(test_images, test_labels, predictions, origin_test_imgs)
    tf.keras.models.save_model(model, 'mymodel.h5py', overwrite=True, include_optimizer=True)
    
if __name__ == "__main__":
    main()