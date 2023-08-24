from keras.datasets import cifar10, cifar100
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

#You can remove the save module feature if you don't want to save it else just save it in whatever directory.

(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()

# Normalize CIFAR-10 data
x_train_10 = x_train_10.astype('float32') / 255.0
x_test_10 = x_test_10.astype('float32') / 255.0

# Load CIFAR-100
(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data(label_mode='fine')

# Normalize CIFAR-100 data
x_train_100 = x_train_100.astype('float32') / 255.0
x_test_100 = x_test_100.astype('float32') / 255.0

# Take a subset (first 10 classes) from CIFAR-100 
x_train_100_subset = x_train_100[np.isin(y_train_100, np.arange(10)).flatten()]
y_train_100_subset = y_train_100[np.isin(y_train_100, np.arange(10)).flatten()] + 10  # adjusting class indices

x_test_100_subset = x_test_100[np.isin(y_test_100, np.arange(10)).flatten()]
y_test_100_subset = y_test_100[np.isin(y_test_100, np.arange(10)).flatten()] + 10  # adjusting class indices

# Concatenate the datasets
x_train = np.concatenate((x_train_10, x_train_100_subset), axis=0)
y_train = np.concatenate((y_train_10, y_train_100_subset), axis=0)

x_test = np.concatenate((x_test_10, x_test_100_subset), axis=0)
y_test = np.concatenate((y_test_10, y_test_100_subset), axis=0)

def test_func():
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
                "beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout"]


    # Plotting the first 25 test images from the dataset
    plt.figure(figsize=(10,10))
    for i in range(20):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_test[i][0]])
    plt.show()


    index = []
    img = []
    for i in range(20):
        idx = np.random.randint(0, x_test.shape[0])
        index.append(idx)
        img.append(x_test[idx])

    # Display the image
    for imgs in img:
        plt.imshow(imgs, cmap=plt.cm.binary, interpolation='nearest')
        plt.show()


    loaded_model = load_model('C:\\Users\\James\\image_recognition_model.h5')

    # Make predictions using the trained model
    predictions = loaded_model.predict(np.array(img))
    predicted_classes = np.argmax(predictions, axis=1)

    for i, predicted_class in enumerate(predicted_classes):
        print(f"Predicted Class: {class_names[predicted_class]}")
        print(f"Actual Class: {class_names[y_test[index[i]][0]]}")
    return

test_func()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(20, activation='softmax'))  # Updated the number of output units

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

predictions = model.predict(x_test)
predicted_class = np.argmax(predictions[0])
model.save('YourDIR')

test_results = {(1,"cfar10"):"0.6758000254631042", 
                (2,"cfar10,cfar100"):"0.659818172454834",
                (3,"cfar10,cfar100"):"0.6576363444328308",
                (4,"cfar10,cfar100"):"0.6014545559883118"}

#By Jermaine Miller

  
