#Main file for image classifications
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

#the images in the dataset
class_names = ['Plane','Car','Bird','Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#shows if  the image is correctly classified or not
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i], cmap = plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])
    
# plt.show()

#Helps with making sure my computer doesnt blow up
#Caps the amount of data sets I will be using to build neural network
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]


# training neural network
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation ='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation = 'relu')) #convolutional looks at features and filters it out.
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation = 'relu'))
# model.add(layers.Dense(10, activation = 'softmax')) #how likely one clasification is the case

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images,testing_labels)) #epochs = how often will the model see the data over and over again

# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# model.save('image_classifier.keras')

model = models.load_model('image_classifier.keras')

image_files = ['horse.jpg', 'car.jpg', 'deer.jpg', 'plane.jpg']
images = []
predictions = []

for file in image_files:
    img = cv.imread(file)
    if img is not None:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert to RGB
        images.append(img)
        img_array = np.expand_dims(img, axis=0) / 255  # Normalize and add batch dimension
        prediction = model.predict(img_array)
        predictions.append(np.argmax(prediction))

num = len(images)

fig, axs = plt.subplots(1, num, figsize=(10, 5))

for i, img in enumerate(images):
    axs[i].imshow(img)
    axs[i].axis('off')
    print(f'Prediction is {class_names[predictions[i]]}')

plt.show()