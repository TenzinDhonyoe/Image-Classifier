<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>ImageClassifier</h1>
    <p><code>ImageClassifier</code> is a Python project utilizing Convolutional Neural Networks (CNNs) to classify images into ten different categories. This project uses the CIFAR-10 dataset, which includes images of airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The model is built using TensorFlow and Keras with a simple yet effective architecture to ensure high accuracy and performance.</p>
    
<h2>Prerequisites</h2>
    <p>Before running this project, you'll need to have the following installed:</p>
    <ul>
        <li>Python 3.x</li>
        <li>TensorFlow 2.x</li>
        <li>OpenCV (cv2)</li>
        <li>NumPy</li>
        <li>Matplotlib</li>
    </ul>
    
<h2>Installation</h2>
    <p>Clone the repository to your local machine:</p>
    <pre><code>git clone https://github.com/yourusername/ImageClassifier.git</code></pre>
    <p>Navigate to the cloned directory:</p>
    <pre><code>cd ImageClassifier</code></pre>
    <p>Install the required packages:</p>
    <pre><code>pip install -r requirements.txt</code></pre>
    
<h2>Dataset</h2>
    <p>The project uses the CIFAR-10 dataset, automatically loaded via TensorFlow's dataset module. It includes 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.</p>
    
<h2>Model</h2>
    <p>The neural network model consists of three convolutional layers followed by max pooling layers, a flattening layer, and two dense layers. The model uses the ReLU activation function for its hidden layers and softmax for its output layer, optimizing the sparse categorical crossentropy loss function with the Adam optimizer.</p>
    
<h2>Training</h2>
    <p>The model is trained on a subset of 20,000 images from the training set for efficiency and tested on a subset of 4,000 images from the test set. It undergoes training over 10 epochs to adjust weights for accurate predictions.</p>
    
<h2>Prediction</h2>
    <p>The model can classify images into one of the ten CIFAR-10 categories. It preprocesses the images by converting them to RGB, normalizing their pixel values, and resizing them to fit the input shape of the model before making predictions.</p>
    
<h2>Usage</h2>
    <p>To classify your images, place them in the project directory and run the main script:</p>
    <pre><code>python image_classifier.py</code></pre>
    <p>Ensure your images are named appropriately (e.g., <code>horse.jpg</code>, <code>car.jpg</code>, etc.) as the script includes a predefined list of image filenames to process.</p>
    
<h2>Contributions</h2>
    <p>Contributions are welcome! If you'd like to improve the model, add features, or report a bug, please feel free to open an issue or a pull request.</p>
    
</body>
</html>
