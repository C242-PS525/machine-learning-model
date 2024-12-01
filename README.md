# Machine Learning Model: Multi-Class Classification
# Identification of Vegetables and Fruits

This repository contains a machine learning model for classifying data into 41 distinct categories. Below, you will find details about the project, including the dataset, training process, and how to use the model for predictions.

---

## Project Overview
The purpose of this project is to develop a multi-class classification model using a pre-trained neural network as the base. The model is fine-tuned to accurately classify input data into one of 41 categories. The implementation leverages TensorFlow and Keras libraries.

### Key Features
- **Model Architecture**: Utilizes a pre-trained model with added dense layers for multi-class classification.
- **Optimization**: Includes dropout layers to prevent overfitting and early stopping to optimize training.
- **Visualization**: Provides training and validation accuracy/loss graphs to analyze model performance.

---

## Project Structure

```
├── dataset/                    # Folder containing training and validation datasets
├── model/                      # Folder for saving trained model files
├── fruitandvegetables.ipynb    # Jupyter Notebook for model development
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```

---

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ilmalyakinn/fruit-and-vegetables-model.git
   cd fruit-and-vegetables-model
   ```

2. **Install Dependencies**:
   Install the required Python packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   Place your training and validation data in the `dataset/` folder. Ensure the data is properly labeled.

---

## Model Training

To train the model, execute the following steps:

1. **Load and Prepare Data**:
   Ensure your training and validation data are preprocessed and loaded correctly.

2. **Run the Training Script**:
   Open `fruitandvegetables.ipynb` or run the Python script to train the model.

3. **Monitor Performance**:
   Training and validation accuracy/loss graphs will be displayed to track the model's performance over epochs.

---

## Testing the Model

Test the model using validation data:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load model
model = load_model('/content/model.h5')

# Predefined class labels
labels = ['apple', 'avocado', 'banana', 'beetroot', 'cabbage', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'durian', 'eggplant', 'garlic', 'ginger', 'grapes', 'guava', 'kiwi', 'langsat', 'lemon', 'lettuce', 'mango', 'mangosteen', 'melon', 'onion', 'orange', 'papaya', 'paprika', 'pear', 'peas', 'pineapple', 'potato', 'raddish', 'salak', 'soy beans', 'spinach', 'strawberies', 'sweetpotato', 'tomato', 'turnip', 'water-guava', 'watermelon']

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Convert RGBA to RGB
    img_width, img_height = img.size
    
    # If image is larger than 224x224, resize to 224x224
    if img_width > 224 or img_height > 224:
        img = img.resize((224, 224))
    
    # Alternatively, for larger images, you can crop them to 224x224
    # img = img.crop((0, 0, 224, 224))  # Crop top-left 224x224 region
    
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Apply MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = labels[predicted_class_index]
    return predicted_class, predictions

# Create an upload button
upload_button = widgets.FileUpload(accept='image/*', multiple=False)

# Create an output widget to display the results
output = widgets.Output()

def on_upload_change(change):
    with output:
        clear_output()
        if upload_button.value:
            uploaded_file = list(upload_button.value.values())[0]
            image_path = '/content/uploaded_image.jpg'  # Temporary file path
            with open(image_path, 'wb') as f:
                f.write(uploaded_file['content'])

            predicted_class, predictions = predict_image(image_path)

            print(f"Predicted Class: {predicted_class}")
            print("Probabilities:")
            for i, prob in enumerate(predictions[0]):
                print(f"- {labels[i]}: {prob:.4f}")

# Register the callback function
upload_button.observe(on_upload_change, names='value')

# Display the upload button and the output widget
display(upload_button)
display(output)


```

---

## Visualization

To visualize training and validation accuracy/loss, use the following:

```python
import matplotlib.pyplot as plt

# Example plots
plt.figure(figsize=(14, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'r', label='Training Loss')
plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## Visualization of Model Training Results

<img width="649" alt="Screenshot 2024-11-30 205119" src="https://github.com/user-attachments/assets/d7dab1d8-349b-4e96-8b90-b5890dd6c763">
)


## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. Feel free to open issues for suggestions or bug reports.

---

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute this project as per the terms of the license.
## Contributors
we would like to thank the following people for their contributions to this project:

- [Aan Andiyana Sandi](https://github.com/aan-andiyanaS)
- [Ilmal Yakin Nurahman](https://github.com/ilmalyakinn)
- [Ridwan Fadillah](https://github.com/RidwanFadillah)

---

