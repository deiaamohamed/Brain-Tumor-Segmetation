import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Test ResNet50
model = ResNet50(weights='imagenet')
print("ResNet50 loaded successfully.")

# Test Sequential
seq_model = Sequential()
print("Sequential model initialized.")

# Test ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255)
print("ImageDataGenerator initialized.")
