from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('ai_classification_with_inceptionv3.h5')
print('Input file location')
path=str(input())

img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)/255

images = np.vstack([x])

pred = model.predict(images)


classes=['ai','blar','eevee','loki','luna']
classes.sort()

index=np.argmax(pred[0])
print(classes[index])