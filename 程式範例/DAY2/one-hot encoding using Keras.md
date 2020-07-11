#
```
one-hot encoding using Keras
https://www.educative.io/edpresso/how-to-perform-one-hot-encoding-using-keras
```
```
The Keras API provides a to_categorical() method that can be used to one-hot encode integer data. 
If the integer data represents all the possible values of the classes, then the to_categorical() method can be used directly; 
otherwise, the number of classes can be passed to the method as the num_classes parameter
```

```
import numpy as np
from keras.utils import to_categorical

### Categorical data to be converted to numeric data
colors = ["red", "green", "yellow", "red", "blue"]

### Universal list of colors
total_colors = ["red", "green", "blue", "black", "yellow"]

### map each color to an integer
mapping = {}
for x in range(len(total_colors)):
  mapping[total_colors[x]] = x

# integer representation
for x in range(len(colors)):
  colors[x] = mapping[colors[x]]

one_hot_encode = to_categorical(colors)
```

```
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1.],
       [1., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0.]], dtype=float32)
```
