# Circle Detection ML Challenge

This GitHub repository contains code used to create, train, and test a CNN model to correctly identify the position and radius of circles in 100x100 pixel images with varying levels of noise. The library used is TensorFlow.

## Model

The model is as mentioned created and trained using TensorFlow libraries and contains close to 12.8M parameters.

```python
def create_circle_detector_model(input_shape=(100, 100, 1)):
    model = models.Sequential(
        [
            layers.Conv2D(128, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(512, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(512, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(3),  # Output layer for (x, y, radius)
        ]
    )
    return model
```

## Training

The training was done using TensorFlow's model.fit()-method. Data was created for every step of every epoch using the generator function below, which generates one batch (In this case 32) of datapoints using the given function ```generate_examples()```. During the training process noise levels between 0.5 and 0.9 were used and the model was trained on 10 or 20 epochs, consisting of 200 steps each, for every noise level amounting to a total of 5 120 000 separate data points used for training. 
```python
def generate_train_data(batch_size: int = 32, noise_level: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    while True:
        images = []
        labels = []
        gen = generate_examples(noise_level = noise_level)
        for _ in range(batch_size):
            img, params = next(gen)
            images.append(np.expand_dims(img, axis=-1))
            labels.append(params)
        yield np.array(images), np.array(labels)
```

## Evaluation

The model was evaluated on four different noise levels between 0.4 and 0.7. For the testing of each noise level, 2048 data points were used. In addition to the average IOU for every noise level, the percentage of predicted samples with an IOU above 0.5, 0.75, and 0.9 were also recorded with complementary plots of the best and worst predictions at every noise level

### Measures for noise_level = 0.4
```
Average IOU: 0.855139341651044
Percentage of IoU over 0.5: 98.93%
Percentage of IoU over 0.75: 87.74%
Percentage of IoU over 0.9: 40.62%
```

### Measures for noise_level = 0.5
```
Average IOU: 0.8428614379726439
Percentage of IoU over 0.5: 98.24%
Percentage of IoU over 0.75: 84.28%
Percentage of IoU over 0.9: 35.94%
```

### Measures for noise_level = 0.6
```
Average IOU: 0.6500921158099726
Percentage of IoU over 0.5: 78.27%
Percentage of IoU over 0.75: 41.16%
Percentage of IoU over 0.9: 7.57%
```

### Measures for noise_level = 0.7
```
Average IOU: 0.3600532404117393
Percentage of IoU over 0.5: 33.30%
Percentage of IoU over 0.75: 6.40%
Percentage of IoU over 0.9: 0.39%
```
