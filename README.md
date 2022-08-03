# daisi_exp1_monkeypox

## Pox Detection with Daisies

### How to Call


First, we simply load the PyDaisi package:

```
import pydaisi as pyd
```

Next, we connect to the Daisi:

```
pox_detection = pyd.Daisi("ashhadulislam/PoxDetection")
```

Consequently we build the model
```
model=pox_detection.load_model()
```

Then we pass a URL of an image to predict the type of pox
```
image_url="https://drive.google.com/uc?export=view&id=14sF_FaFvfYzrQCCQRX6IK87aBPFerfWb"
predictions=pox_detection.predict(model,image_url)

# if you are using daisi, you have to put .value
print(predictions.value)
```


The output is going to be a list of pox types. The first item in the list has the highest probability of occurring.
```
['Monkeypox', 'Chickenpox', 'Measles', 'Normal']
```
Monkeypox is the most likely disease.


