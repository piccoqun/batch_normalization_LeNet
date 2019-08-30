# Customized Batch Normalization on LeNet

## Model
**Data**: MNIST

**Neural Network**: LeNet

**Batch Normalization layer**: modified from the source code 
https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/layers/normalization.py#L138-L209

For more analysis, see the file 'Understanding Batch Normalization.pdf'

## Code
**Dependence**: Keras 2.2.5, tensorflow 1.14.0

**Run** in terminal

```python
python lenet.py
```

**Result**

![picture](https://github.com/piccoqun/batch_normalization_LeNet/blob/master/accuracy%20result.png)


