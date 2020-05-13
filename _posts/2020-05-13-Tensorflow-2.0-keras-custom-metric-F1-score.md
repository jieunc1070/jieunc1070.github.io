---
title: "[Tensorflow] Tensorflow 2.0 keras custom metric F1-score"
description: F1-score을 custom metric으로 정의합니다.
date: 2020-05-13 21:33:00
categories:
- Python
tags:
- Tensorflow
- Keras
---

Tensorflow 2.0 keras에서 F1 score를 custom metric으로 정의했는데, 모델 학습 과정을 보니 값이 좀 이상했습니다.


F1 score를 계산하는 함수 자체는 잘못된 점이 없는 것 같아서 K.print_tensor로 y_true, y_pred 값을 출력해 보았습니다.

{% highlight ruby linenos %}
import tensorflow.keras.backend as K
def f1_score(y_true, y_pred):
    K.print_tensor(y_true)
    K.print_tensor(y_pred)
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
{% endhighlight %}

{% highlight ruby%}
[[1]
 [1]
 [0]
 ...
 [0]
 [1]
 [0]]
 [[0.0013190686 0.99868089]
 [0.66807431 0.33192572]
 [1.14283484e-05 0.999988556]
 ...
 [9.42765865e-09 1]
 [0.0198234059 0.980176568]
 [1.95544952e-11 1]]
{% endhighlight %}

y_true와 y_pred가 shape이 다른데다가 y_true로는 라벨이, y_pred로는 softmax 결과 값이 들어오고 있었습니다.


그래서 아래와 같은 코드로 수정했더니 F1 score가 정상적으로 계산되었습니다.

{% highlight ruby linenos %}
import tensorflow.keras.backend as K
def f1_score(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.cast(tf.math.argmax(y_pred, 1), float)
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
{% endhighlight %}

{% highlight ruby%}
 [0 0 1 ... 0 0 1]
 [1 0 0 ... 1 1 0]
 [1 1 1 ... 0 0 0]
 [1 0 1 ... 1 1 1]
 [0 1 1 ... 1 1 0]
 [1 1 1 ... 1 0 1]
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 1 1]
 [0 1 0 ... 1 0 1]
 [1 1 1 ... 1 1 0]
 [0 0 0 ... 1 0 1]
{% endhighlight %}
