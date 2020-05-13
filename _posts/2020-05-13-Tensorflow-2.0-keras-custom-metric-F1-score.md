---
title: [Tensorflow] Tensorflow 2.0 keras custom metric F1-score
date: 2020-05-13 21:33:00
categories:
- Python
tags:
---

fdgd

{% highlight ruby linenos %}
import tensorflow.keras.backend as K
def f1_score(y_true, y_pred):
    K.print_tensor(y_true)
    K.print_tensor(y_pred)

    # taken from old keras source code
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

