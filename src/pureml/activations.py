# sigmoid
# relu
# tanh
# softmax
# log_loss

import math

# --------------------------------------------------
# ReLU activation
# Used in neural networks
# Returns 0 if input is negative, else returns same value
# Range: [0, ∞)
# --------------------------------------------------
def relu(x):
    if x > 0:
        return x
    else:
        return 0

# --------------------------------------------------
# Sigmoid activation
# Converts input into probability
# Range: (0, 1)
# Used in binary classification
# --------------------------------------------------
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

# --------------------------------------------------
# Tanh activation
# Similar to sigmoid but centered at 0
# Range: (-1, 1)
# Better for hidden layers
# --------------------------------------------------
def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


# --------------------------------------------------
# Softmax function
# Converts list of values into probabilities
# Output sum = 1
# Used in multi-class classification
# --------------------------------------------------
def softmax(x_list):

    max_val = max(x_list)

    exp_values = []

    for x in x_list:
        exp_values.append(math.exp(x - max_val))

    total = sum(exp_values)

    result = []

    for val in exp_values:
        result.append(val / total)

    return result


# --------------------------------------------------
# Log Loss (Binary Cross Entropy)
# Measures error between true and predicted values
# Lower value = better model
# Used in classification problems
# --------------------------------------------------
def log_loss(y_true, y_pred):
    n = len(y_true)
    loss = 0

    epsilon = 1e-15
    
    for i in range(n):

        pred = max(min(y_pred[i], 1 - epsilon), epsilon)

        loss = loss + (
            y_true[i] * math.log(pred) + 
            (1 - y_true[i]) * math.log(1 - pred))
    
    return -loss / n