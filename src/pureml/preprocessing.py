# Standardization (Z-score)
# Mean (μ)
# Min-Max Scaling 
# Range
# Normalization (L2)
 

import math

# 26. Standardization (Z-score)
# Used to scale data around mean with unit variance
# zi = (xi - μ) / σ

# zi → Standardized value (Z-score)
# xi → Original data value
# μ (mu) → Mean (average of all values)
# σ (sigma) → Standard deviation

def standardization(x):

    Total = 0
    avg = 0
    n = len(x)

    # Mean
    for i in range(n):
        Total = Total + x[i]

    avg = Total / n

    # variance
    variance = 0

    for i in range(n):
        variance = variance + ((x[i] - avg) ** 2)

    variance = variance / n

    # standard deviation
    std_dev = math.sqrt(variance)

    # z-score
    z_score = []

    for i in range(n):
        z_score.append((x[i] - avg) / std_dev)


    return z_score

# 27. Mean (μ)
# Average value of all elements
# μ = (1/n) * Σ xi

def mean(x):

    Total = 0
    n = len(x)
    mean_value = 0

    for i in range(n):
        Total = Total + x[i]

    
    mean_value = (1/n) * Total

    return mean_value

# 28. Min-Max Scaling
# Used to scale values between 0 and 1
# xi' = (xi - min) / (max - min)

def Min_Max_scaling(x):

    max_val = float('-inf')
    min_val = float('inf')

    magnitude = 0

    n = len(x)

    for i in range(n):

        if(x[i] > max_val):
            max_val = x[i]


        if(x[i] < min_val):
            min_val = x[i]

    scaled = []

    for i in range(n):
    
        value = (x[i] - min_val) / (max_val - min_val)
        scaled.append(value)


    return scaled
    
    
# 29. Range
# Difference between maximum and minimum value
# range = max - min
def range(x):

    max_val = float('-inf')
    min_val = float('inf')

    n = len(x)

    for i in range(n):

        if(x[i] > max_val):
            max_val = x[i]

        if(x[i] < min_val):
            min_val = x[i]

    return max_val - min_val

        
# 30. Normalization (L2)
# Used to convert vector into unit length
# x' = x / sqrt(Σ xi²)


def normalization(x):

    Total = 0
    n = len(x)
    magnitude = 0

    for i in range(n):
        Total = Total + (x[i] ** 2)


    magnitude = math.sqrt(Total)

    return magnitude


def normalized_vector_transform(x, magnitude):

    normalized = []
    n = len(x)

    for i in range(n):
        normalized.append(x[i] / magnitude)

    return normalized


    


