import math

# euclidean_distance
# manhattan_distance
# minkowski_distance
# cosine_similarity
# hamming_distance

# | p | Distance Type |
# | - | ------------- |
# | 1 | Manhattan     |
# | 2 | Euclidean     |
# | ∞ | Chebyshev     |



# Hamming distance: number of positions where values differ
# d = Σ (xi != yi)
def hamming_distance(x, y):
    if len(x) != len(y):
        raise ValueError("Length mismatch")

    count = 0
    n = len(x)

    i = 0
    while i < n:
        if x[i] != y[i]:
            count += 1
        i += 1

    return count

 

# Euclidean Distance: straight-line distance between two points in space
# d = sqrt(Σ (xi - yi)^2)
def euclidean_distance(x, y):

    Total = 0
    n = len(x)
    result = 0

    for i in range(n):
        Total = Total + (x[i] - y[i]) ** 2 #(p == 2)
     
    result = math.sqrt(Total)


    return result


# Manhattan Distance: sum of absolute differences between coordinates (grid-like path)
# d = Σ |xi - yi|
def manhattan_distance(x, y):

    Total = 0
    n = len(x)

    for i in range(n):
        Total = Total + abs(x[i] - y[i]) # (p == 1) power of parameter

    return Total



# Cosine Similarity: measures how similar two vectors are based on the angle between them
# cos(θ) = (Σ xi*yi) / (sqrt(Σ xi^2) * sqrt(Σ yi^2))

# Minimum angle  → Maximum similarity
# Maximum angle  → Minimum similarity
def cosine_similarity(x, y):

    dot_product = 0
    mag_x = 0
    mag_y = 0
    n = len(x)


    for i in range(n):
      dot_product = dot_product + x[i] * y[i]
      mag_x = mag_x + x[i] ** 2
      mag_y = mag_y + y[i] ** 2

    return dot_product / (math.sqrt(mag_x) * math.sqrt(mag_y)) 

# Minkowski Distance: generalized distance formula that includes Euclidean and Manhattan as special cases
# d = (Σ |xi - yi|^p)^(1/p)

def minkowski_distance(x, y, p):

    total = 0
    n = len(x)

    for i in range(n):
        total = total + abs(x[i] - y[i]) ** p

    
    return total ** (1/p)

