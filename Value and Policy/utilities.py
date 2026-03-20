#harsh khandelwal
import numpy as np
import math

def poisson_function(lambda_val:int, r:int):
    """
    Returns a NUMPY ARRAY of length r+1 containing
    values in a poisson distribution where the expected
    value is lambda_val. An element at index i in the
    vector that is returned should equal P(i|lambda_val),
    assuming P is a Poisson distribution.  See the spec
    for the definition of a Poisson distribution.

    Ensure that all values in your distribution sum to 1.
    To do this, set the final value in your distribution
    to 1 minus the sum of all the other values.
     >>> isinstance(poisson_function(1,10),np.ndarray)
     True
     >>> abs(poisson_function(5, 20).tolist()[0] - 0.006737946999085467) < 0.001
     True
     >>> abs(poisson_function(5, 20).tolist()[5] - 0.1754673697678507) < 0.001
     True
     >>> abs(poisson_function(8, 10).tolist()[0] - 0.00033546262790251185) < 0.001
     True
     >>> abs(poisson_function(8, 10).tolist()[5] - 0.09160366159257924) < 0.001
     True
     >>> abs(poisson_function(3, 44).tolist()[0] - 0.049787068367863944) < 0.001
     True
     >>> len(poisson_function(3, 44).tolist())
     45
     >>> abs(poisson_function(3, 44).tolist()[5] - 0.10081881344492448) < 0.001
     True
     """
    result = np.zeros(r + 1)
    for i in range(r):
        result[i] = (lambda_val ** i / math.factorial(i)) * math.exp(-lambda_val)
    # Set last value so all sum to 1
    result[r] = 1.0 - np.sum(result[:r])
    return result


if __name__ == '__main__':
    import doctest
    doctest.testmod()
