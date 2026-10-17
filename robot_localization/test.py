
import numpy as np
arr1 = np.array([5, 1, 3.1, 4, 2])
arr2 = np.array([2, 3, 5.1029910, 1, 4])

arr1inds = arr1.argsort()
print(arr1inds.transpose())
sorted_arr1 = arr1[arr1inds[::-1]]
sorted_arr2 = arr2[arr1inds[::-1]]

print(sorted_arr1[:3])
print(sorted_arr2)

array1 = np.array([[1, 2], [3, 4]])

print(array1[-1, :])