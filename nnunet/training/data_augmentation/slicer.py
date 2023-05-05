import numbers


def slicer(arr, slices):
    """
    A dynamic N-dimensional array slicer that returns a tuple that can be used for slicing an N-dimensional array. Works exactly as Python and Numpy slicing, only with different syntax.
    The conventional slicing method has the drawback that one must know the dimensionality of the array beforehand. By contrast, this slicer can be adapted dynamically at runtime.
    :param arr: The array that should be sliced.
    :param slices: A list of slices that obeys the syntax described below.
    :return A tuple that can be used for slicing the array.

    Use case example - Center crop augmentation for an N-dimensional image:
    Imaging you have an N-dimensional array. You don't know what dimension it has. You want to crop each dimension to the center:
        image = ...
        print(image.shape)  # (32, 32) (48, 32, 32) or ...
        indices = comp_indices(image, width=8)
        # crop = image[???]  # Not possible to define a crop with arbitrary dimensions with regular slicing. Instead ...
        if len(image.shape) == 2:
            crop = image[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1]]  # image[10:22, 10:22])
        elif len(image.shape) == 3:
            crop = image[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1]]  # image[18:30, 10:22, 10:22]
        elif ...
    As you see, this cannot be done with regular slicing as the slicing syntax needs to be hardcoded.
    You would need to create an if case for each possible dimension.
    However, this slicer function enables you to do just that dynamically for data with arbitrary dimensions:
        image = ...
        print(image.shape)  # (32, 32) (48, 32, 32) or ...
        indices = comp_indices(image, width=8)
        crop = data[slicer(data, indices)]  # Center crop the data

    Syntax:
    - ... -> None
    - : -> [None]
    - i -> i
    - i:j -> [i, j]
    - i: -> [i, None]
    - :j -> [None, j]

    Syntax examples:
    - sub_arr = arr[5] -> sub_arr = arr[slicer(arr, [5])]
    - sub_arr = arr[...] -> sub_arr = arr[slicer(arr, [None])]
    - sub_arr = arr[..., 5] -> sub_arr = arr[slicer(arr, [None, 5])]
    - sub_arr = arr[5:9] -> sub_arr = arr[slicer(arr, [[5, 9]])]
    - sub_arr = arr[14:30, 12:17] -> sub_arr = arr[slicer(arr, [[14, 30], [12, 17]])]
    - sub_arr = arr[14:30, 12:17, ...] -> sub_arr = arr[slicer(arr, [[14, 30], [12, 17], None])]
    - sub_arr = arr[14:30, 12:17, :] -> sub_arr = arr[slicer(arr, [[14, 30], [12, 17], [None]])]
    - sub_arr = arr[14:30, :-17, :] -> sub_arr = arr[slicer(arr, [[14, 30], [None, -17], [None]])]
    - arr[5] = 7 -> arr[slicer(arr, [5])] = 7
    - arr[15:30, 12:19] = np.zeros((15, 7)) -> arr[slicer(arr, [[15, 30], [12, 19]])] = np.zeros((15, 7))

    This function was inspired by the following stackoverflow thread: https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis/37729566#37729566
    """
    slc = [slice(None)] * len(arr.shape)
    axis = 0
    for i in range(len(slices)):
        if slices[i] is None:  # Ellipsis: Take all values from multiple axes: array[..., ???]
            axis = len(arr.shape) - (len(slices) - 1)
        elif isinstance(slices[i], numbers.Number):  # Take single value from a single axis: array[???, i, ???]
            slc[axis] = slices[i]
            axis += 1
        elif len(slices[i]) == 1 and slices[i][0] is None:  # Colon: Take all values from a single axis: array[???, :, ???]
            axis += 1
        else:  # Take all values from the range i to j on a single axis: array[???, i:j, ???]. i or j can also be None
            slc[axis] = slice(slices[i][0], slices[i][1])
            axis += 1
    return tuple(slc)
