def check_shape(file):
    shape = file.shape
    if shape != (1259, 300, 300) and shape == (300, 1259, 300):
        return file.T