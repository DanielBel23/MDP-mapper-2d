import numpy as np

def load_and_validate(npz_path):
    # Load the .npz file and validate it.
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Error: Unable to load file: {e}")
        return False, None
    matrices_arr = [data[key] for key in data.files if isinstance(data[key], np.ndarray)]
    if not validate_data_amount(matrices_arr):
        return False, None
    if not validate_data_content(matrices_arr):
        return False, None
    return True, matrices_arr


def validate_data_amount(matrices_arr):
    # Extract only arrays
    if len(matrices_arr) != 2:
        print(f"Error: Expected exactly 2 NumPy arrays, found {len(matrices_arr)}.")
        return False
    return True

def validate_data_content(matrices_arr):
    # verify the content of given arrays
    if matrices_arr[0].shape != matrices_arr[1].shape:
        print(f"Error: Matrices do not have the same shape: {matrices_arr[0].shape} and {matrices_arr[1].shape}")
        return False
    if matrices_arr[0].dtype.kind != 'i':
        print(f"Error: states matrix dtype {matrices_arr[0].dtype} is not integer.")
        return False
    if not np.issubdtype(matrices_arr[1].dtype, np.number):
        print(f"Error: rewards matrix is not numerical: {matrices_arr[1]}")
        return False
    if not allowed_state_values(matrices_arr[0]):
        print(f"Error: states matrix has invalid values (not -1, 0, 1): {matrices_arr[0]}")
        return False
    return True

def allowed_state_values(matrix):
    # supporting function, validating if there are no values beside: -1, 0, 1
    allowed = [-1, 0, 1]
    return np.all(np.isin(matrix, allowed))