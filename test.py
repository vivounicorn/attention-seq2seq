import pickle


def params_dump(params, params_path):
    if params is not None:
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)


if __name__ == '__main__':
    pass