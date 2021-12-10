import pickle
import json
import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def pickle2json(load_filename, save_filename):
    with open(load_filename, 'rb') as handle:
        data = pickle.load(handle)

    print(data)

    with open(save_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=NumpyArrayEncoder, ensure_ascii=False, indent=4)


def json2pickle(load_filename, save_filename):
    with open(load_filename) as f:
        data = json.load(f)

    print(data)

    with open(save_filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def modify_pickle(load_filename, save_filename):
    with open(load_filename, 'rb') as handle:
        data = pickle.load(handle)

    print(data)

    data["plans_per_stage"][0]["batch_size"] = 4
    data["plans_per_stage"][1]["batch_size"] = 4

    print(data)

    with open(save_filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # load_filename = "D:/Datasets/tmp/nnUNetPlansv2.1_plans_3D.pkl"
    # save_filename = "D:/Datasets/tmp/nnUNetPlansv2.1_plans_3D.json"
    # pickle2json(load_filename, save_filename)

    # load_filename = "D:/Datasets/tmp/nnUNetPlansv2.1_plans_3D.json"
    # save_filename = "D:/Datasets/tmp/nnUNetPlansv2.1_plans_3D_new.pkl"
    # json2pickle(load_filename, save_filename)

    load_filename = "D:/Datasets/tmp/nnUNetPlansv2.1_plans_3D.pkl"
    save_filename = "D:/Datasets/tmp/nnUNetPlansv2.1_plans_3D_new.pkl"
    modify_pickle(load_filename, save_filename)