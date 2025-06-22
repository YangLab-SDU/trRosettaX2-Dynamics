import json
import numpy as np


def save_to_json(obj, file):
    with open(file, "w") as f:
        jso = json.dumps(obj, cls=NpEncoder)
        f.write(jso)


def read_json(file):
    with open(file, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
