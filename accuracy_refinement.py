from utils.superclasses import super_map


def predict_superclass(pred_distribution):
    top_args = pred_distribution.argsort()[-1:][::-1]
    super_dict = {}

    for class_index in top_args:
        superclass = super_map[class_index]
        if superclass in super_dict:
            super_dict[superclass] += 1
        else:
            super_dict[superclass] = 1

    top_super = {k: v for k, v in sorted(super_dict.items(), key=lambda item: item[1], reverse=True)}

    pred_superclass = list(top_super.keys())[0]

    return pred_superclass


def accuracy_refinement(pred_distribution, low_bandwith=True):
    if low_bandwith:
        return predict_superclass(pred_distribution)