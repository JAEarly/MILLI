import torch.nn.functional as F


class InherentInterpretabilityError(Exception):
    pass


def get_pred(bag, model):
    return F.softmax(model(bag), dim=0)


def get_clz_proba(bag, model, clz):
    pred = get_pred(bag, model)
    proba = pred[clz].detach().cpu().item()
    return proba


def get_clz_probas(bags, model, clz):
    probas = []
    for bag in bags:
        proba = get_clz_proba(bag, model, clz)
        probas.append(proba)
    return probas
