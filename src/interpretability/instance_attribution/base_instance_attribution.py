from abc import ABC, abstractmethod

import torch.nn.functional as F

from interpretability.interpretability_util import InherentInterpretabilityError


class InstanceAttributionMethod(ABC):

    @abstractmethod
    def get_instance_clz_attributions(self, bag, model, original_pred, clz):
        pass


class InherentAttribution(InstanceAttributionMethod):

    def get_instance_clz_attributions(self, bag, model, original_pred, clz):
        _, attribution = model.forward_verbose([bag])
        if attribution is None or attribution[0] is None:
            raise InherentInterpretabilityError("Model does not provide an inherent interpretability measure.")
        attribution = attribution[0]

        # Check if instance attributions are n_instance x n_classes or 1 x n_instances
        if attribution.shape[0] != 1:
            attribution = F.softmax(attribution, dim=1)[:, clz]

        attribution = attribution.squeeze().detach().cpu().numpy()

        return attribution
