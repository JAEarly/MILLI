import torch

from interpretability.tef_interpretability import TefInterpretabilityStudy


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_repeats = 10
    dataset_name = "tiger"
    study = TefInterpretabilityStudy(device, dataset_name, n_repeats=n_repeats)
    gather_data = True
    if gather_data:
        study.run_study()
    study.output_study(n_repeats)
