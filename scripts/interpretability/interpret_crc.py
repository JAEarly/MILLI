import torch

from interpretability.crc_interpretability import CrcInterpretabilityStudy


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_repeats = 10
    study = CrcInterpretabilityStudy(device, n_repeats=n_repeats)
    gather_data = True
    if gather_data:
        study.run_study()
    study.output_study(n_repeats)
