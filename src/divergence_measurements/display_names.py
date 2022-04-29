__model_display_names = {
    "mbert-base": "MBERT (no fine-tuning)",
    "en-base-model-simultaneous": "EN base model",
    "fr-base-model-simultaneous": "FR base model",
    "en-fr-cl-baseline-simultaneous": "EN-FR CL baseline",
    "fr-en-cl-baseline-simultaneous": "FR-EN CL baseline",
    "en-fr-kd": "EN-FR Knowledge distillation",
    "en-fr-kd-mimic": "EN-FR Knowledge distillation (MIMIC)",
    "en-fr-ewc": "EN-FR EWC",
    "fr-en-ewc": "FR-EN EWC",
    # "fr-fr-mt-cl": "FR (golden)-FR MT CL baseline",
    "en-frmt-fr-cl": "EN-FR Machine translation intermediate",
    # "en-frmt-embedding-kd": "EN-FR MT intermediate (w/ regularization)",
    "en-frmt-fr-embedding-kd": "EN-FR MT-FR (w/ regularization)",
    # "fr-frmt-en-cl": "FR (golden)-FR MT-EN CL baseline",
    "en-fr-mt-cl": "EN-FR MT CL baseline",
    "non-cl-baseline": "Non-CL baseline"
}

__metric_display_names = {
    "svcca_correlations": "SVCCA divergence",
    "cca_correlations": "CCA divergence",
    "pearsonr_correlations": "Pearson R",
    "pearsonr_attention": "Pearson R",
    "jsdivergence_attention": "JS divergence",
}

__dataset_display_names = {
    "/home/zrlgca/data/fr-golden-2021-11-15/eval/": "FreGo dev set",
    "/home/zrlgca/data/en-silver-balanced-2021-11-24/split_1/eval/sampled/": "EnSi-5k dev set",
    "/home/zrlgca/data/en-silver-izumo-2021-11-15/sampled/": "EN silver test set"
}

__title_display_names = {
    "svcca_correlations": "SVCCA divergence of hidden representations",
    "cca_correlations": "CCA divergence of hidden representations",
    "pearsonr_correlations": "Pearson R of hidden representations",
    "pearsonr_attention": "Pearson R of attention outputs",
    "jsdivergence_attention": "JS divergence of attention distributions",
}


def model_display_name(name: str) -> str:
    return __model_display_names.get(name, name)


def metric_display_name(name: str) -> str:
    return __metric_display_names.get(name, name)


def title_display_name(name: str) -> str:
    return __title_display_names.get(name, name)


def dataset_display_name(name: str) -> str:
    return __dataset_display_names.get(name, name)
