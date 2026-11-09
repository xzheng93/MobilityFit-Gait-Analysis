from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from Utils import (
    fitted_space,
    transitions,
    gee_sensitivity,
    temporal_consistency,
    table_s11,
    nonlinear_sensitivity,
    trial_reliability,
    fragility_analysis,
)
from analysis import (
    baseline_characteristics,
    pca_results,
    cluster_diagnostics,
    cluster_profiles,
    cluster_comparisons,
    group_time_anova,
    fit_surrogate,
    shap_analysis,
    global_importance,
    save_models,
)


if __name__ == '__main__':
    # load data and set analysis parameters
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Run the gait-pattern analysis.')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=root / 'data',
        help='Directory containing the input CSV files and analysis settings.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=root / 'results',
        help='Directory for the generated tables and figures.',
    )
    args = parser.parse_args()
    data_path = args.data_dir.expanduser().resolve()
    save_path = args.output_dir.expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    config = json.loads((data_path / 'analysis_settings.json').read_text())
    gait = pd.read_csv(data_path / 'session_means.csv')
    participants = pd.read_csv(data_path / 'participants.csv')
    dictionary = pd.read_csv(data_path / 'feature_dictionary.csv')
    feature_names = config['feature_names']
    clinical_variables = ['sppb_total_pre', 'sppb_without_gait_pre', 'frail_nh_pre']
    group_names = ['intervention', 'control']
    cluster_colors = {1: '#26cfa5', 2: '#8583ff', 3: '#ff715f'}
    n_boot = 1000
    n_permutations = 200
    resample_seed = 20261012
    cpu_limit = threadpool_limits(limits=1)
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )

    print(f'{len(participants)} participants; {len(gait)} sessions')

    baseline_characteristics(gait, participants, feature_names, group_names, save_path)

    space = fitted_space(gait, feature_names, config)
    pca_results(space, dictionary, save_path)
    cluster_diagnostics(space, config, n_boot, cluster_colors, save_path)
    baseline = cluster_profiles(space, participants, config, clinical_variables, save_path)
    participant_draws = np.random.default_rng(resample_seed).integers(
        0, len(space['pre']), (n_boot, len(space['pre']))
    )
    cluster_comparisons(baseline, feature_names, clinical_variables, participant_draws, save_path)

    _, pairs = transitions(
        space['pre'], space['post'], participants, save_path, group_names, cluster_colors
    )
    group_time_anova(gait, participants, feature_names, save_path)
    gee_sensitivity(pairs, save_path, 'table_s8_gee')

    model_results = fit_surrogate(space, config, save_path)
    shap_pre, shap_post = shap_analysis(
        space,
        model_results,
        pairs,
        gait,
        participants,
        dictionary,
        group_names,
        resample_seed,
        save_path,
    )
    global_importance(
        space,
        model_results,
        shap_post,
        participant_draws,
        n_permutations,
        resample_seed,
        save_path,
    )

    temporal_consistency(space, config, save_path)
    table_s11(pairs, save_path, group_names)
    nonlinear_sensitivity(gait, participants, feature_names, config, save_path, group_names)
    trials = pd.read_csv(data_path / 'gait_trials.csv')
    trial_reliability(trials, space, participants, save_path)
    fragility_analysis(pairs, save_path)
    save_models(space, model_results, shap_pre, shap_post, save_path)
    print('Analysis complete.')
