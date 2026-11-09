import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from xgboost import XGBClassifier
import xgboost as xgb
import shap
from Utils import (
    save_table,
    table1,
    canonical_labels,
    domain_scores,
    epsilon,
    write_json,
    bootstrap_jaccard_full,
    shap_difference_ci,
)
from plot_figure import (
    plot_pca,
    plot_loadings,
    plot_cluster_diagnostics,
    plot_dendrogram,
    plot_feature_distributions,
    plot_global_shap,
    plot_shap_intervals,
)


def baseline_characteristics(gait, participants, feature_names, group_names, save_path):
    table1(participants, save_path, group_names=group_names)
    included_rows = []
    for variable in ['age', 'height_cm', 'weight_kg', 'bmi', 'sppb_total_pre']:
        x = participants[variable].dropna()
        included_rows.append(
            {
                'variable': variable,
                'valid_n': len(x),
                'included_n90': f'{x.mean():.1f} ({x.std():.1f})',
            }
        )
    for variable in ['sex', 'walking_aid']:
        for category in sorted(participants[variable].unique()):
            n = participants[variable].eq(category).sum()
            included_rows.append(
                {
                    'variable': f'{variable}: {category}',
                    'valid_n': len(participants),
                    'included_n90': f'{n} ({n / len(participants) * 100:.1f}%)',
                }
            )
    for variable in feature_names:
        q1, median, q3 = gait.loc[gait.timepoint.eq('pre'), variable].quantile([0.25, 0.5, 0.75])
        included_rows.append(
            {
                'variable': variable,
                'valid_n': len(participants),
                'included_n90': f'{median:.2f} ({q1:.2f}–{q3:.2f})',
            }
        )
    table_s2 = pd.DataFrame(included_rows)
    save_table(table_s2, 'table_s2_included.csv', save_path=save_path)


def pca_results(space, dictionary, save_path):
    pre, zpre, pca = space['pre'], space['zpre'], space['pca']
    feature_names = space['features']
    pc = np.arange(1, pca.n_components_ + 1)
    variance = pd.DataFrame(
        {
            'PC': pc,
            'variance_fraction': pca.explained_variance_ratio_,
            'cumulative_fraction': np.cumsum(pca.explained_variance_ratio_),
        }
    )
    save_table(variance, 'figure_s1_pca_variance.csv', save_path=save_path)
    plot_pca(pc, variance, save_path)
    standardized = space['scaler'].transform(pre[feature_names])
    loading = np.corrcoef(standardized.T, zpre.T)[: len(feature_names), len(feature_names) :]
    loading = pd.DataFrame(loading, index=feature_names, columns=[f'PC{i}' for i in pc])
    ordered_features = []
    for domain in ['pace', 'regularity', 'symmetry', 'smoothness', 'complexity', 'stability']:
        ordered_features.extend(dictionary.loc[dictionary.domain.eq(domain), 'variable'])
    save_table(loading, 'figure_s2_loadings.csv', index=True, save_path=save_path)
    plot_loadings(loading, ordered_features, pc, save_path)


def cluster_diagnostics(space, config, n_boot, cluster_colors, save_path):
    pre, zpre = space['pre'], space['zpre']
    cluster_seed = config['random_state']
    diagnostic_rows = []
    for k in range(2, 6):
        labels = KMeans(
            n_clusters=k, n_init=config['kmeans_n_init'], random_state=cluster_seed
        ).fit_predict(zpre)
        diagnostic_rows.append(
            {
                'k': k,
                'silhouette': silhouette_score(zpre, labels),
                'davies_bouldin': davies_bouldin_score(zpre, labels),
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)
    save_table(diagnostics, 'cluster_diagnostics.csv', save_path=save_path)
    plot_cluster_diagnostics(diagnostics, save_path)
    plot_dendrogram(zpre, cluster_colors, save_path)
    cluster_draws = np.random.default_rng(cluster_seed).integers(0, len(pre), (n_boot, len(pre)))
    stability_rows = []
    for k in [2, 3, 4, 5]:
        original_model = KMeans(
            n_clusters=k, n_init=config['kmeans_n_init'], random_state=cluster_seed
        ).fit(zpre)
        original_labels = canonical_labels(original_model.labels_, pre.gait_speed.to_numpy())
        scores = []
        for b, draw in enumerate(cluster_draws):
            boot_model = KMeans(
                n_clusters=k, n_init=config['kmeans_n_init'], random_state=cluster_seed + b + 1
            ).fit(zpre[draw])
            scores.append(bootstrap_jaccard_full(original_labels, boot_model.predict(zpre), k))
        print(f'k={k}: completed {n_boot} bootstrap samples')
        scores = np.asarray(scores)
        for c in range(1, k + 1):
            values = scores[:, c - 1]
            stability_rows.append(
                {
                    'k': k,
                    'cluster': c,
                    'n': int((original_labels == c).sum()),
                    'mean_jaccard': np.nanmean(values),
                    'sd_jaccard': np.nanstd(values, ddof=1),
                    'valid_resamples': int(np.isfinite(values).sum()),
                }
            )
    table_s4 = pd.DataFrame(stability_rows)
    save_table(table_s4, 'table_s4_cluster_stability.csv', save_path=save_path)


def cluster_profiles(space, participants, config, clinical_variables, save_path):
    pre = space['pre']
    feature_names = space['features']
    baseline = pre.merge(participants, on='participant_id')
    baseline = pd.concat([baseline, domain_scores(baseline, config)], axis=1)
    table2_rows = []
    for variable in [*clinical_variables, *feature_names[:5], *config['domain_features']]:
        row = {'variable': variable}
        for c in [1, 2, 3]:
            x = baseline.loc[baseline.cluster.eq(c), variable].dropna()
            q1, median, q3 = x.quantile([0.25, 0.5, 0.75])
            digits = 0 if variable in clinical_variables else 2
            row[f'C{c}'] = f'{median:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]'
        table2_rows.append(row)
    table2 = pd.DataFrame(table2_rows)
    save_table(table2, 'table2_cluster_profiles.csv', save_path=save_path)
    group_counts = pd.crosstab(baseline.cluster, baseline.group)
    save_table(group_counts, 'table2_cluster_group_counts.csv', index=True, save_path=save_path)
    return baseline


def cluster_comparisons(baseline, feature_names, clinical_variables, participant_draws, save_path):
    table_s6_rows = []
    for variable in feature_names + clinical_variables:
        values = baseline[variable].to_numpy(float)
        labels = baseline.cluster.to_numpy()
        groups = [values[(labels == c) & np.isfinite(values)] for c in [1, 2, 3]]
        h, kw_p = (
            (0.0, 1.0) if np.unique(np.concatenate(groups)).size == 1 else stats.kruskal(*groups)
        )
        estimates = np.array([epsilon(values[draw], labels[draw]) for draw in participant_draws])
        finite = estimates[np.isfinite(estimates)]
        low, high = np.quantile(finite, [0.025, 0.975])
        row = {
            'variable': variable,
            'kruskal_wallis_p': kw_p,
            'epsilon_squared': h / (sum(map(len, groups)) - 1),
            'epsilon_ci_low': low,
            'epsilon_ci_high': high,
            'valid_bootstrap': len(finite),
        }
        for a, b in [(1, 2), (2, 3), (1, 3)]:
            row[f'p_C{a}_vs_C{b}'] = stats.mannwhitneyu(
                groups[a - 1], groups[b - 1], alternative='two-sided', method='auto'
            ).pvalue
        table_s6_rows.append(row)
    table_s6 = pd.DataFrame(table_s6_rows)
    save_table(table_s6, 'table_s6_pairwise_and_effect_sizes.csv', save_path=save_path)


def group_time_anova(gait, participants, feature_names, save_path):
    long = gait.merge(participants[['participant_id', 'group']], on='participant_id')
    anova_rows = []
    factor_names = {'group': 'group', 'timepoint': 'time', 'Interaction': 'interaction'}
    for feature in feature_names:
        result = pg.mixed_anova(
            data=long,
            dv=feature,
            within='timepoint',
            between='group',
            subject='participant_id',
            correction=False,
            effsize='np2',
        )
        for _, row in result.iterrows():
            anova_rows.append(
                {
                    'feature': feature,
                    'factor': factor_names[row.Source],
                    'df1': row.DF1,
                    'df2': row.DF2,
                    'F': row.F,
                    'p_value': row['p-unc'],
                    'partial_eta_squared': row.np2,
                }
            )
    anova = pd.DataFrame(anova_rows)
    interaction = anova.factor.eq('interaction')
    anova['q_interaction_bh_32'] = np.nan
    anova.loc[interaction, 'q_interaction_bh_32'] = multipletests(
        anova.loc[interaction, 'p_value'], method='fdr_bh'
    )[1]
    save_table(anova, 'table_s3_mixed_anova.csv', save_path=save_path)


def fit_surrogate(space, config, save_path):
    pre, post = space['pre'], space['post']
    feature_names = space['features']
    cluster_seed = config['random_state']
    xpre, xpost = (pre[feature_names], post[feature_names])
    ypre, ypost = (pre.cluster.to_numpy() - 1, post.cluster.to_numpy() - 1)
    parameters = config['xgboost_parameters']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cluster_seed)
    oof_probability = np.zeros((len(pre), 3))
    fold_scores = []
    for train, test in cv.split(xpre, ypre):
        fold_model = XGBClassifier(**parameters, objective='multi:softprob', num_class=3, n_jobs=1)
        fold_model.fit(xpre.iloc[train], ypre[train])
        probability = fold_model.predict_proba(xpre.iloc[test]).astype(float)
        probability /= probability.sum(axis=1, keepdims=True)
        oof_probability[test] = probability
        fold_scores.append(accuracy_score(ypre[test], probability.argmax(axis=1)))
    model = XGBClassifier(**parameters, objective='multi:softprob', num_class=3, n_jobs=1).fit(
        xpre, ypre
    )
    post_probability = model.predict_proba(xpost)
    post_prediction = post_probability.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        ypost, post_prediction, labels=[0, 1, 2], zero_division=0
    )
    confusion = confusion_matrix(ypost, post_prediction, labels=[0, 1, 2])
    confusion_table = pd.DataFrame(
        confusion,
        index=['Frozen C1', 'Frozen C2', 'Frozen C3'],
        columns=['Predicted C1', 'Predicted C2', 'Predicted C3'],
    )
    save_table(confusion_table, 'table_s9_confusion_matrix.csv', index=True, save_path=save_path)
    class_metrics = []
    for c in range(3):
        binary = (ypost == c).astype(int)
        auc = roc_auc_score(binary, post_probability[:, c])
        brier = np.mean((post_probability[:, c] - binary) ** 2)
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        tn = len(post) - tp - fp - fn
        class_metrics.append(
            {
                'cluster': c + 1,
                'n': support[c],
                'precision': precision[c],
                'recall': recall[c],
                'specificity': tn / (tn + fp),
                'f1': f1[c],
                'auc_ovr': auc,
                'brier_ovr': brier,
            }
        )
    table4 = pd.DataFrame(class_metrics)
    save_table(table4, 'table4_classification.csv', save_path=save_path)
    model_summary = {
        'baseline_cv_accuracy': accuracy_score(ypre, oof_probability.argmax(axis=1)),
        'fold_accuracies': fold_scores,
        'post_accuracy': accuracy_score(ypost, post_prediction),
        'macro_auc': table4.auc_ovr.mean(),
        'macro_brier': table4.brier_ovr.mean(),
    }
    write_json(save_path / 'table4_model_summary.json', model_summary)
    return dict(
        model=model,
        oof_probability=oof_probability,
        post_probability=post_probability,
        post_prediction=post_prediction,
    )


def shap_analysis(
    space,
    model_results,
    pairs,
    gait,
    participants,
    dictionary,
    group_names,
    resample_seed,
    save_path,
):
    feature_names = space['features']
    xpre, xpost = space['pre'][feature_names], space['post'][feature_names]
    model = model_results['model']
    explainer = shap.TreeExplainer(
        model, model_output='raw', feature_perturbation='tree_path_dependent'
    )
    shap_pre = np.asarray(explainer.shap_values(xpre, check_additivity=True))
    shap_post = np.asarray(explainer.shap_values(xpost, check_additivity=True))
    target_cluster = 3
    delta_shap = shap_pre[:, :, target_cluster - 1] - shap_post[:, :, target_cluster - 1]
    means_rows = []
    for group in group_names:
        for response in [1, 0]:
            mask = (pairs.group.eq(group) & pairs.favorable.eq(response)).to_numpy()
            means = pd.Series(delta_shap[mask].mean(axis=0), index=feature_names)
            for feature, value in means.items():
                means_rows.append(
                    {
                        'group': group,
                        'response': 'favorable' if response else 'unfavorable',
                        'n': int(mask.sum()),
                        'feature': feature,
                        'mean_delta_shap': value,
                    }
                )
    shap_changes = pd.DataFrame(means_rows)
    save_table(shap_changes, 'shap_group_means.csv', save_path=save_path)

    selected_features = [
        'stride_frequency_variability_v',
        'stride_frequency_variability_ml',
        'rms_acceleration',
        'stride_frequency_variability_ap',
        'stride_regularity_ml',
    ]
    shap_ci = shap_difference_ci(shap_pre, shap_post, pairs, feature_names, save_path)
    plot_shap_intervals(shap_ci, save_path)
    plot_feature_distributions(
        selected_features, gait, participants, dictionary, resample_seed, save_path
    )
    return shap_pre, shap_post


def global_importance(
    space, model_results, shap_post, participant_draws, n_permutations, resample_seed, save_path
):
    feature_names = space['features']
    post = space['post']
    xpost, ypost = post[feature_names], post.cluster.to_numpy() - 1
    model, post_prediction = model_results['model'], model_results['post_prediction']
    individual_importance = np.abs(shap_post).astype(float).mean(axis=2)
    global_importance = individual_importance.mean(axis=0)
    bootstrap_means = individual_importance[participant_draws].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)
    cutoff = np.sort(bootstrap_means, axis=1)[:, -5]
    above = bootstrap_means > cutoff[:, None]
    tied = bootstrap_means == cutoff[:, None]
    selection = above.astype(float) + tied * ((5 - above.sum(axis=1)) / tied.sum(axis=1))[:, None]
    global_shap = pd.DataFrame(
        {
            'feature': feature_names,
            'mean_absolute_shap': global_importance,
            'shap_ci_low': lower,
            'shap_ci_high': upper,
            'top5_frequency': selection.mean(axis=0),
        }
    )
    global_shap['rank'] = global_shap.mean_absolute_shap.rank(
        method='first', ascending=False
    ).astype(int)
    global_shap = global_shap.sort_values('rank')
    save_table(global_shap, 'figure_s6_global_shap.csv', save_path=save_path)
    plot_global_shap(global_shap, save_path)
    booster = model.get_booster()
    booster.set_param({'nthread': 1})
    rng = np.random.default_rng(resample_seed)
    permutation_draws = np.stack([rng.permutation(len(post)) for _ in range(n_permutations)])
    base_correct = int((post_prediction == ypost).sum())
    permutation_rows = []
    for feature in feature_names:
        perturbed = xpost.copy()
        original_column = xpost[feature].to_numpy()
        count_drop = []
        for draw in permutation_draws:
            perturbed[feature] = original_column[draw]
            probability = booster.predict(xgb.DMatrix(perturbed), training=False, strict_shape=True)
            shuffled_correct = int((probability.argmax(axis=1) == ypost).sum())
            count_drop.append(base_correct - shuffled_correct)
        count_drop = np.asarray(count_drop)
        fraction = (int((count_drop <= 0).sum()) + 1) / (n_permutations + 1)
        permutation_rows.append(
            {
                'feature': feature,
                'mean_accuracy_drop': count_drop.mean() / len(post),
                'operational_tail_fraction': fraction,
            }
        )
    permutation = pd.DataFrame(permutation_rows)
    permutation['operational_bh_q'] = multipletests(
        permutation.operational_tail_fraction, method='fdr_bh'
    )[1]
    save_table(permutation, 'figure_s6_permutation.csv', save_path=save_path)


def save_models(space, model_results, shap_pre, shap_post, save_path):
    pre, pca, km = space['pre'], space['pca'], space['km']
    model = model_results['model']
    post_probability = model_results['post_probability']
    oof_probability = model_results['oof_probability']
    post_prediction = model_results['post_prediction']
    model.get_booster().save_model(save_path / 'xgboost_model.json')
    np.savez_compressed(
        save_path / 'model_arrays.npz',
        participant_ids=pre.participant_id.astype(str).to_numpy(dtype=str),
        baseline_centers=km.cluster_centers_,
        scaler_mean=space['scaler'].mean_,
        scaler_scale=space['scaler'].scale_,
        pca_components=pca.components_,
        shap_pre=shap_pre,
        shap_post=shap_post,
        post_probability=post_probability,
        oof_probability=oof_probability,
        post_prediction=post_prediction,
    )
