from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.proportion import proportion_confint
from patsy import dmatrix
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.cov_struct import Independence
from plot_figure import plot_transitions


def save_table(frame, filename, save_path, index=False):
    frame.to_csv(save_path / filename, index=index)


def table1(participants, out, group_names):
    rows = []
    for variable in ['age', 'height_cm', 'weight_kg', 'bmi', 'sppb_total_pre']:
        vals = [participants.loc[participants.group.eq(g), variable].dropna() for g in group_names]
        normality = [float(stats.shapiro(x).pvalue) for x in vals]
        normal = all((v >= 0.05 for v in normality))
        test = 'Independent t-test' if normal else 'Mann-Whitney U, two-sided'
        p = (
            stats.ttest_ind(*vals, equal_var=True).pvalue
            if normal
            else stats.mannwhitneyu(*vals, alternative='two-sided').pvalue
        )
        rows.append(
            dict(
                variable=variable,
                **{
                    g: f"{x.mean():.{(2 if variable == 'sppb_total_pre' else 1)}f} ({x.std():.1f}); n={len(x)}"
                    for g, x in zip(group_names, vals)
                },
                p_value=p,
                display='mean (SD)',
                test=test,
                shapiro_intervention=normality[0],
                shapiro_control=normality[1],
            )
        )
    for variable in ['sex', 'walking_aid']:
        contingency = pd.crosstab(participants.group, participants[variable])
        chi = stats.chi2_contingency(contingency, correction=False)
        use_fisher = contingency.shape == (2, 2) and (chi.expected_freq < 5).any()
        continuity = contingency.shape == (2, 2) and (not use_fisher)
        p = (
            stats.fisher_exact(contingency.to_numpy()).pvalue
            if use_fisher
            else stats.chi2_contingency(contingency, correction=continuity).pvalue
        )
        for category in sorted(participants[variable].dropna().unique()):
            record = {
                'variable': variable + ':' + str(category),
                'p_value': p,
                'display': 'n (%)',
                'test': 'Fisher exact'
                if use_fisher
                else 'Chi-square with Yates continuity correction'
                if continuity
                else 'Pearson chi-square without continuity correction',
            }
            for g in group_names:
                x = participants.loc[participants.group.eq(g), variable]
                n = int(x.eq(category).sum())
                record[g] = f'{n} ({n / len(x) * 100:.1f}%)'
            rows.append(record)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'table1.csv', index=False)
    return result


def fitted_space(gait, features, config):
    """Fit scaling, PCA and clusters at baseline; assign post-test observations."""
    pre = gait[gait.timepoint.eq('pre')].sort_values('participant_id').reset_index(drop=True).copy()
    post = (
        gait[gait.timepoint.eq('post')]
        .set_index('participant_id')
        .loc[pre.participant_id]
        .reset_index()
        .copy()
    )
    scaler = StandardScaler().fit(pre[features])
    pca = PCA(n_components=config['variance_threshold'], svd_solver='full')
    zpre = pca.fit_transform(scaler.transform(pre[features]))
    zpre = pca.transform(scaler.transform(pre[features]))
    zpost = pca.transform(scaler.transform(post[features]))
    km = KMeans(
        n_clusters=3, n_init=config['kmeans_n_init'], random_state=config['random_state']
    ).fit(zpre)
    speed_order = (
        pre.assign(raw=km.labels_)
        .groupby('raw')
        .gait_speed.mean()
        .sort_values(ascending=False)
        .index
    )
    mapping = {int(raw): int(label) for label, raw in enumerate(speed_order, 1)}
    pre['cluster'] = [mapping[int(v)] for v in km.labels_]
    post['cluster'] = [mapping[int(v)] for v in km.predict(zpost)]
    return dict(
        pre=pre,
        post=post,
        scaler=scaler,
        pca=pca,
        km=km,
        zpre=zpre,
        zpost=zpost,
        mapping=mapping,
        features=list(features),
    )


def canonical_labels(raw, speed):
    order = sorted(np.unique(raw), key=lambda c: -float(speed[raw == c].mean()))
    mapping = {int(c): i + 1 for i, c in enumerate(order)}
    return np.array([mapping[int(c)] for c in raw])


def domain_scores(frame, config):
    scores = pd.DataFrame(index=frame.index)
    for domain, features in config['domain_features'].items():
        terms = []
        for f in features:
            s = config['scaling'][f]
            value = (frame[f] - s['lower']) / (s['upper'] - s['lower'])
            if not s['higher_is_better']:
                value = 1 - value
            if config['clip_domain_scores']:
                value = value.clip(0, 1)
            terms.append(value)
        scores[domain] = pd.concat(terms, axis=1).mean(axis=1)
    return scores


def epsilon(values, labels):
    """Calculate the Kruskal-Wallis epsilon-squared effect size."""
    mask = np.isfinite(values)
    vals, groups = (values[mask], labels[mask])
    samples = [vals[groups == c] for c in (1, 2, 3)]
    if len(vals) < 4 or any((len(v) == 0 for v in samples)):
        return np.nan
    if np.unique(vals).size == 1:
        return 0.0
    return float(stats.kruskal(*samples).statistic / (len(vals) - 1))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def transitions(pre, post, participants, out, group_names, cluster_colors):
    pairs = (
        pre[['participant_id', 'cluster']]
        .rename(columns={'cluster': 'cluster_pre'})
        .merge(
            post[['participant_id', 'cluster']].rename(columns={'cluster': 'cluster_post'}),
            on='participant_id',
        )
        .merge(participants, on='participant_id')
    )
    pairs['favorable'] = (
        pairs.cluster_post.lt(pairs.cluster_pre)
        | pairs.cluster_post.eq(pairs.cluster_pre) & pairs.cluster_pre.isin([1, 2])
    ).astype(int)
    pairs.to_csv(out / 'cluster_assignments.csv', index=False)
    matrices = {}
    for group in group_names:
        d = pairs[pairs.group.eq(group)]
        m = pd.crosstab(d.cluster_pre, d.cluster_post).reindex(
            index=[1, 2, 3], columns=[1, 2, 3], fill_value=0
        )
        m.to_csv(out / f'figure2_transitions_{group}.csv')
        matrices[group] = m.to_numpy()
    plot_transitions(matrices, group_names, cluster_colors, out)
    favorable = []
    for group in group_names:
        d = pairs[pairs.group.eq(group)]
        favorable.append(
            {
                'group': group,
                'n': len(d),
                'favorable': int(d.favorable.sum()),
                'unfavorable': int((1 - d.favorable).sum()),
                'proportion_favorable': d.favorable.mean(),
            }
        )
    pd.DataFrame(favorable).to_csv(out / 'paper_favorable_response_counts.csv', index=False)
    from statsmodels.stats.proportion import proportion_confint

    a, b = favorable
    contingency = np.array([[a['favorable'], a['unfavorable']], [b['favorable'], b['unfavorable']]])
    chi2, pvalue, dof, expected = stats.chi2_contingency(contingency, correction=False)
    rr = a['favorable'] / a['n'] / (b['favorable'] / b['n'])
    se = np.sqrt(1 / a['favorable'] - 1 / a['n'] + 1 / b['favorable'] - 1 / b['n'])
    table3 = {
        'test': 'Pearson chi-square without continuity correction',
        'chi_square': float(chi2),
        'degrees_of_freedom': int(dof),
        'p_value': float(pvalue),
        'unadjusted_rr': float(rr),
        'rr_ci_low': float(np.exp(np.log(rr) - 1.96 * se)),
        'rr_ci_high': float(np.exp(np.log(rr) + 1.96 * se)),
        'rr_ci_method': 'log-risk-ratio normal approximation',
        'group_counts': favorable,
    }
    write_json(out / 'table3_unadjusted_comparison.json', table3)
    proportions = []
    for group, m in matrices.items():
        for i in range(3):
            for j in range(3):
                n = int(m[i].sum())
                count = int(m[i, j])
                lo, hi = proportion_confint(count, n, method='wilson')
                proportions.append(
                    dict(
                        group=group,
                        cluster_pre=i + 1,
                        cluster_post=j + 1,
                        count=count,
                        baseline_cluster_n=n,
                        proportion=count / n,
                        ci_low=float(lo),
                        ci_high=float(hi),
                        ci_method='Wilson score; conditional on baseline cluster',
                    )
                )
    pd.DataFrame(proportions).to_csv(out / 'figure2_transition_proportions_wilson.csv', index=False)
    return (matrices, pairs)


def gee_sensitivity(pairs, out, prefix):
    """Modified Poisson GEE with MD covariance and t(K-p) inference."""
    formula = "C(group, Treatment('control')) + C(cluster_pre, Treatment(3)) + age + C(sex, Treatment('male')) + C(walking_aid, Treatment('none'))"
    arm_term = "C(group, Treatment('control'))[T.intervention]"
    design = dmatrix(formula, pairs, return_type='dataframe', NA_action='raise')
    names = list(design.columns)
    x, y = design.to_numpy(), pairs.favorable.to_numpy()
    groups = pairs.facility_id.to_numpy()
    k, p = pairs.facility_id.nunique(), x.shape[1]
    j = names.index(arm_term)
    fit = GEE(y, x, groups=groups, family=Poisson(), cov_struct=Independence()).fit(
        maxiter=120, ctol=1e-9, cov_type='bias_reduced'
    )
    summary = dict(
        formula='favorable ~ ' + formula,
        family='poisson',
        link='log',
        working_correlation='independence',
        working_correlation_estimate=0.0,
        covariance='Mancl-DeRouen bias-reduced sandwich',
        reference='t',
        df=int(k - p),
        n=len(pairs),
        facilities=int(k),
        parameters=p,
        converged=bool(fit.converged),
    )
    covariance = np.asarray(fit.cov_params())
    mu = np.exp(x @ fit.params)
    se = np.sqrt(np.diag(covariance))
    critical = stats.t.ppf(0.975, k - p)
    low, high = np.exp(fit.params - critical * se), np.exp(fit.params + critical * se)
    pv = 2 * stats.t.sf(np.abs(fit.params / se), k - p)
    coeff = pd.DataFrame(
        dict(
            term=names,
            coefficient=fit.params,
            robust_se=se,
            exponentiated_coefficient=np.exp(fit.params),
            ratio_ci_low=low,
            ratio_ci_high=high,
            p_value=pv,
            ratio_type='risk_ratio',
        )
    )
    coeff.to_csv(out / f'{prefix}_coefficients.csv', index=False)
    x1, x0 = x.copy(), x.copy()
    x1[:, j], x0[:, j] = 1, 0
    mu1, mu0 = np.exp(x1 @ fit.params), np.exp(x0 @ fit.params)
    rd = float(np.mean(mu1 - mu0))
    gradient = np.mean(mu1[:, None] * x1 - mu0[:, None] * x0, axis=0)
    rd_se = float(np.sqrt(gradient @ covariance @ gradient))
    pd.DataFrame(
        dict(
            participant_id=pairs.participant_id,
            favorable=y,
            observed_prediction=mu,
            intervention_standardized_prediction=mu1,
            control_standardized_prediction=mu0,
        )
    ).to_csv(out / f'{prefix}_standardized_predictions.csv', index=False)
    summary.update(
        status='ok' if fit.converged else 'not_converged',
        arm_coefficient=coeff.iloc[j].to_dict(),
        standardized_risks=dict(intervention=float(mu1.mean()), control=float(mu0.mean())),
        standardized_risk_difference=dict(
            estimate=rd,
            robust_se=rd_se,
            ci_low=rd - critical * rd_se,
            ci_high=rd + critical * rd_se,
            ci_method='Marginal standardization; delta method with MD covariance; t(K-p) CI',
        ),
        predictions_above_one=dict(
            observed=int((mu > 1).sum()),
            intervention=int((mu1 > 1).sum()),
            control=int((mu0 > 1).sum()),
        ),
        interpretation='Conditions on derived labels. Poisson predictions are not constrained to [0,1].',
    )
    write_json(out / f'{prefix}_summary.json', summary)
    return summary


def temporal_consistency(space, config, out):
    post_km = KMeans(
        n_clusters=3, n_init=config['kmeans_n_init'], random_state=config['random_state']
    ).fit(space['zpost'])
    cost = cdist(space['km'].cluster_centers_, post_km.cluster_centers_)
    baseline_raw, post_raw = linear_sum_assignment(cost)
    match = dict(zip(baseline_raw, post_raw))
    post_to_canonical = {int(j): space['mapping'][int(i)] for i, j in match.items()}
    rows = []
    for i, j in match.items():
        row = dict(
            cluster=space['mapping'][int(i)],
            baseline_kmeans_label=int(i),
            matched_post_kmeans_label=int(j),
            centroid_drift=float(cost[i, j]),
            baseline_n=int((space['km'].labels_ == i).sum()),
            refit_post_n=int((post_km.labels_ == j).sum()),
        )
        for pc, (a, b) in enumerate(
            zip(space['km'].cluster_centers_[i], post_km.cluster_centers_[j]), 1
        ):
            row[f'pre_PC{pc}'] = a
            row[f'post_PC{pc}'] = b
        rows.append(row)
    centers = pd.DataFrame(rows).sort_values('cluster')
    centers.to_csv(out / 'note3_centroid_drift.csv', index=False)
    assigned = space['post'][['participant_id', 'cluster']].rename(
        columns={'cluster': 'fixed_baseline_cluster'}
    )
    assigned['independent_post_cluster_matched'] = [
        post_to_canonical[int(v)] for v in post_km.labels_
    ]
    assigned.to_csv(out / 'note3_post_assignment_comparison.csv', index=False)
    pd.crosstab(assigned.fixed_baseline_cluster, assigned.independent_post_cluster_matched).reindex(
        index=[1, 2, 3], columns=[1, 2, 3], fill_value=0
    ).to_csv(out / 'note3_post_agreement_matrix.csv')
    separation = float(pdist(space['km'].cluster_centers_).mean())
    drift = float(centers.centroid_drift.mean())
    summary = dict(
        retained_pcs=int(space['pca'].n_components_),
        pca_explained_variance=float(space['pca'].explained_variance_ratio_.sum()),
        centroid_drifts=centers[['cluster', 'centroid_drift']].to_dict('records'),
        mean_baseline_intercentroid_distance=separation,
        mean_matched_centroid_drift=drift,
        drift_to_baseline_intercentroid_ratio=drift / separation,
        adjusted_rand_index=float(
            adjusted_rand_score(
                assigned.fixed_baseline_cluster, assigned.independent_post_cluster_matched
            )
        ),
        agreement_after_centroid_matching=float(
            (assigned.fixed_baseline_cluster == assigned.independent_post_cluster_matched).mean()
        ),
        matching='Hungarian assignment minimizing total Euclidean distance in fixed baseline PCA space',
    )
    write_json(out / 'note3_temporal_consistency.json', summary)
    return summary


def comparison(outcome, pairs, definition, group_names):
    outcome = np.asarray(outcome, dtype=int)
    group = pairs.group.to_numpy()
    n1, n0 = (int((group == a).sum()) for a in group_names)
    a, c = (int(outcome[group == arm].sum()) for arm in group_names)
    tab = np.array([[a, n1 - a], [c, n0 - c]])
    chi2, p, df, expected = stats.chi2_contingency(tab, correction=False)
    rr = a / n1 / (c / n0)
    se = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    lo, hi = np.exp(np.log(rr) + np.array([-1, 1]) * stats.norm.ppf(0.975) * se)
    return dict(
        definition=definition,
        intervention_favorable=a,
        intervention_n=n1,
        intervention_proportion=a / n1,
        control_favorable=c,
        control_n=n0,
        control_proportion=c / n0,
        unadjusted_rr=rr,
        rr_ci_low=lo,
        rr_ci_high=hi,
        chi_square=chi2,
        df=int(df),
        p_value=p,
        test='Pearson chi-square, no continuity correction',
        rr_ci_method='Normal CI for log(RR), no continuity correction',
    )


def table_s11(pairs, out, group_names):
    definitions = {
        'original_improvement_or_maintenance_C1_C2': pairs.favorable,
        'post_C1_or_C2_regardless_of_baseline': pairs.cluster_post.isin([1, 2]),
        'improvement_only': pairs.cluster_post.lt(pairs.cluster_pre),
    }
    rows = [
        comparison(outcome, pairs, name, group_names=group_names)
        for name, outcome in definitions.items()
    ]
    pd.DataFrame(rows).to_csv(out / 'table_s11_alternative_response_definitions.csv', index=False)
    return rows


def paired_labels(space, participants):
    pairs = (
        space['pre'][['participant_id', 'cluster']]
        .rename(columns={'cluster': 'cluster_pre'})
        .merge(
            space['post'][['participant_id', 'cluster']].rename(
                columns={'cluster': 'cluster_post'}
            ),
            on='participant_id',
        )
        .merge(participants, on='participant_id')
    )
    pairs['favorable'] = (
        pairs.cluster_post.lt(pairs.cluster_pre)
        | pairs.cluster_post.eq(pairs.cluster_pre) & pairs.cluster_pre.isin([1, 2])
    ).astype(int)
    return pairs


def nonlinear_sensitivity(gait, participants, features, config, out, group_names):
    excluded = [f for f in features if f.startswith('sample_entropy_') or f.startswith('lyapunov_')]
    reduced_features = [f for f in features if f not in excluded]
    space = fitted_space(gait, reduced_features, config)
    pairs = paired_labels(space, participants)
    pairs.to_csv(out / 'reduced23_cluster_assignments.csv', index=False)
    variance = space['pca'].explained_variance_ratio_
    pd.DataFrame(
        dict(
            component=np.arange(1, len(variance) + 1),
            variance_fraction=variance,
            cumulative_fraction=np.cumsum(variance),
        )
    ).to_csv(out / 'reduced23_pca_variance.csv', index=False)
    profiles = space['pre'].groupby('cluster')[reduced_features].mean()
    profiles.insert(0, 'n', space['pre'].groupby('cluster').size())
    profiles.to_csv(out / 'reduced23_baseline_cluster_profiles.csv')
    transitions = []
    for arm in group_names:
        arm_pairs = pairs[pairs.group.eq(arm)]
        matrix = pd.crosstab(arm_pairs.cluster_pre, arm_pairs.cluster_post).reindex(
            index=[1, 2, 3], columns=[1, 2, 3], fill_value=0
        )
        matrix.to_csv(out / f'reduced23_transitions_{arm}.csv')
        for i in [1, 2, 3]:
            for j in [1, 2, 3]:
                n = int(matrix.loc[i].sum())
                transitions.append(
                    dict(
                        group=arm,
                        cluster_pre=i,
                        cluster_post=j,
                        n=int(matrix.loc[i, j]),
                        baseline_cluster_n=n,
                        proportion=float(matrix.loc[i, j] / n) if n else None,
                    )
                )
    pd.DataFrame(transitions).to_csv(out / 'reduced23_transition_counts.csv', index=False)
    unadjusted = comparison(
        pairs.favorable, pairs, 'reduced23_original_response_rule', group_names=group_names
    )
    fits = {'poisson_independence': gee_sensitivity(pairs, out, 'reduced23_poisson_independence')}
    summary = dict(
        excluded_features=excluded,
        retained_features=reduced_features,
        retained_pcs=int(space['pca'].n_components_),
        pca_explained_variance=float(variance.sum()),
        baseline_counts=space['pre'].cluster.value_counts().sort_index().to_dict(),
        post_counts=space['post'].cluster.value_counts().sort_index().to_dict(),
        unadjusted=unadjusted,
        gee=fits,
        label_order='Descending baseline mean gait_speed; k fixed at 3; frozen baseline transforms assign post-test',
    )
    write_json(out / 'note8_reduced23_sensitivity.json', summary)
    return summary


def bootstrap_jaccard_full(original, bootstrap_labels, k):
    """Match each original cluster over every original participant."""
    scores = []
    for c in range(1, k + 1):
        a = original == c
        scores.append(
            max(
                np.count_nonzero(a & (bootstrap_labels == b))
                / np.count_nonzero(a | (bootstrap_labels == b))
                for b in range(k)
            )
        )
    return scores


def trial_reliability(trials, space, participants, out, n_boot=1000, seed=20261012):
    """Apply the fixed session-mean baseline model to both walk trials."""
    from statsmodels.stats.contingency_tables import SquareTable
    from sklearn.metrics import cohen_kappa_score

    features = space['features']
    assigned = trials.sort_values(['participant_id', 'timepoint', 'trial']).copy()
    z = space['pca'].transform(space['scaler'].transform(assigned[features]))
    assigned['cluster'] = [space['mapping'][int(v)] for v in space['km'].predict(z)]
    assigned[['participant_id', 'timepoint', 'trial', 'cluster']].to_csv(
        out / 'trial_cluster_assignments.csv', index=False
    )
    wide = assigned.pivot(index='participant_id', columns=['timepoint', 'trial'], values='cluster')
    rows, matrices, boundaries = [], [], []
    rng = np.random.default_rng(seed)

    def agreement(a, b, label, categories):
        a, b = np.asarray(a), np.asarray(b)
        table = pd.crosstab(a, b).reindex(index=categories, columns=categories, fill_value=0)
        ks, agreements = [], []
        for draw in rng.integers(0, len(a), size=(n_boot, len(a))):
            aa, bb = a[draw], b[draw]
            po = np.mean(aa == bb)
            pe = sum(np.mean(aa == c) * np.mean(bb == c) for c in categories)
            ks.append((po - pe) / (1 - pe) if pe < 1 else np.nan)
            agreements.append(po)
        row = dict(
            comparison=label,
            n=len(a),
            agree=int((a == b).sum()),
            agreement=float((a == b).mean()),
            agreement_ci_low=float(np.quantile(agreements, 0.025)),
            agreement_ci_high=float(np.quantile(agreements, 0.975)),
            kappa=float(cohen_kappa_score(a, b, labels=categories)),
            kappa_ci_low=float(np.nanquantile(ks, 0.025)),
            kappa_ci_high=float(np.nanquantile(ks, 0.975)),
            valid_kappa_bootstraps=int(np.isfinite(ks).sum()),
            ci_method='Percentile participant-pair bootstrap; no facility resampling',
        )
        if len(categories) == 3:
            test = SquareTable(table.to_numpy(), shift_zeros=False).homogeneity(
                method='stuart_maxwell'
            )
            row.update(
                stuart_maxwell_statistic=float(test.statistic), stuart_maxwell_p=float(test.pvalue)
            )
        rows.append(row)
        for c in categories:
            for d in categories:
                matrices.append(dict(comparison=label, trial1=c, trial2=d, n=int(table.loc[c, d])))

    for visit in ['pre', 'post']:
        a, b = wide[(visit, 1)], wide[(visit, 2)]
        agreement(a, b, visit + '_cluster', [1, 2, 3])
        for c, d in [(1, 2), (1, 3), (2, 3)]:
            count = int((((a == c) & (b == d)) | ((a == d) & (b == c))).sum())
            boundaries.append(
                dict(
                    timepoint=visit,
                    cluster_a=c,
                    cluster_b=d,
                    disagreements=count,
                    total_disagreements=int((a != b).sum()),
                )
            )
    pair_sets, comparisons = {}, []
    for trial in [1, 2]:
        data = wide.loc[:, [('pre', trial), ('post', trial)]].copy()
        data.columns = ['cluster_pre', 'cluster_post']
        data = data.reset_index().merge(participants, on='participant_id')
        data['favorable'] = (
            data.cluster_post.lt(data.cluster_pre)
            | (data.cluster_post.eq(data.cluster_pre) & data.cluster_pre.isin([1, 2]))
        ).astype(int)
        data.to_csv(out / f'trial{trial}_response_assignments.csv', index=False)
        pair_sets[trial] = data.set_index('participant_id').sort_index()
        row = comparison(data.favorable, data, f'trial_{trial}', ['intervention', 'control'])
        adjusted = gee_sensitivity(data, out, f'trial{trial}_gee')
        c = adjusted['arm_coefficient']
        row.update(
            adjusted_rr=c['exponentiated_coefficient'],
            adjusted_ci_low=c['ratio_ci_low'],
            adjusted_ci_high=c['ratio_ci_high'],
            adjusted_p=c['p_value'],
            gee_df=adjusted['df'],
        )
        row['gee_status'] = adjusted['status']
        comparisons.append(row)
    a, b = pair_sets[1], pair_sets[2]
    for group in ['all', 'intervention', 'control']:
        mask = np.ones(len(a), dtype=bool) if group == 'all' else a.group.eq(group).to_numpy()
        agreement(a.loc[mask, 'favorable'], b.loc[mask, 'favorable'], 'response_' + group, [0, 1])
    pd.DataFrame(rows).to_csv(out / 'table_s5_trial_agreement.csv', index=False)
    pd.DataFrame(matrices).to_csv(out / 'trial_agreement_matrices.csv', index=False)
    pd.DataFrame(boundaries).to_csv(out / 'trial_boundary_disagreements.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out / 'trial_response_group_comparisons.csv', index=False)
    return rows, comparisons


def fragility_analysis(pairs, out):
    group = pairs.groupby('group').favorable.agg(['sum', 'count'])
    a, n1 = int(group.loc['intervention', 'sum']), int(group.loc['intervention', 'count'])
    c, n0 = int(group.loc['control', 'sum']), int(group.loc['control', 'count'])
    rows = []
    for direction, maximum in [
        ('intervention_favorable_to_unfavorable', a),
        ('control_unfavorable_to_favorable', n0 - c),
    ]:
        for changes in range(maximum + 1):
            aa, cc = (a - changes, c) if direction.startswith('intervention') else (a, c + changes)
            table = np.array([[aa, n1 - aa], [cc, n0 - cc]])
            p_chi = stats.chi2_contingency(table, correction=False).pvalue
            p_fisher = stats.fisher_exact(table, alternative='two-sided').pvalue
            rows.append(
                dict(
                    direction=direction,
                    changed=changes,
                    intervention_favorable=aa,
                    control_favorable=cc,
                    pearson_p=float(p_chi),
                    fisher_two_sided_p=float(p_fisher),
                )
            )
            if p_chi >= 0.05 and p_fisher >= 0.05:
                break
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'fragility_reclassification.csv', index=False)
    return frame


def shap_difference_ci(shap_pre, shap_post, pairs, feature_names, out, n_boot=1000, seed=20261027):
    """Bootstrap paired Cluster 3 SHAP differences with the model held fixed."""
    panels = {
        ('intervention', 1): [
            'stride_frequency_variability_v',
            'stride_frequency_variability_ml',
            'rms_acceleration',
            'stride_frequency_variability_ap',
            'stride_regularity_ml',
        ],
        ('intervention', 0): [
            'stride_frequency_variability_v',
            'stride_regularity_all',
            'stride_time_variability',
            'stride_regularity_v',
            'stride_regularity_ap',
        ],
        ('control', 1): [
            'stride_frequency_variability_ml',
            'gait_symmetry_index',
            'stride_regularity_all',
            'stride_regularity_v',
            'stride_frequency_variability_ap',
        ],
        ('control', 0): [
            'stride_regularity_all',
            'stride_regularity_ap',
            'stride_time_variability',
            'gait_speed_variability',
            'stride_frequency_variability_ml',
        ],
    }
    delta = (shap_pre[:, :, 2] - shap_post[:, :, 2]).astype(float)
    rng = np.random.default_rng(seed)
    rows, individual = [], []
    for i, person in pairs.reset_index(drop=True).iterrows():
        for j, f in enumerate(feature_names):
            individual.append(
                dict(
                    participant_id=person.participant_id,
                    group=person.group,
                    favorable=int(person.favorable),
                    feature=f,
                    delta_pre_minus_post=float(delta[i, j]),
                )
            )
    for (group, response), selected in panels.items():
        mask = (pairs.group.eq(group) & pairs.favorable.eq(response)).to_numpy()
        values = delta[mask]
        draw = rng.integers(0, len(values), size=(n_boot, len(values)))
        means = values.mean(axis=0)
        lower, upper = np.quantile(values[draw].mean(axis=1), [0.025, 0.975], axis=0)
        order = pd.Series(means, index=feature_names).rank(
            method='first', ascending=not bool(response)
        )
        for j, f in enumerate(feature_names):
            rows.append(
                dict(
                    group=group,
                    response='favorable' if response else 'unfavorable',
                    n=len(values),
                    feature=f,
                    mean_delta_shap=float(means[j]),
                    ci_low=float(lower[j]),
                    ci_high=float(upper[j]),
                    includes_zero=bool(lower[j] <= 0 <= upper[j]),
                    current_directional_rank=int(order[f]),
                    displayed=f in selected,
                    display_order=selected.index(f) + 1 if f in selected else None,
                )
            )
    pd.DataFrame(individual).to_csv(out / 'shap_individual_C3_differences.csv', index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'shap_difference_ci_all_outcomes.csv', index=False)
    shown = frame[frame.displayed].copy()
    shown.to_csv(out / 'shap_difference_ci_displayed.csv', index=False)
    write_json(
        out / 'shap_difference_ci_method.json',
        dict(
            reference_cluster=3,
            repetitions=n_boot,
            seed=seed,
            method='Within arm-response strata, resample participants with replacement; paired pre/post difference; percentile CI.',
            display='Fixed outcomes and display order for Figures 3 and S5; no selection by confidence interval.',
            limits='Pointwise exploratory intervals condition on the fitted model and selected outcomes; facility dependence and model-fitting uncertainty are not included.',
        ),
    )
    return shown
