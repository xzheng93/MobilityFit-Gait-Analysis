import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch, Rectangle, Patch
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette


def save_figure(fig, filename, save_path):
    from datetime import datetime

    now = datetime.now().astimezone()
    fig.savefig(
        save_path / filename,
        dpi=180,
        bbox_inches='tight',
        metadata={'Author': 'xiaoping zheng', 'Software': 'xiaoping zheng'},
    )
    fig.savefig(
        (save_path / filename).with_suffix('.pdf'),
        bbox_inches='tight',
        metadata={
            'Author': 'xiaoping zheng',
            'Creator': 'xiaoping zheng',
            'Producer': 'xiaoping zheng',
            'CreationDate': now,
            'ModDate': now,
        },
    )
    plt.close(fig)


def plot_loadings(loading, ordered_features, pc, save_path):
    fig, ax = plt.subplots(figsize=(10, 12))
    matrix = loading.loc[ordered_features].to_numpy()
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(pc)), loading.columns)
    ax.set_yticks(
        np.arange(len(ordered_features)), [readable_label(f) for f in ordered_features], fontsize=8
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f'{matrix[i, j]:.2f}',
                ha='center',
                va='center',
                fontsize=7,
                color='white' if abs(matrix[i, j]) > 0.55 else 'black',
            )
    fig.colorbar(im, ax=ax, shrink=0.5, label='Variable–PC correlation')
    save_figure(fig, 'figure_s2_loadings.png', save_path=save_path)


def plot_pca(pc, variance, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(pc, variance.variance_fraction * 100, color='#4f93bd', label='Variance explained')
    ax.set(xlabel='Principal component', ylabel='Variance explained (%)', xticks=pc)
    right = ax.twinx()
    right.plot(pc, variance.cumulative_fraction * 100, 'o-', color='#df2626', label='Cumulative')
    right.axhline(85, color='#aaaaaa', linestyle='--')
    right.set(ylabel='Cumulative variance (%)', ylim=(0, 110))
    save_figure(fig, 'figure_s1_pca.png', save_path=save_path)


def plot_dendrogram(zpre, cluster_colors, save_path):
    ward = linkage(zpre, method='ward')
    threshold = (ward[-3, 2] + ward[-2, 2]) / 2
    fig, ax = plt.subplots(figsize=(10, 5))
    set_link_color_palette(list(cluster_colors.values()))
    dendrogram(
        ward, color_threshold=threshold, above_threshold_color='#888888', no_labels=True, ax=ax
    )
    set_link_color_palette(None)
    ax.set(xlabel='Participants', ylabel='Ward linkage distance')
    save_figure(fig, 'figure_s4_ward.png', save_path=save_path)


def plot_cluster_diagnostics(diagnostics, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(diagnostics.k, diagnostics.silhouette, 'o-', color='#2684b7', label='Silhouette')
    ax.set(
        xlabel='Number of clusters', ylabel='Silhouette (higher is better)', xticks=diagnostics.k
    )
    right = ax.twinx()
    right.plot(diagnostics.k, diagnostics.davies_bouldin, 's-', color='#da3131', label='DBI')
    right.set_ylabel('Davies–Bouldin index (lower is better)')
    ax.legend(loc='upper left')
    right.legend(loc='upper right')
    save_figure(fig, 'cluster_diagnostics.png', save_path=save_path)


def plot_global_shap(global_shap, save_path):
    top = global_shap.head(15).sort_values('mean_absolute_shap')
    fig, ax = plt.subplots(figsize=(10, 7))
    positions = np.arange(len(top))
    ax.barh(positions, top.mean_absolute_shap, color='#b5b5b5', height=0.64)
    ax.hlines(positions, top.shap_ci_low, top.shap_ci_high, color='#333333')
    ax.set_yticks(positions, [readable_label(f) for f in top.feature])
    ax.set_xlabel('Mean |SHAP| with 95% participant-bootstrap CI')
    save_figure(fig, 'figure_s6_global_shap.png', save_path=save_path)


def plot_feature_distributions(
    selected_features, gait, participants, dictionary, resample_seed, save_path
):
    unit = dictionary.set_index('variable').unit.to_dict()
    intervention = gait.merge(participants[['participant_id', 'group']], on='participant_id')
    intervention = intervention[intervention.group.eq('intervention')]
    rng = np.random.default_rng(resample_seed)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, feature in zip(axes.flat, selected_features):
        for position, visit, color in [(0, 'pre', '#c8c8c8'), (1, 'post', '#8ac9b7')]:
            values = intervention.loc[intervention.timepoint.eq(visit), feature].to_numpy()
            violin = ax.violinplot(values, [position + 0.14], widths=0.52, showextrema=False)
            for body in violin['bodies']:
                coordinates = body.get_paths()[0].vertices
                coordinates[:, 0] = np.maximum(coordinates[:, 0], position + 0.14)
                body.set_facecolor(color)
                body.set_edgecolor('none')
                body.set_alpha(1)
            ax.boxplot(
                values,
                positions=[position],
                widths=0.13,
                showfliers=False,
                patch_artist=True,
                boxprops={'facecolor': color},
                medianprops={'color': '#4b4b4b'},
            )
            ax.scatter(
                position + rng.uniform(-0.26, -0.11, len(values)),
                values,
                s=12,
                color=color,
                alpha=0.9,
            )
        label = readable_label(feature)
        if unit[feature] != 'dimensionless':
            label += f' ({unit[feature]})'
        ax.set(
            xticks=[0, 1], xticklabels=['Pre-test', 'Post-test'], xlim=(-0.38, 1.55), ylabel=label
        )
    axes.flat[-1].axis('off')
    fig.tight_layout()
    save_figure(fig, 'figure4_raw_distributions.png', save_path=save_path)


def plot_transitions(matrices, group_names, cluster_colors, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 6))
    for ax, group in zip(axs, group_names):
        m = matrices[group]
        left = m.sum(1)
        right = m.sum(0)
        gap = 3
        tops_left = np.r_[0, np.cumsum(left[:-1] + gap)]
        tops_right = np.r_[0, np.cumsum(right[:-1] + gap)]
        used_left = np.zeros(3)
        used_right = np.zeros(3)
        for i in range(3):
            for j in range(3):
                n = m[i, j]
                if not n:
                    continue
                a = tops_left[i] + used_left[i]
                b = tops_right[j] + used_right[j]
                vertices = [
                    (0.15, a),
                    (0.4, a),
                    (0.6, b),
                    (0.85, b),
                    (0.85, b + n),
                    (0.6, b + n),
                    (0.4, a + n),
                    (0.15, a + n),
                    (0.15, a),
                ]
                codes = [
                    PlotPath.MOVETO,
                    PlotPath.CURVE4,
                    PlotPath.CURVE4,
                    PlotPath.CURVE4,
                    PlotPath.LINETO,
                    PlotPath.CURVE4,
                    PlotPath.CURVE4,
                    PlotPath.CURVE4,
                    PlotPath.CLOSEPOLY,
                ]
                ax.add_patch(
                    PathPatch(
                        PlotPath(vertices, codes),
                        facecolor=cluster_colors[i + 1],
                        alpha=0.4,
                        edgecolor='white',
                        lw=0.5,
                    )
                )
                used_left[i] += n
                used_right[j] += n
        for i in range(3):
            ax.add_patch(
                Rectangle((0.105, tops_left[i]), 0.045, left[i], color=cluster_colors[i + 1])
            )
            ax.add_patch(
                Rectangle((0.85, tops_right[i]), 0.045, right[i], color=cluster_colors[i + 1])
            )
            ax.text(
                0.08, tops_left[i] + left[i] / 2, f'C{i + 1}\nn={left[i]}', ha='right', va='center'
            )
            ax.text(
                0.92,
                tops_right[i] + right[i] / 2,
                f'C{i + 1}\nn={right[i]}',
                ha='left',
                va='center',
            )
        ax.set(
            xlim=(-0.02, 1.1),
            ylim=(m.sum() + 2 * gap + 2, -4),
            title=f'{group.title()} (n={m.sum()})',
        )
        ax.text(0.12, -1, 'Pre', ha='center', weight='bold')
        ax.text(0.87, -1, 'Post', ha='center', weight='bold')
        ax.axis('off')
    fig.suptitle(
        'Gait-pattern transitions',
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, 'figure2_transitions.png', save_path=out)


def readable_label(feature):
    labels = {
        'rms_acceleration': 'Dynamic root mean square acceleration',
        'gait_symmetry_index': 'Gait symmetry index',
        'stride_frequency': 'Stride frequency',
        'index_harmonicity': 'Index of harmonicity',
        'lyapunov': 'Maximal Lyapunov exponent',
        'lyapunov_per_stride': 'Maximal Lyapunov exponent per stride',
        'lyapunov_per_second': 'Maximal Lyapunov exponent per second',
    }
    base = feature
    axis = ''
    for suffix, label in [('_all', 'All'), ('_v', 'V'), ('_ml', 'ML'), ('_ap', 'AP')]:
        if feature.endswith(suffix):
            base = feature[: -len(suffix)]
            axis = f' ({label})'
            break
    return labels.get(base, base.replace('_', ' ').capitalize()) + axis


def plot_shap_intervals(frame, save_path):
    """Plot SHAP differences and 95% CIs without bar-value labels."""
    for group, title, name in [
        ('intervention', 'Intervention group', 'figure3_shap'),
        ('control', 'Control group', 'figure_s5_shap'),
    ]:
        data = frame[frame.group.eq(group)].copy()
        data['panel_order'] = data.response.map({'favorable': 0, 'unfavorable': 1})
        data = data.sort_values(['panel_order', 'display_order']).reset_index(drop=True)
        y = np.arange(len(data))
        means = data.mean_delta_shap.to_numpy()
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ['#60ad95' if r == 'favorable' else '#818181' for r in data.response]
        ax.barh(y, means, color=colors, height=0.62)
        ax.hlines(y, data.ci_low, data.ci_high, color='#333333', linewidth=1.3)
        ax.plot(means, y, 'o', color='#333333', markersize=3)
        for yy, lo, hi in zip(y, data.ci_low, data.ci_high):
            ax.vlines([lo, hi], yy - 0.07, yy + 0.07, color='#333333', linewidth=1.3)
        ax.set_yticks(y, [readable_label(f) for f in data.feature])
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=1)
        ax.axhline(4.5, color='#cccccc', linestyle='--')
        ax.set(xlabel='Mean SHAP difference (pre-test − post-test), Cluster 3', title=title)
        nf = int(data.loc[data.response.eq('favorable'), 'n'].iloc[0])
        nu = int(data.loc[data.response.eq('unfavorable'), 'n'].iloc[0])
        ax.legend(
            handles=[
                Patch(color='#60ad95', label=f'Favorable response (n={nf})'),
                Patch(color='#818181', label=f'Unfavorable response (n={nu})'),
            ],
            loc='best',
        )
        save_figure(fig, name + '.png', save_path)
