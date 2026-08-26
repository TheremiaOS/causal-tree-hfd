import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from causaltreehfd.causaltreehfd import CausalTreeHFD
from causaltreehfd.forest_structure import fit_grf

if __name__ == "__main__":

    seed = 0

    print("Loading and preprocessing data...")
    # Load data
    data = pd.read_csv("data/STAR.csv")

    # Preprocess data
    periods = pd.PeriodIndex(data['birth'].str.replace(' ', ''), freq='Q')
    data['year'] = periods.year
    data['quarter'] = periods.quarter

    cols = ['gender', 'ethnicity', 'year', 'quarter', 'star1', 'read1', 'math1', 'lunch1', 'school1', 'degree1', 'ladder1', 'experience1', 'tethnicity1', 'system1', 'schoolid1', 'readk', 'mathk']

    star_map = {
        'regular': 0,
        'regular+aide': 0,
        'small': 1
    }
    data['star1'] = data['star1'].map(star_map)

    df = data.dropna(subset=cols)[cols]

    print("Total number of samples : ", df.shape[0])
    print("Number of treated samples : ", int(df["star1"].sum()))
    print("Number of controls : ", int((1 - df["star1"]).sum()))

    level_map = {
        'apprentice': 1,
        'probation': 2,
        'level1': 3,
        'level2': 4,
        'level3': 5,
        'notladder': 0,
    }
    df['ladder1'] = df['ladder1'].map(level_map)

    degree_map = {
        'bachelor': 1,
        'specialist': 2,
        'master': 3,
        'phd': 4,
    }
    df['degree1'] = df['degree1'].map(degree_map)

    school_map = {
        'rural': 1,
        'suburban': 2,
        'urban': 3,
        'inner-city': 4,
    }
    df['school1'] = df['school1'].map(school_map)

    """
    df = pd.get_dummies(
        df, 
        columns=['school1'], 
        drop_first=False,
        dtype=int
    )
    """

    df['ethnicity'] = (df['ethnicity'] == 'cauc').astype(int)
    df['tethnicity1'] = (df['tethnicity1'] == 'cauc').astype(int)
    df['gender'] = (df['gender'] == 'female').astype(int)
    df['lunch1'] = (df['lunch1'] == 'free').astype(int)

    X_df = df.drop(columns=['star1', 'read1', 'math1', 'system1', 'schoolid1'])
    X = np.array(X_df)
    W = np.array(df['star1'])
    Y = np.array(df['math1'] + df['read1'] - df['mathk'] - df['readk'])
    clusters = np.array(df['schoolid1'])

    print("Fitting causal forest to data and extracting forest structure...")
    num_trees = 2000
    min_node_size = 5
    forest_df, tau_hat = fit_grf(X, Y, W, num_trees=num_trees, min_node_size=min_node_size, clusters=clusters, seed=seed)

    print("Fitting CausalTreeHFD...")
    CTHFD_model = CausalTreeHFD(forest_df, depth_variable=(5, 5), max_depth=10)
    CTHFD_model.fit(X)

    print("Computing decomposition...")
    tau_main, tau_order2 = CTHFD_model.predict(X)

    print("Computing variance matrix...")
    var1 = np.var(tau_main, axis=0)
    var2 = np.var(tau_order2, axis=0)
    n_features = X_df.shape[1]
    covariates = X_df.columns

    S2_mat = np.zeros((n_features, n_features))
    for val, (i,j) in zip(var2, CTHFD_model.interaction_list):
        S2_mat[i,j] = val
        S2_mat[j,i] = val
    for i in range(n_features):
        S2_mat[i,i] = var1[i]

    corr_df = pd.DataFrame(S2_mat, index=covariates, columns=covariates)

    fig, ax = plt.subplots(figsize=(6, 8))
    cmap = plt.get_cmap("coolwarm")

    mask = np.zeros_like(S2_mat, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    ax = sns.heatmap(corr_df.round(2), ax=ax, mask=mask, annot_kws={'fontsize':5}, cmap=cmap)
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10)

    fig.savefig("output/figures/STAR_variance_matrix.png")

    print("Plotting main components...")

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 18))

    tau_max = np.max(np.abs(tau_main))*1.1

    for i in range(3):
        for j in range(4):
            idx = i*4+j
            x, y = X[:, idx], tau_main[:, idx]

            if len(np.unique(x)) <= 15:
                axes[i, j].bar(x, y, color='red', alpha=1, width=0.1, edgecolor="black", linewidth=3)
                #axes[i, j].scatter(x, y, s=10)
            else:
                axes[i, j].scatter(x, y, alpha=1, c="red", s=5)

            axes[i, j].set_xlabel(covariates[idx])
            axes[i, j].set_ylim(-tau_max, tau_max)
            axes[i, j].grid(alpha=.3)

    fig.savefig("output/figures/STAR_main_components.png")

    print('Done !')