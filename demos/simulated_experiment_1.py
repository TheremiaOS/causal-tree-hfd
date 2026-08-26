import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import shap

from causaltreehfd.causaltreehfd import CausalTreeHFD
from causaltreehfd.forest_structure import fit_grf
from causaltreehfd.utils import build_random_forest_from_df

if __name__ == "__main__":

    print("Generating experimental data...")

    d = 8
    n = 2000
    sigma = .5
    seed = 1234
    rng = np.random.default_rng(seed)

    X = rng.normal(loc=0, scale=1, size=(n, d))
    e = 0.4 + 0.2 * np.where(X[:, 0] >= 0, 1, 0)
    m0 = (X[:, 2] * X[:, 3])**2
    tau = np.where((X[:, 0] >= 0) & (X[:, 1] >= 0), 1, 0)
    W = rng.binomial(1, e).ravel()
    noise = rng.normal(0, sigma, n)
    Y = m0 + W*tau + noise

    print("Fitting causal forest to data and extracting forest structure...")
    num_trees = 2000
    min_node_size = 5
    forest_df, tau_hat = fit_grf(X, Y, W, num_trees, min_node_size, seed=seed)

    print("Fitting CausalTreeHFD...")
    CTHFD_model = CausalTreeHFD(forest_df, depth_variable=(10, 10))
    CTHFD_model.fit(X)

    print("Computing decomposition...")
    tau_main, tau_order2 = CTHFD_model.predict(X)

    print("Computing SHAP...")
    rf = build_random_forest_from_df(forest_df, d)
    rf_explainer = shap.TreeExplainer(rf, X, feature_perturbation="tree_path_dependent")
    shap_interaction_values = rf_explainer.shap_interaction_values(X)

    print("Computing prediction performance...")
    tau1 = .5 * (np.where(X[:, 0] >= 0, 1, 0) - .5)
    tau2 = .5 * (np.where(X[:, 1] >= 0, 1, 0) - .5)
    tau12 = (np.where(X[:, 0] >= 0, 1, 0) - .5) * (np.where(X[:, 1] >= 0, 1, 0) - .5)
    idx12 = np.where(np.all(CTHFD_model.interaction_list == (0, 1), axis=1))[0][0]
    R2 = [r2_score(tau1, tau_main[:, 0]), r2_score(tau2, tau_main[:, 1]), r2_score(tau12, tau_order2[:, idx12])]
    R2_shap = [r2_score(tau1, shap_interaction_values[:, 0, 0]), r2_score(tau2, shap_interaction_values[:, 1, 1]), r2_score(tau12, shap_interaction_values[:, 0, 1] + shap_interaction_values[:, 1, 0])]

    print("Plotting results...")
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    sort_idx = np.argsort(X[:, 0])
    axs[0].plot(
        X[sort_idx, 0], 
        tau1[sort_idx], 
        color='black', 
        linewidth=2, 
        label='Ground truth'
    )
    axs[0].scatter(X[:, 0], tau_main[:, 0], s=5, color='red', label='CausalTreeHFD')
    axs[0].scatter(X[:, 0], shap_interaction_values[:, 0, 0], s=5, color='blue', label=f'TreeSHAP R2 = {np.round(R2_shap[0], 3)}')
    axs[0].legend(loc='upper left')
    axs[0].set_xlabel(r"$X_1$")
    axs[0].set_title(f"$R^2 = {R2[0]:.3f}$")
    axs[0].grid(True, linestyle=':', alpha=0.4)

    sort_idx = np.argsort(X[:, 1])
    axs[1].plot(
        X[sort_idx, 1], 
        tau2[sort_idx], 
        color='black', 
        linewidth=2, 
        label='Ground truth'
    )
    axs[1].scatter(X[:, 1], tau_main[:, 1], s=5, color='red', label='CausalTreeHFD')
    axs[1].scatter(X[:, 1], shap_interaction_values[:, 1, 1], s=5, color='blue', label=f'TreeSHAP R2 = {np.round(R2_shap[1], 3)}')
    axs[1].legend(loc='upper left')
    axs[1].set_xlabel(r"$X_2$")
    axs[1].set_title(f"$R^2 = {R2[1]:.3f}$")
    axs[1].grid(True, linestyle=':', alpha=0.4)

    sc = axs[2].scatter(
        X[:, 0], 
        X[:, 1], 
        c=tau_order2[:, idx12],
        cmap='coolwarm', 
        s=15, 
        alpha=0.8,
        edgecolors='none'
    )
    cbar = fig.colorbar(sc, ax=axs[2])
    cbar.set_label(r'$\hat{\tau}_{12}$', fontsize=10)
    axs[2].set_xlabel(r'$X_1$')
    axs[2].set_ylabel(r'$X_2$')
    axs[2].set_title(f"Interaction Term ($R^2 = {R2[2]:.3f}$) SHAP : ($R^2 = {R2_shap[2]:.3f}$)", fontsize=12)
    axs[2].grid(True, linestyle=':', alpha=0.4)

    fig.savefig("output/figures/treehfd_simulated_data_1.png")

    fig, axs = plt.subplots(2, 3, figsize=(12, 12))

    axs[0, 0].scatter(X[:, 0], tau_main[:, 0], s=5, color='red')
    mse0 = mean_squared_error(tau1, tau_main[:, 0])
    axs[0, 0].set_xlabel(f'$X_1$, MSE = {np.round(mse0, 3)}')

    axs[0, 1].scatter(X[:, 1], tau_main[:, 1], s=5, color='red')
    mse1 = mean_squared_error(tau2, tau_main[:, 1])
    axs[0, 1].set_xlabel(f'$X_2$, MSE = {np.round(mse1, 3)}')

    axs[0, 2].scatter(X[:, 7], tau_main[:, 7], s=5, color='red')
    mse7 = mean_squared_error(np.zeros(n), tau_main[:, 7])
    axs[0, 2].set_xlabel(f'$X_8$, MSE = {np.round(mse7, 3)}')

    axs[1, 0].scatter(X[:, 2], tau_main[:, 2], s=5, color='red')
    mse2 = mean_squared_error(np.zeros(n), tau_main[:, 2])
    axs[1, 0].set_xlabel(f'$X_3$, MSE = {np.round(mse2, 3)}')

    axs[1, 1].scatter(X[:, 3], tau_main[:, 3], s=5, color='red')
    mse3 = mean_squared_error(np.zeros(n), tau_main[:, 3])
    axs[1, 1].set_xlabel(f'$X_4$, MSE = {np.round(mse3, 3)}')

    axs[1, 2].scatter(X[:, 4], tau_main[:, 4], s=5, color='red')
    mse4 = mean_squared_error(np.zeros(n), tau_main[:, 4])
    axs[1, 2].set_xlabel(f'$X_5$, MSE = {np.round(mse4, 3)}')

    fig.savefig("output/figures/main_components_exp_1.png")
    plt.show()

    print("Done !")