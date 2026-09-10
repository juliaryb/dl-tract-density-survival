import pandas as pd
import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def concordance_index_ipcw_path(cox, comps_training, comps_testing, DATA_train, DATA_test, tau=None):
    """
    Compute C-index at every alpha on the regularisation path by using inverse probability of censoring weights (IPCW).
    """
    
    # Compute C-index at every alpha on the regularisation path
    cindex_ipcw_train_path = []
    cindex_ipcw_test_path  = []

    for alpha in cox.alphas_:
        # Predict risk scores at this specific alpha
        risk_train = cox.predict(comps_training, alpha=alpha)
        risk_test  = cox.predict(comps_testing,  alpha=alpha)

        ci_train = concordance_index_ipcw(survival_train=DATA_train, survival_test=DATA_train, estimate=risk_train, tau=tau)[0]
        ci_test  = concordance_index_ipcw(survival_train=DATA_train, survival_test=DATA_test, estimate=risk_test, tau=tau)[0]

        cindex_ipcw_train_path.append(ci_train)
        cindex_ipcw_test_path.append(ci_test)

    cindex_ipcw_train_path = np.array(cindex_ipcw_train_path)
    cindex_ipcw_test_path  = np.array(cindex_ipcw_test_path)

    return cindex_ipcw_train_path, cindex_ipcw_test_path, np.argmax(cindex_ipcw_test_path)

def concordance_index_path(cox, comps_training, comps_testing, DATA_train, DATA_test, duration_col, event_col):
    """
    Compute C-index at every alpha on the regularisation path.
    """
    
    # Compute C-index at every alpha on the regularisation path
    cindex_train_path = []
    cindex_test_path  = []

    for alpha in cox.alphas_:
        # Predict risk scores at this specific alpha
        risk_train = cox.predict(comps_training, alpha=alpha)
        risk_test  = cox.predict(comps_testing,  alpha=alpha)

        event_train = DATA_train[event_col]
        time_train  = DATA_train[duration_col]
        event_test  = DATA_test[event_col]
        time_test   = DATA_test[duration_col]
        ci_train = concordance_index_censored(event_train, time_train, risk_train)[0]
        ci_test  = concordance_index_censored(event_test,  time_test,  risk_test)[0]

        cindex_train_path.append(ci_train)
        cindex_test_path.append(ci_test)

    cindex_train_path = np.array(cindex_train_path)
    cindex_test_path  = np.array(cindex_test_path)

    return cindex_train_path, cindex_test_path, np.argmax(cindex_test_path)

def concordance_index_ipcw_path(cox, comps_training, comps_testing, DATA_train, DATA_test, tau=None):
    """
    Compute C-index at every alpha on the regularisation path by using inverse probability of censoring weights (IPCW).
    """
    
    # Compute C-index at every alpha on the regularisation path
    cindex_ipcw_train_path = []
    cindex_ipcw_test_path  = []

    for alpha in cox.alphas_:
        # Predict risk scores at this specific alpha
        risk_train = cox.predict(comps_training, alpha=alpha)
        risk_test  = cox.predict(comps_testing,  alpha=alpha)

        ci_train = concordance_index_ipcw(survival_train=DATA_train, survival_test=DATA_train, estimate=risk_train, tau=tau)[0]
        ci_test  = concordance_index_ipcw(survival_train=DATA_train, survival_test=DATA_test, estimate=risk_test, tau=tau)[0]

        cindex_ipcw_train_path.append(ci_train)
        cindex_ipcw_test_path.append(ci_test)

    cindex_ipcw_train_path = np.array(cindex_ipcw_train_path)
    cindex_ipcw_test_path  = np.array(cindex_ipcw_test_path)

    return cindex_ipcw_train_path, cindex_ipcw_test_path, np.argmax(cindex_ipcw_test_path)

def plot_shrinkage_coefficients(coefs, n_highlight, best_alpha):

    fig, ax = plt.subplots(figsize=(9, 6))
    alphas = coefs.columns
    top_coefs = coefs.loc[:, best_alpha].map(abs).sort_values().tail(n_highlight)

    for row, nm in zip(coefs.itertuples(), coefs.index):
        if nm in top_coefs.index:
            ax.semilogx(alphas, row[1:], ".-", alpha=0.5, linewidth=.75, markersize=3, color="salmon", label=None)
            coef = coefs.loc[nm, alphas.min()]
            plt.text(alphas.min(), coef, str(nm) + "   ", horizontalalignment="right", verticalalignment="center", fontsize=6)
        else:
            ax.semilogx(alphas, row[1:], ".-", alpha=0.5, linewidth=.75, markersize=3, color="gray", label=None)

    ax.axvline(best_alpha, color="red", linestyle="--", label="Best "+r"$\lambda$"+" at test")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("Regularization ("+r"$\lambda$"+")")
    ax.set_ylabel("Log-HRs ("+r"$\beta_k$"+")")
    ax.spines[['top','left']].set_visible(False)
    ax.legend(frameon=True)

    return fig, ax

def plot_concordance_index_path(cox, cindex_train_path, cindex_test_path):
    """
    Plot the concordance index for the training and testing set for a given CV split across the entire regularization path.
    """

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(cox.alphas_, cindex_train_path, ".-", label="Train", alpha=1, linewidth=.75, markersize=3, color="black")
    ax.semilogx(cox.alphas_, cindex_test_path, ".-", label="Test", alpha=0.5, linewidth=.75, markersize=3, color="black")
    ax.axvline(cox.alphas_[np.argmax(cindex_test_path)], color="red", linestyle="--", label="Best "+r"$\lambda$"+" at test")
    ax.set_xlabel("Regularization ("+r"$\lambda$"+")")
    ax.set_ylabel("C-index")
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim([0.5, .75])

    return fig, ax

def plot_concordance_index_ipcw_path(cox, cindex_ipcw_train_path, cindex_ipcw_test_path):
    """
    Plot the ipcw concordance index for the training and testing set for a given CV split across the entire regularization path.
    """

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(cox.alphas_, cindex_ipcw_train_path, ".-", label="Train", alpha=1, linewidth=.75, markersize=3, color="black")
    ax.semilogx(cox.alphas_, cindex_ipcw_test_path, ".-", label="Test", alpha=0.5, linewidth=.75, markersize=3, color="black")
    ax.axvline(cox.alphas_[np.argmax(cindex_ipcw_test_path)], color="tab:green", linestyle="--", label="Best "+r"$\lambda$"+" at test")
    ax.set_xlabel("Regularization ("+r"$\lambda$"+")")
    ax.set_ylabel("IPCW C-index")
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim([0.5, .75])

    return fig, ax

def plot_deviance_ratio_path(cox, best_alpha=None, max_DR=1):

    """
    Plot the concordance index for the training and testing set for a given CV split across the entire regularization path.
    """
    dr = cox.deviance_ratio_

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(cox.alphas_, dr, ".-", alpha=1, linewidth=.75, markersize=3, color="black")
    if best_alpha is not None:
        ax.axvline(best_alpha, color="red", linestyle="--", label="Best "+r"$\lambda$"+" at test")

    ax.set_xlabel("Regularization ("+r"$\lambda$"+")")
    ax.set_ylabel("Deviance ratio: "+r'$DR = 1 - \frac{deviance}{null_deviance}$')
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim([0, max_DR])

    return fig, ax

def plot_correlation_matrix(comps, labels):

    """
    comps: (N, P) array [z_1..z_dim, age, sex] (last column = sex).
    Pearson correlation for every pair except those involving sex (binary),
    which use Spearman rank correlation instead -- same convention already
    used by the console collinearity check above.
    """
    P = comps.shape[1]
    corr = np.corrcoef(comps[:, :-1].T)
    corr = np.pad(corr, ((0, 1), (0, 1)))
    for i in range(P - 1):
        rho, _ = spearmanr(comps[:, i], comps[:, -1])
        corr[i, -1] = corr[-1, i] = rho
    corr[-1, -1] = 1.0

    # Fixed label font size; figure grows with P (inches/label) so labels
    # stay legibly spaced at any latent dimensionality (SVG output stays
    # crisp at any zoom, so a larger physical canvas is the right lever
    # rather than shrinking the font).
    size = max(6, P * 0.22)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        corr, vmin=-1, vmax=1, cmap="coolwarm", square=True,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Correlation (Pearson; Spearman for sex)"},
        ax=ax,
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelrotation=0, labelsize=6)
    fig.tight_layout()

    return fig, ax

if __name__ == "__main__":

    ############################################################################################################################################
    ### INITIAL DATA AND PARAMETERS

    ##
    # Hint: YOU MIGHT CREATE A SUPER-LOOP OR SUPERSCRIPT THAT CALLS THIS SCRIPT MULTIPLE TIMES WITH DIFFERENT PARAMETERS 
    #       (E.G., LATENT DIMENSIONS, L1 RATIO, NUMBER OF FOLDS, ETC.) AND THEN COLLECTS THE RESULTS IN A SINGLE CSV FILE.
    ##
    
    ## General parameters
    latent_dimensions = 64            # Number of latent dimensions (LCs) to use in the Cox model
    fmt = "svg"                       # Figure format
    dpi = 200                         # Figure resolution in dots per inch
    daysXmonth = 365/12               # Conversion between days and months
    random_seed = 42                  # Random seed for reproducibility
    tau = None                        # Used in the calculation of the IPCW concordance index

    # General directories
    # root =  "/home/joan/Desktop/PROJECTS/Glioblastomas"
    root = "/home/joan/Desktop/PROJECTS/Julia/code/cv-latent-comp"
    os.makedirs(root, exist_ok=True)
    results_root = os.path.join(root,f"{root}/RESULTS-GBM_4-cohorts_Latent-components")
    os.makedirs(results_root, exist_ok=True)

    # Loading data
    DATA = pd.read_csv(f"{root}/clinical_latent_{latent_dimensions}.csv", sep=',').sort_values(by=["cohort", "id"]).reset_index(drop=True)
    duration_col = 'OS (days) - corrected'
    event_col = 'status'

    # Regularization path
    regularize_demographics = True   # If True, age and sex are included in the regularization path
    reg_precision = 6
    reg_steps = 50
    regularization_path = np.round(np.logspace(-1, -3, 50), reg_precision)
    L1_ratio = 0.8                    # L1 ratio for elastic net regularization (1.0 = LASSO, 0.0 = Ridge)

    ## KFold CV
    split_type = "rnd_stratified"     # Training and validation data split. Choices: ['rnd_stratified', 'kfold_stratified']
    repeated_kfolds = True            # If True, use RepeatedStratifiedKFold. If False, use StratifiedKFold
    n_splits = 5                      # Number of folds for cross-validation.
    n_repeats = 6                     # Number of repeats for RepeatedStratifiedK.
    plot_folds = True                 # Produce log-HR and C-indices along the regularization path

    save_dir = os.path.join(results_root, f"AutoEncoder-TD_Latent-{latent_dimensions}_L1-{L1_ratio}")

    ############################################################################################################################################
    ### LOADING TRACT DENSITY MAPS 
    columns = [f"z{i}" for i in range(1, latent_dimensions+1)]

    # UCSF cohort
    comps_ucsf = DATA.loc[DATA["cohort"] == 0, columns].values

    # UPENN cohort
    comps_upenn = DATA.loc[DATA["cohort"] == 1, columns].values

    # TCGA cohort
    comps_tcga = DATA.loc[DATA["cohort"] == 2, columns].values

    # RHUH cohort
    comps_rhuh = DATA.loc[DATA["cohort"] == 3, columns].values
    comps_all = np.vstack([comps_ucsf, comps_upenn, comps_tcga, comps_rhuh])

    print(f"INFO:   All latent scores have been loaded")

    ############################################################################################################################################
    ### REPEATED AND STRATIFIED KFOLD CROSS-VALIDATION 

    # The stratification preserves 1) proportion of cohorts and 2) the proportion of events (censoring) in each cohort. 
    DATA["stratify key"] = DATA["cohort"].astype(str) + "_" + DATA[event_col].astype(str)
    if repeated_kfolds is True:
        kf_cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_seed
        )
        CVresults = np.empty(shape=(n_repeats * n_splits, ), dtype=object)
        if float(L1_ratio) == 1.0:
            save_dir = os.path.join(save_dir, f"KFold-Lasso_Repeated-Stratified")
        else:
            save_dir = os.path.join(save_dir, f"KFold-Elasticnet_L1-{L1_ratio}_Repeated-Stratified")

    else: 
        kf_cv = StratifiedKFold(
            n_splits=n_splits,
            random_state=random_seed,
            shuffle=True
        )
        CVresults = np.empty(shape=(n_splits, ), dtype=object)
        if float(L1_ratio) == 1.0:
            save_dir = os.path.join(save_dir, f"KFold-Lasso_Stratified")
        else:
            save_dir = os.path.join(save_dir, f"KFold-Elasticnet_L1-{L1_ratio}_Stratified")

    os.makedirs(save_dir, exist_ok=True)
    if plot_folds:
        os.makedirs(os.path.join(save_dir, "Lasso-Regularization_LogHazard-Ratios"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "Lasso-Regularization_Concordance-Indices"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "Lasso-Regularization_Deviance-Ratio"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "Lasso-Regularization_IPCW-Indices"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "Correlation-Matrix"), exist_ok=True)

    CVbaseline = np.zeros((n_splits*n_repeats if repeated_kfolds is True else n_splits, 4))         # Baseline Cox model results
    for fold_idx, (train_idx, test_idx) in enumerate(kf_cv.split(X=DATA, y=DATA["stratify key"])):
        print(f"\nINFO:   ++++++++++++++++++++ Fold {fold_idx+1}/{n_splits*n_repeats if repeated_kfolds is True else n_splits} ++++++++++++++++++++") 

        DATA_train = DATA.iloc[train_idx].copy()
        DATA_test = DATA.iloc[test_idx].copy()

        comps_training = comps_all[train_idx]
        comps_testing  = comps_all[test_idx]

        # Return basic description as validation
        for split_name, split_df in [("Training set", DATA_train), ("Validation set", DATA_test)]:
            print(f"INFO:   The {split_name} (N={len(split_df)}) has a censoring rate of {(split_df['status'] == 0).values.astype(float).mean().round(4) * 100:.2f}%")
            print(f"INFO:   Cohort proportions (%):", [round(float(pp) * 100,2) for pp in split_df["cohort"].value_counts(normalize=True).sort_index().values])

        # Account for age and sex
        comps_training = np.hstack([
            comps_training, 
            DATA_train[["age", "sex"]].values
        ])
        comps_testing = np.hstack([
            comps_testing, 
            DATA_test[["age", "sex"]].values
        ])

        # Checking orthogonality and colinearity --> LCs should pass this check
        comps_training_norm = np.linalg.norm(comps_training[:, :latent_dimensions], axis=0, keepdims=True) ** 2
        comps_training_prod = (comps_training[:, :latent_dimensions].T @ comps_training[:, :latent_dimensions]) / comps_training_norm
        if not np.allclose(comps_training_prod, np.eye(comps_training_prod.shape[0]), atol=1e-6):
            print("        ++++++++++++++++++++++++++++++++++++\n        WARNING!\n        The LC scores in the training set are not orthogonal!\n        Computations will proceed but results may be unreliable; consider L2 regularization.\n        ++++++++++++++++++++++++++++++++++++")
        comps_training_rho = np.corrcoef(comps_training[:, :latent_dimensions+1].T)
        comps_training_rho_age = comps_training_rho[-1, :-1]
        comps_training_rho = comps_training_rho[:-2, :-2]
        if not np.allclose(comps_training_rho, np.eye(comps_training_rho.shape[0]), atol=1e-6):
            print("        ++++++++++++++++++++++++++++++++++++\n        WARNING!\n        The LC scores in the training set might be colinear!\n        Computations will proceed but results may be unstable; consider L2 regularization.\n        ++++++++++++++++++++++++++++++++++++")

        # Check for collinearity between LCs and age/sex
        print(f"INFO:   The maximum/minimum Pearson correlation between LCs and age in the training set is {np.max(comps_training_rho_age):.4f}/{np.min(comps_training_rho_age):.4f}")
        comps_training_srho_sex = np.zeros((latent_dimensions,))
        for i in range(latent_dimensions):
            comps_training_srho_sex[i], _ = spearmanr(comps_training[:, i], comps_training[:, -1])
        print(f"INFO:   The maximum/minimum Spearman correlation between LCs and sex in the training set is {np.max(comps_training_srho_sex):.4f}/{np.min(comps_training_srho_sex):.4f}")

        if plot_folds:
            corr_labels = [f"z{i+1}" for i in range(latent_dimensions)] + ["age", "sex"]
            fig, ax = plot_correlation_matrix(comps_training, corr_labels)
            fig.savefig(os.path.join(save_dir, "Correlation-Matrix", f"Fold-{fold_idx+1}.{fmt}"),
                        dpi=dpi, format=fmt)
            plt.close(fig)

        # Further standardization of the principal components scores
        zscorer_ = StandardScaler()
        comps_training = np.hstack([
            zscorer_.fit_transform(comps_training[:, :latent_dimensions+1]),             # Standardize LCs and age
            comps_training[:, -1].reshape(-1, 1)                                    # Standardize sex
        ])
        comps_testing = np.hstack([
            zscorer_.transform(comps_testing[:, :latent_dimensions+1]),                  # Standardize LCs and age using the training distribution
            comps_testing[:, -1].reshape(-1, 1)                                     # Standardize sex using the training distribution
        ])

        # Transformation to structured array for sksurv
        DATA_train  = Surv.from_dataframe(event=event_col, time=duration_col, data=DATA_train)
        DATA_test  = Surv.from_dataframe(event=event_col, time=duration_col, data=DATA_test)

        # Baseline Cox model fit — age and sex only
        cox_baseline = CoxPHSurvivalAnalysis(alpha=0)
        cox_baseline.fit(comps_training[:, -2:], DATA_train)
        risk_baseline_train = cox_baseline.predict(comps_training[:, -2:])
        risk_baseline_test  = cox_baseline.predict(comps_testing[:, -2:])    
        cindex_baseline_train = concordance_index_censored(DATA_train[event_col].astype(bool), DATA_train[duration_col],risk_baseline_train)[0]              # Harrell C-index training
        cindex_baseline_test = concordance_index_censored(DATA_test[event_col].astype(bool), DATA_test[duration_col],risk_baseline_test)[0]                  # Harrell C-index testing
        cindex_ipcw_baseline_train = concordance_index_ipcw(survival_train=DATA_train, survival_test=DATA_train, estimate=risk_baseline_train, tau=tau)[0]   # Uno IPCW C-index training
        cindex_ipcw_baseline_test = concordance_index_ipcw(survival_train=DATA_train, survival_test=DATA_test, estimate=risk_baseline_test, tau=tau)[0]      # Uno IPCW C-index testing
        CVbaseline[fold_idx, :] = [cindex_baseline_train, cindex_baseline_test, cindex_ipcw_baseline_train, cindex_ipcw_baseline_test]
        
        # LASSO fit
        cox_lasso = CoxnetSurvivalAnalysis(
            l1_ratio=L1_ratio, 
            alphas=regularization_path,
            penalty_factor=np.concatenate([np.ones((latent_dimensions,)), np.array([0, 0])]) if not regularize_demographics else None 
        )
        cox_lasso.fit(comps_training, DATA_train)
        cox_lasso.alphas_ = np.round(cox_lasso.alphas_, reg_precision)              # To ensure correct indexing and avoid precision errors

        # Concordance indices
        cindex_train_path, cindex_test_path, best_alpha_idx = concordance_index_path(cox_lasso, comps_training, comps_testing, DATA_train, DATA_test, duration_col, event_col)
        cindex_ipcw_train_path, cindex_ipcw_test_path, _ = concordance_index_ipcw_path(cox_lasso, comps_training, comps_testing, DATA_train, DATA_test, tau=tau)
        
        if plot_folds:
            fig, ax = plot_concordance_index_path(cox_lasso, cindex_train_path, cindex_test_path)
            fig.savefig(os.path.join(save_dir, "Lasso-Regularization_Concordance-Indices", f"Fold-{fold_idx+1}.{fmt}"), dpi=dpi, format=fmt)
            plt.close(fig)

            fig, ax = plot_concordance_index_ipcw_path(cox_lasso, cindex_ipcw_train_path, cindex_ipcw_test_path)
            fig.savefig(os.path.join(save_dir, "Lasso-Regularization_IPCW-Indices", f"Fold-{fold_idx+1}.{fmt}"), dpi=dpi, format=fmt)
            plt.close(fig)
            
            coefficients_lasso = pd.DataFrame(cox_lasso.coef_, columns=cox_lasso.alphas_, index=[f"PC-{i+1}" for i in range(comps_training.shape[1]-2)]+["age", "sex"])
            fig, ax = plot_shrinkage_coefficients(coefficients_lasso, n_highlight=5, best_alpha=cox_lasso.alphas_[best_alpha_idx])
            fig.savefig(os.path.join(save_dir, "Lasso-Regularization_LogHazard-Ratios", f"Fold-{fold_idx+1}.{fmt}"), dpi=dpi, format=fmt)
            plt.close(fig)

            fig, ax = plot_deviance_ratio_path(cox_lasso, best_alpha=cox_lasso.alphas_[best_alpha_idx], max_DR=.2)
            fig.savefig(os.path.join(save_dir, "Lasso-Regularization_Deviance-Ratio", f"Fold-{fold_idx+1}.{fmt}"), dpi=dpi, format=fmt)
            plt.close(fig)

        CVresults[fold_idx] = np.array([cox_lasso.alphas_, cindex_train_path, cindex_test_path, cindex_ipcw_train_path, cindex_ipcw_test_path, cox_lasso.deviance_ratio_])

    Nfolds = fold_idx + 1
    np.save(os.path.join(save_dir, f"CV-results_Nfolds-{Nfolds}.npy"), CVresults, allow_pickle=True) 

    ############################################################################################################################################
    ### AVERAGING ACROSS FOLDS AND FEATURE SELECTION

    # This is coded this way  because the regularization path has been discretized in the same way across folds. However, in certain cases, 
    #      it could be that the path was not fit in its entirety, therefore, we are averaging taking into account this fact. In practice, 
    #      the path was likely fit without problems since it was specifically chosen.

    collected = {
        'cindex_train':      {alpha: [] for alpha in regularization_path},
        'cindex_test':       {alpha: [] for alpha in regularization_path},
        'cindex_ipcw_train': {alpha: [] for alpha in regularization_path},
        'cindex_ipcw_test':  {alpha: [] for alpha in regularization_path},
        'dr':                {alpha: [] for alpha in regularization_path},
    }

    for fold_idx in range(Nfolds):
        fold_alphas            = CVresults[fold_idx][0, :]
        fold_cindex_train      = CVresults[fold_idx][1, :]
        fold_cindex_test       = CVresults[fold_idx][2, :]
        fold_cindex_ipcw_train = CVresults[fold_idx][3, :]
        fold_cindex_ipcw_test  = CVresults[fold_idx][4, :]
        fold_dr                = CVresults[fold_idx][5, :]

        for alpha, c_tr, c_ts, c_ip_tr, c_ip_ts, dr in zip(
            fold_alphas,
            fold_cindex_train,
            fold_cindex_test,
            fold_cindex_ipcw_train,
            fold_cindex_ipcw_test,
            fold_dr
        ):
            if alpha in collected['cindex_train']:
                # Guard against NaN/Inf from unstable Cox fits at low alpha
                if np.isfinite(c_tr) and np.isfinite(c_ts) and \
                np.isfinite(c_ip_tr) and np.isfinite(c_ip_ts) and \
                np.isfinite(dr):
                    collected['cindex_train'][alpha].append(c_tr)
                    collected['cindex_test'][alpha].append(c_ts)
                    collected['cindex_ipcw_train'][alpha].append(c_ip_tr)
                    collected['cindex_ipcw_test'][alpha].append(c_ip_ts)
                    collected['dr'][alpha].append(dr)

    # Aggregate
    avg_cindex_train,      sem_cindex_train      = {}, {}
    avg_cindex_test,       sem_cindex_test       = {}, {}
    avg_cindex_ipcw_train, sem_cindex_ipcw_train = {}, {}
    avg_cindex_ipcw_test,  sem_cindex_ipcw_test  = {}, {}
    avg_dr,                sem_dr                = {}, {}
    k_per_alpha = {}       # track how many folds contributed — exposes the sparsity problem

    for alpha in regularization_path:
        for avg, sem, metric in [
            (avg_cindex_train,      sem_cindex_train,      'cindex_train'),
            (avg_cindex_test,       sem_cindex_test,       'cindex_test'),
            (avg_cindex_ipcw_train, sem_cindex_ipcw_train, 'cindex_ipcw_train'),
            (avg_cindex_ipcw_test,  sem_cindex_ipcw_test,  'cindex_ipcw_test'),
            (avg_dr,                sem_dr,                'dr'),
        ]:
            vals = np.array(collected[metric][alpha])
            k    = len(vals)
            avg[alpha] = np.mean(vals) if k > 0 else np.nan
            sem[alpha] = np.std(vals, ddof=1) / np.sqrt(k) if k > 1 else np.nan
        
        k_per_alpha[alpha] = len(collected['cindex_test'][alpha])

    # ── Best alpha selection ───────────────────────────────────────────────────────
    # Only consider alphas where ALL folds contributed — partial averages are unreliable
    valid_alphas = [a for a in regularization_path if k_per_alpha[a] == Nfolds]

    if len(valid_alphas) == 0:
        print("        ++++++++++++++++++++++++++++++++++++\n        WARNING!\n        No regularization path has full fold coverage — relaxing to k >= Nfolds//2")
        valid_alphas = [a for a in regularization_path if k_per_alpha[a] >= Nfolds // 2]

    # Best by Harrell C-index
    best_alpha_cindex = max(valid_alphas, key=lambda a: avg_cindex_test[a])
    best_cindex_val   = avg_cindex_test[best_alpha_cindex]

    # 1-SE rule: sparsest model within 1 SE of best
    one_se_threshold_cindex = best_cindex_val - sem_cindex_test[best_alpha_cindex]
    # alphas are sorted descending (large = sparse), so take the first (largest) alpha
    # whose mean C-index is still above the threshold
    best_alpha_cindex_1se = next(
        a for a in regularization_path
        if a in valid_alphas and avg_cindex_test[a] >= one_se_threshold_cindex
    )

    # Best by Uno's IPCW C-index
    best_alpha_ipcw    = max(valid_alphas, key=lambda a: avg_cindex_ipcw_test[a])
    best_ipcw_val      = avg_cindex_ipcw_test[best_alpha_ipcw]

    one_se_threshold_ipcw = best_ipcw_val - sem_cindex_ipcw_test[best_alpha_ipcw]
    best_alpha_ipcw_1se   = next(
        a for a in regularization_path
        if a in valid_alphas and avg_cindex_ipcw_test[a] >= one_se_threshold_ipcw
    )

    print(f"\nINFO:   Fold coverage per alpha (should be {Nfolds} everywhere)")
    low_coverage = [(a, k_per_alpha[a]) for a in regularization_path if k_per_alpha[a] < Nfolds]
    if low_coverage:
        print(f"        ++++++++++++++++++++++++++++++++++++\n        WARNING!\n        {len(low_coverage)} regularization paths have incomplete coverage:")
        for a, k in low_coverage:
            print(f"        ++++++++++++++++++++++++++++++++++++\n        WARNING!\n        lambda={a:.{reg_precision}f} → {k}/{Nfolds} folds")
    else:
        print("INFO:     All alphas have full fold coverage.")

    print(f"\nINFO:   Best alpha:              {best_alpha_cindex:.{reg_precision}f}  → Testing Harrell's C-index (± SE) of {best_cindex_val:.{reg_precision}f} ± {sem_cindex_test[best_alpha_cindex]:.{reg_precision}f}")
    print(f"INFO:   Best alpha (1-SE rule):  {best_alpha_cindex_1se:.{reg_precision}f}")
    print(f"INFO:   Training Harrell's C-index (± SE) of {avg_cindex_train[best_alpha_cindex]:.{reg_precision}f} ± {sem_cindex_train[best_alpha_cindex]:.{reg_precision}f}")
    print(f"INFO:   Best alpha:              {best_alpha_ipcw:.{reg_precision}f}  → Testing Uno's IPCW C-index (± SE) of {best_ipcw_val:.{reg_precision}f} ± {sem_cindex_ipcw_test[best_alpha_ipcw]:.{reg_precision}f}")
    print(f"INFO:   Best alpha (1-SE rule):  {best_alpha_ipcw_1se:.{reg_precision}f}")
    print(f"INFO:   Training IPCW C-index (± SE) of {avg_cindex_ipcw_train[best_alpha_ipcw]:.{reg_precision}f} ± {sem_cindex_ipcw_train[best_alpha_ipcw]:.{reg_precision}f}")
    print(f"\nINFO:   Baseline training C-index (± SE):               {CVbaseline[:,0].mean():.{reg_precision}f} (± {CVbaseline[:,0].std()/np.sqrt(Nfolds):.{reg_precision}f})")
    print(f"INFO:   Baseline testing C-index (± SE):                {CVbaseline[:,1].mean():.{reg_precision}f} (± {CVbaseline[:,1].std()/np.sqrt(Nfolds):.{reg_precision}f})")
    print(f"INFO:   Baseline training IPCW C-index (± SE):          {CVbaseline[:,2].mean():.{reg_precision}f} (± {CVbaseline[:,2].std()/np.sqrt(Nfolds):.{reg_precision}f})")
    print(f"INFO:   Baseline testing IPCW C-index (± SE):           {CVbaseline[:,3].mean():.{reg_precision}f} (± {CVbaseline[:,3].std()/np.sqrt(Nfolds):.{reg_precision}f})")

    def to_array(d, mask_incomplete=True):
        arr = np.array([d[a] for a in regularization_path], dtype=float)
        if mask_incomplete:
            for i, a in enumerate(regularization_path):
                if k_per_alpha[a] < Nfolds:
                    arr[i] = np.nan
        return arr

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    cl_train, cl_test = "black", "royalblue"

    # 1. C-Index
    ax = axes[0]
    y_tr, e_tr = to_array(avg_cindex_train), to_array(sem_cindex_train)
    y_ts, e_ts = to_array(avg_cindex_test), to_array(sem_cindex_test)
    ax.plot(regularization_path, y_tr, ".--", label="Training folds", alpha=1, linewidth=.75, markersize=3, color=cl_train)
    ax.fill_between(regularization_path, y_tr - e_tr, y_tr + e_tr, color=cl_train, alpha=0.2)
    ax.plot(regularization_path, y_ts, ".-", label="Testing folds", alpha=0.5, linewidth=.75, markersize=3, color=cl_test)
    ax.fill_between(regularization_path, y_ts - e_ts, y_ts + e_ts, color=cl_test, alpha=0.2)
    ax.axhline(best_cindex_val, color="red", linestyle="-", linewidth=0.5, label=f"Best C-index at test: {best_cindex_val:.{reg_precision}f}")
    ax.axvline(best_alpha_cindex, color="red", linestyle="--", label="Best "+r"$\lambda$"+f" at test: {best_alpha_cindex:.{reg_precision}f}")
    baseline_mean_test = CVbaseline[:, 1].mean()
    baseline_sem_test  = CVbaseline[:, 1].std() / np.sqrt(Nfolds)
    ax.axhline(baseline_mean_test, color="gray", linestyle=":", linewidth=1,
               label=f"Baseline (age+sex) C-index: {baseline_mean_test:.{reg_precision}f}")
    ax.axhspan(baseline_mean_test - baseline_sem_test, baseline_mean_test + baseline_sem_test,
               color="gray", alpha=0.15)
    ax.set_ylabel("C-index", fontsize=12)

    # 2. IPCW C-Index
    ax = axes[1]
    y_ip_tr, e_ip_tr = to_array(avg_cindex_ipcw_train), to_array(sem_cindex_ipcw_train)
    y_ip_ts, e_ip_ts = to_array(avg_cindex_ipcw_test), to_array(sem_cindex_ipcw_test)
    ax.plot(regularization_path, y_ip_tr, ".--", label="Training folds", alpha=1, linewidth=.75, markersize=3, color=cl_train)
    ax.fill_between(regularization_path, y_ip_tr - e_ip_tr, y_ip_tr + e_ip_tr, color=cl_train, alpha=0.2)
    ax.plot(regularization_path, y_ip_ts, ".-", label="Testing folds", alpha=0.5, linewidth=.75, markersize=3, color=cl_test)
    ax.fill_between(regularization_path, y_ip_ts - e_ip_ts, y_ip_ts + e_ip_ts, color=cl_test, alpha=0.2)
    ax.axhline(best_ipcw_val, color="tab:green", linestyle="-", linewidth=0.5, label=f"Best IPCW C-index at test: {best_ipcw_val:.{reg_precision}f}")
    ax.axvline(best_alpha_ipcw, color="tab:green", linestyle="--", label="Best "+r"$\lambda$"+f" at test: {best_alpha_ipcw:.{reg_precision}f}")
    baseline_mean_ipcw_test = CVbaseline[:, 3].mean()
    baseline_sem_ipcw_test  = CVbaseline[:, 3].std() / np.sqrt(Nfolds)
    ax.axhline(baseline_mean_ipcw_test, color="gray", linestyle=":", linewidth=1,
               label=f"Baseline (age+sex) IPCW C-index: {baseline_mean_ipcw_test:.{reg_precision}f}")
    ax.axhspan(baseline_mean_ipcw_test - baseline_sem_ipcw_test, baseline_mean_ipcw_test + baseline_sem_ipcw_test,
               color="gray", alpha=0.15)
    ax.set_ylabel("IPCW C-index", fontsize=12)

    # 3. Deviance Ratio
    ax = axes[2]
    y_dr, e_dr = to_array(avg_dr), to_array(sem_dr)
    ax.plot(regularization_path, y_dr, ".--", alpha=1, linewidth=.75, markersize=3, color=cl_train)
    ax.fill_between(regularization_path, y_dr - e_dr, y_dr + e_dr, color=cl_train, alpha=0.2)
    ax.set_ylabel("Deviance ratio: "+r'$DR = 1 - \frac{deviance}{null_deviance}$', fontsize=12)

    # Styling loop
    for i, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_xlabel("Regularization ("+r"$\lambda$"+")", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        if i<2:
            ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"CV-results_Nfolds-{Nfolds}.{fmt}"), dpi=dpi, format=fmt)
    plt.close(fig)