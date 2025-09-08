from ampire.distributed import DistEnetConvexHull, DistThresholdFinder
from ampire.utils import DataReader
from dynaconf import Dynaconf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import logging
import os
from pathlib import Path
from collections import defaultdict
from ampire.utils.helpers import (
    write_report,
    find_almost_boundaries,
    make_inlier_outlier_label,
    preprocess,
    extract_directory_name_from_file_path,
    create_directory,
    train_test_split
)

# Setup logging
logging.basicConfig(
    filename='logs/dist_enet_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load configuration
try:
    cnf = Dynaconf(
        settings_files=["configs/model-config.yaml", "configs/dataset-config.yaml"]
    )
    logging.info("Loaded configuration successfully")
except Exception as e:
    logging.error(f"Failed to load configuration: {str(e)}")
    raise

# Parameters for grid search
params = {
    'kernels': cnf.get('kernels.all', ['rbf', 'linear']),
    'gamma': cnf.get('gamma', [0.1, 1.0, 10.0, 100.0]),
    'landa1': cnf.get('landa1', [0.1, 0.5, 0.9, 1.0]),
    'tau': [0.5, 0.7, 0.9],
    'alpha': [0.7, 0.9],
    'cv': cnf.get('cv', 5),
    'random_state': cnf.get('random_state', 42),
    'train_size': cnf.get('train_size', 0.8),
    'target': cnf.get('target', 1)
}

# Output directory
root = 'ThesisExperiments/results_dist_enet/'
errors = defaultdict(list)
report = dict()

# Graph for DistEnetConvexHull
graph = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])

# Total iterations for progress tracking
total = len(params['kernels']) * len(params['gamma']) * len(params['landa1']) * len(params['tau']) * len(params['alpha'])
print(f"Total datasets: {len(cnf.data.all)}, Total iterations per dataset: {total}")

# Subsampling for DistThresholdFinder
n_subsamples = 10

for data_path, inlier in (ddes := tqdm(cnf.data.all, leave=False, colour='green')):
    directory_name = extract_directory_name_from_file_path(data_path)
    ddes.set_description(f'\33[32mdata[{directory_name}]')
    report['dataset'] = directory_name

    the_best_f1 = -1
    the_best_acc = -1
    the_best_precision = -1
    the_best_recall = -1
    the_best_kappa = -1
    confusion = np.ones((2, 2))
    results = []
    sheet_number = 0
    counter = 0

    # Create directory
    try:
        create_directory(directory=directory_name, des=root)
        logging.info(f"Created directory: {os.path.join(root, directory_name)}")
    except Exception as e:
        logging.error(f"Failed to create directory {directory_name}: {str(e)}")
        errors[f"Directory creation error: {str(e)}"].append(directory_name)
        continue

    file_path = os.path.join(root, directory_name, f'{directory_name}.xlsx')

    # Ensure Excel file is created and writable
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(file_path):
            pd.DataFrame().to_excel(file_path, sheet_name='init', index=False, engine='openpyxl')
            logging.info(f"Created Excel file: {file_path}")
    except Exception as e:
        logging.error(f"Failed to create Excel file {file_path}: {str(e)}")
        errors[f"Excel creation error: {str(e)}"].append(directory_name)
        continue

    # Load data
    try:
        data = DataReader(data_path)(header=None)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        logging.info(f"Loaded data: {data_path}")
    except Exception as e:
        logging.error(f"Failed to load data {data_path}: {str(e)}")
        errors[f"Data loading error: {str(e)}"].append(directory_name)
        continue

    # Convert labels to +1/-1
    try:
        y = make_inlier_outlier_label(labels=y, inlier=inlier, target=params['target'])
        logging.info(f"Converted labels for {directory_name}")
    except Exception as e:
        logging.error(f"Failed to convert labels for {directory_name}: {str(e)}")
        errors[f"Label conversion error: {str(e)}"].append(directory_name)
        continue

    # Preprocess data
    try:
        X = preprocess(scaler=StandardScaler, data=X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, target=params['target'], random_state=params['random_state'], train_size=params['train_size']
        )
        logging.info(f"Preprocessed and split data for {directory_name}: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    except Exception as e:
        logging.error(f"Failed to preprocess/split data for {directory_name}: {str(e)}")
        errors[f"Preprocessing error: {str(e)}"].append(directory_name)
        continue

    # Subsample indices for DistThresholdFinder
    try:
        np.random.seed(params['random_state'])
        n_inlier_samples = X_train[y_train == params['target']].shape[0]
        sample_indices = np.random.choice(n_inlier_samples, size=min(n_subsamples, n_inlier_samples), replace=False)
        logging.info(f"Subsampled {len(sample_indices)} indices for {directory_name} (inliers: {n_inlier_samples}): {sample_indices}")
    except Exception as e:
        logging.error(f"Failed to subsample indices for {directory_name}: {str(e)}")
        errors[f"Subsampling error: {str(e)}"].append(directory_name)
        continue

    for metric in (des := tqdm(params['kernels'], leave=False, colour='red')):
        des.set_description(f'\33[31mmetric[{metric}]')
        report['metric'] = metric

        for gamma in (gam := tqdm(params['gamma'], leave=False, colour='blue')):
            gam.set_description(f'\33[34mgamma[{gamma}]')
            report['gamma'] = gamma

            for landa in (lan := tqdm(params['landa1'], leave=False, colour='yellow')):
                lan.set_description(f'\33[33mlanda[{landa}]')
                report['landa'] = landa

                for tau in (tau_loop := tqdm(params['tau'], leave=False, colour='green')):
                    tau_loop.set_description(f'\33[32mtau[{tau}]')
                    report['tau'] = tau

                    for alpha in (alpha_loop := tqdm(params['alpha'], leave=False, colour='cyan')):
                        alpha_loop.set_description(f'\33[36malpha[{alpha}]')
                        report['alpha'] = alpha

                        try:
                            # Set kernel_params based on metric
                            kernel_params = {'gamma': gamma} if metric == 'rbf' else {}

                            # Initialize DistEnetConvexHull
                            model = DistEnetConvexHull(
                                landa1=landa,
                                alpha=alpha,
                                tau=tau,
                                node_max_iter=20,
                                num_triggers=20,
                                graph=graph,
                                metric=metric,
                                kernel_params=kernel_params,
                                random_state=params['random_state'],
                                verbose=False
                            )

                            # Debug classifier status
                            logging.info(f"Model estimator type for {directory_name} (metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha}): {getattr(model, '_estimator_type', 'Unknown')}")

                            # Fit model
                            try:
                                model.fit(X_train, y_train)
                                logging.info(f"Fitted model for {directory_name}: metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha}")
                            except Exception as e:
                                logging.error(f"Model fitting failed for {directory_name} (metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha}): {str(e)}")
                                errors[f"Model fitting error: {str(e)}"].append(f"{directory_name}:{metric}-{gamma}-{landa}-{tau}-{alpha}")
                                continue

                            # Compute thresholds using DistThresholdFinder
                            try:
                                tf = DistThresholdFinder(model, model.X_target)
                                z = np.zeros((model.X_target.shape[0], 1))
                                logging.info(f"Initialized z with shape={z.shape} for threshold computation in {directory_name}")
                                for idx in sample_indices:
                                    z_value, _, _ = tf.find(outs='max', specific_index=idx)
                                    logging.info(f"Computed z[{idx}]={z_value}, shape={np.array(z_value).shape}")
                                    z[idx, 0] = z_value
                                logging.info(f"All z values for {directory_name}: {z.flatten()}")
                                _, _, thrs = tf.find(outs='max')
                                # Scale thresholds to avoid numerical underflow
                                if np.max(thrs) < 1e-10:
                                    thrs = thrs * 1e10
                                    logging.info(f"Scaled thresholds by 1e10 for {directory_name}: {thrs}")
                                logging.info(f"Computed thresholds for {directory_name}: {thrs}")
                                counter += 1
                                ddes.set_description(f'\33[32mdata[{directory_name}]-[{counter}/{total * len(thrs)}]-[{(counter/(total * len(thrs)))*100:.3f}%]')
                            except Exception as e:
                                logging.error(f"Threshold computation failed for {directory_name} (metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha}): {str(e)}")
                                errors[f"Threshold computation error: {str(e)}"].append(f"{directory_name}:{metric}-{gamma}-{landa}-{tau}-{alpha}")
                                thrs = [1e-5]
                                logging.info(f"Using default threshold [1e-5] for {directory_name} (metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha})")

                            # Find boundary samples
                            try:
                                mask_boundary = find_almost_boundaries(z=z, quantile=0.75)
                                X_train_mask, y_train_mask = X_train[mask_boundary, :], y_train[mask_boundary]
                                logging.info(f"Computed boundary samples for {directory_name}: {mask_boundary.shape[0]} samples")
                            except Exception as e:
                                logging.error(f"Boundary sample computation failed for {directory_name} (metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha}): {str(e)}")
                                errors[f"Boundary computation error: {str(e)}"].append(f"{directory_name}:{metric}-{gamma}-{landa}-{tau}-{alpha}")
                                X_train_mask, y_train_mask = X_train, y_train  # Fallback to full training set

                            for thr in (th := tqdm(thrs, leave=False, colour='magenta')):
                                th.set_description(f'\33[91mthr[{thr:.3f}]')
                                report['thr'] = thr

                                # Re-initialize model with threshold
                                model = DistEnetConvexHull(
                                    landa1=landa,
                                    alpha=alpha,
                                    tau=tau,
                                    node_max_iter=20,
                                    num_triggers=20,
                                    graph=graph,
                                    metric=metric,
                                    kernel_params=kernel_params,
                                    random_state=params['random_state'],
                                    verbose=False,
                                    thr=thr
                                )

                                # Debug classifier status
                                logging.info(f"Model estimator type for {directory_name} (metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha}, thr={thr:.3f}): {getattr(model, '_estimator_type', 'Unknown')}")

                                # Cross-validation
                                try:
                                    scores = cross_val_score(
                                        model, X_train_mask, y_train_mask, cv=params['cv'], scoring='f1'
                                    )
                                    report['scores'] = scores.tolist()
                                    report['best_score'] = np.max(scores) if len(scores) > 0 else np.nan
                                    report['avg_score'] = np.mean(scores) if len(scores) > 0 else np.nan
                                    logging.info(f"Computed cross-validation scores for {directory_name}: thr={thr:.3f}, scores={scores.tolist()}")
                                except Exception as e:
                                    logging.error(f"Cross-validation failed for {directory_name} (metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha}, thr={thr:.3f}): {str(e)}")
                                    errors[f"Cross-validation error: {str(e)}"].append(f"{directory_name}:{metric}-{gamma}-{landa}-{tau}-{alpha}-{thr:.3f}")
                                    report['scores'] = [np.nan] * params['cv']
                                    report['best_score'] = np.nan
                                    report['avg_score'] = np.nan

                                # Fit and predict
                                try:
                                    model.fit(X_train_mask, y_train_mask)
                                    y_pred = model.predict(X_test)
                                    logging.info(f"Predictions for {directory_name} (metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha}, thr={thr:.3f}): unique values={np.unique(y_pred)}")
                                    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
                                    f1 = f1_score(y_true=y_test, y_pred=y_pred)
                                    precision = precision_score(y_true=y_test, y_pred=y_pred)
                                    recall = recall_score(y_true=y_test, y_pred=y_pred)
                                    kappa = cohen_kappa_score(y1=y_test, y2=y_pred)
                                    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
                                    confusion = np.concatenate((confusion, conf_matrix), axis=1)
                                    try:
                                        write_report(obj=confusion, path=f"{root}/{directory_name}/{directory_name}_confusion_matrix.pkl")
                                        logging.info(f"Saved confusion matrix for {directory_name}")
                                    except Exception as e:
                                        logging.error(f"Failed to save confusion matrix for {directory_name}: {str(e)}")
                                    report['confusion_matrix'] = conf_matrix.tolist()
                                except Exception as e:
                                    logging.error(f"Prediction failed for {directory_name} (metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha}, thr={thr:.3f}): {str(e)}")
                                    errors[f"Prediction error: {str(e)}"].append(f"{directory_name}:{metric}-{gamma}-{landa}-{tau}-{alpha}-{thr:.3f}")
                                    acc = f1 = precision = recall = kappa = np.nan
                                    conf_matrix = np.zeros((2, 2))
                                    report['confusion_matrix'] = conf_matrix.tolist()

                                # Store metrics
                                report['X_test-acc'] = acc
                                report['X_test-f1'] = f1
                                report['X_test-precision'] = precision
                                report['X_test-recall'] = recall
                                report['X_test-kappa'] = kappa

                                # Debug report contents
                                logging.info(f"Report contents for {directory_name} (metric={metric}, gamma={gamma}, landa={landa}, tau={tau}, alpha={alpha}, thr={thr:.3f}): {report}")

                                # Save to Excel
                                try:
                                    df = pd.DataFrame([report])
                                    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as wr:
                                        df.to_excel(wr, sheet_name=f'{metric}-{sheet_number}-({f1:.4f})', index=False)
                                    logging.info(f"Wrote data to Excel sheet: {metric}-{sheet_number}-({f1:.4f}) in {file_path}")
                                except Exception as e:
                                    logging.error(f"Failed to write to Excel {file_path}: {str(e)}")
                                    errors[f"Excel write error: {str(e)}"].append(f"{directory_name}:{metric}-{sheet_number}")

                                # Save to CSV
                                try:
                                    csv_path = os.path.join(root, directory_name, f'{metric}-{sheet_number}-({f1:.4f}).csv')
                                    df.to_csv(csv_path, index=False)
                                    logging.info(f"Wrote data to CSV: {csv_path}")
                                except Exception as e:
                                    logging.error(f"Failed to write to CSV {csv_path}: {str(e)}")
                                    errors[f"CSV write error: {str(e)}"].append(f"{directory_name}:{metric}-{sheet_number}")

                                results.append(report.copy())
                                sheet_number += 1

                                # Update best metrics
                                try:
                                    if f1 > the_best_f1:
                                        the_best_f1 = f1
                                        write_report(obj=report, path=f'{root}/{directory_name}/best_f1.pkl')
                                    if acc > the_best_acc:
                                        the_best_acc = acc
                                        write_report(obj=report, path=f'{root}/{directory_name}/best_acc.pkl')
                                    if precision > the_best_precision:
                                        the_best_precision = precision
                                        write_report(obj=report, path=f'{root}/{directory_name}/best_precision.pkl')
                                    if recall > the_best_recall:
                                        the_best_recall = recall
                                        write_report(obj=report, path=f'{root}/{directory_name}/best_recall.pkl')
                                    if kappa > the_best_kappa:
                                        the_best_kappa = kappa
                                        write_report(obj=report, path=f'{root}/{directory_name}/best_kappa.pkl')
                                except Exception as e:
                                    logging.error(f"Failed to update best metrics for {directory_name}: {str(e)}")
                                    errors[f"Best metrics update error: {str(e)}"].append(f"{directory_name}:{metric}-{gamma}-{landa}-{tau}-{alpha}-{thr:.3f}")

                        except Exception as e:
                            counter += len(thrs)
                            id = f'{directory_name}:({gamma}-{landa}-{tau}-{alpha})'
                            errors[str(e)].append(id)
                            logging.error(f"Error in {id}: {str(e)}")

    # Save all results to pickle
    try:
        write_report(obj=results, path=f'{root}/{directory_name}/{directory_name}_results.pkl')
        logging.info(f"Saved all results to pickle: {root}/{directory_name}/{directory_name}_results.pkl")
    except Exception as e:
        logging.error(f"Failed to save results pickle for {directory_name}: {str(e)}")
        errors[f"Results pickle error: {str(e)}"].append(directory_name)

# Save errors
try:
    write_report(obj=errors, path=f'{root}/errors_dist_enet.pkl')
    logging.info(f"Saved errors to {root}/errors_dist_enet.pkl")
except Exception as e:
    logging.error(f"Failed to save errors: {str(e)}")
print("Errors:", errors)