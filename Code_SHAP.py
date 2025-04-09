
# Code by Lennart Purucker for the feature selection
def prune_features_binary_classification(X: "pd.DataFrame", y: "pd.DataFrame", *, time_limit_per_split: int = 3600, cv = "default", eval_metric="accuracy"):
    """Obtain the optimal set of features for a given dataset by iterative (clever)
    feature pruning with AutoGluon and TabPFN.

    This will try to find the optimal set of feature to improve predictive performance
    using the cv strategy provided.

    Requirements:
        - autogluon
        - tabpfn

    Args:
        X: The input features.
        y: The target variable.
        time_limit_per_split: The time limit in seconds.
            This much time is used at most per split of the CV.
        cv: The splitter to use for the cross-validation.
            If "default", RepeatedStratifiedKFold with 5 splits and 10 repeats is used.
        eval_metric: The evaluation metric to use for the model.
            Change this to a supported sklearn/autogluon metric to optimize w.r.t.
            your metric of interest.
    """
    import time
    import warnings
    import json

    import pandas as pd


    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    from autogluon.core.utils.feature_selection import FeatureSelector, logger
    from autogluon.core.models import AbstractModel
    from autogluon.common.utils.log_utils import set_logger_verbosity

    set_logger_verbosity(verbosity=4)

    warnings.simplefilter(action='ignore',category=FutureWarning)
    warnings.simplefilter(action='ignore',category=UserWarning)

    class FEModel(AbstractModel):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self._feature_generator=None

        def _fit(self,X: pd.DataFrame, y: pd.Series, **kwargs):

            # TabPFN version, very slow.
            # params=self._get_model_params()
            # from tabpfn import TabPFNClassifier
            # self.model=TabPFNClassifier(
            #     n_estimators=4, device="cuda",
            #     # fit_mode="fit_with_cache",
            #     **params)

            # SVM (because it is fast)
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            self.model=Pipeline(
                [("scaler",StandardScaler().set_output(transform="pandas")),
                    # Standardization step
                    ("classifier",SVC(random_state=42)),  # Classifier
                ],)

            X=self.preprocess(X,is_train=True)
            self.model.fit(X,y)

        # FIXME: uncomment this for TabPFN.
        def predict(self,X,**kwargs) -> np.ndarray:
            """
            Returns class predictions of X.
            For binary and multiclass problems, this returns the predicted class labels as a 1d numpy array.
            For regression problems, this returns the predicted values as a 1d numpy array.
            """
            return self.model.predict(X)

        def _set_default_params(self):
            # FIXME: change depending on model
            default_params={'ignore_pretraining_limits' :True,'n_jobs':8,'random_state':0,}
            for param,val in default_params.items():
                self._set_default_param_value(param,val)

    if cv == "default":
        from sklearn.model_selection import RepeatedStratifiedKFold

        splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    else:
        splitter = cv
    time_limit = time_limit_per_split

    optimal_features_per_split = []
    for train_index, test_index in splitter.split(X=X, y=y):
        # Get split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        feature_generator = AutoMLPipelineFeatureGenerator(verbosity=0)
        X_train_transformed = feature_generator.fit_transform(X=X_train, y=y_train)
        X_train = X_train[X_train_transformed.columns]
        logger.info(
            f"AutoGluon Useless Feature Pruning:\tPruned from {X_train.shape[1]} to {X_train_transformed.shape[1]} features.",
        )

        st_time = time.time()
        candidate_features = X_train.columns.to_list()
        for selection_step, selection_config in enumerate(
            [
                dict(
                    n_fi_subsample=10000,
                    prune_threshold="noise",
                    prune_ratio=0.075,
                    stopping_round=3,
                ),
                dict(
                    n_fi_subsample=100000,
                    prune_threshold="none",
                    prune_ratio=0.05,
                    stopping_round=20,
                ),
                dict(
                    n_fi_subsample=500000,
                    prune_threshold="none",
                    prune_ratio=0.025,
                    stopping_round=40,
                ),
            ],
        ):
            rest_time = time_limit - (time.time() - st_time)
            logger.info(f"AutoGluon Feature Pruning {selection_step} | Time Left: {rest_time:.2f} seconds.")
            fs = FeatureSelector(
                model=FEModel(eval_metric=eval_metric),
                time_limit=time_limit_per_split,
                problem_type="binary",
                seed=0,
                raise_exception=True,
            )
            candidate_features = fs.select_features(
                X=X_train,
                y=y_train,
                X_val=X_test,
                y_val=y_test,
                n_train_subsample=50000,
                min_improvement=0,
                **selection_config,
            )

            X_train = X_train[candidate_features]
            X_test = X_test[candidate_features]



        logger.info(f"AutoGluon Feature Pruning:\tPruned to {X_train.shape[1]} features.")
        logger.info(f"Final Features: {candidate_features}")
        optimal_features_per_split.append(candidate_features)
        with open("optimal_features_per_split_im.json","w") as file:
            json.dump(optimal_features_per_split,file)

    logger.info(f"Optimal Features Per Split: {optimal_features_per_split}")
    with open("optimal_features_per_split.json","w") as file:
        json.dump(optimal_features_per_split,file)
