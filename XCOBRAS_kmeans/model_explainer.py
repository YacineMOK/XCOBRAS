from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import shap

# import 2e modèle
# .....
    

class XCobrasExplainer():
    """
    décrire son intêret... 

    Attributes:
        - self.model (str): which classifier are we going to use
            by default"rbf_svm" 
        - self.clf (sklearn Pipeline): the actual pipeline of this classifier.
            Using "Pipeline()" makes it clearer and easier to fit, predict and manipulate.
        - self.grid_search_cv (dict): The parameters that are going to be fine-tuned
                                      wrt the chosen classifier
        - self.test_size (je l'ai mis ici, comme ça ne touche pas directement 'COBRAS')

    ...    
    """
    def __init__(self, model="rbf_svm", test_size=0.4, verbose=True) -> None:
        """Init function

        Args:
            model (str, optional): Which classifier are we going to use. Defaults to "rbf_svm".
            test_size (float, optional): Proportion of the test dataset. Defaults to (0.4).
        """
        self.model = model
        self.test_size = test_size
        self.param_grid = None
        self.clf = None
        self.grid_search_cv = None
        self.verbose = verbose
        self.explainer = None
        self.shap_values = None


        # ----- Model selection
        if self.model == "rbf_svm":
            # RBF Model
            self.clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
            ])
            # Parameters  of the grid search
            gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
            Cs = [1, 10, 100, 1e3, 1e4, 1e5]
            self.param_grid = {
                "svm_clf__gamma": gammas, 
                "svm_clf__C": Cs
                }
        else:
            ... # un autre modèle avec d'autres hyperparam à fine-tune 

        self.grid_search_cv = GridSearchCV(
            estimator=self.clf, 
            param_grid=self.param_grid, 
            # factor=2, # only half of the candidates are selected
            cv=2 # default value
            )

    def fit(self, X, y):
        """Function that fits the classification model in  `self.clf`.
                  i. splits the data into train-test set
                 ii. gridsearchCV on the train set
                iii. (optional) test on the test set to prevent overfitting
        Args:
            X (np.array/pd.DataFrame): Dataset
            y (np.array/pd.DataFrame): Labels
        """
        # ----- dataset split (X and y)
        # `y_hat` because it is the current "partitionning" of COBRAS.
        # These are not ground truth label of the dataset, but cluster assigniation of COBRAS algorithm

        self.X_train, self.X_test, self.y_hat_train, self.y_hat_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # ----- Cross-Validation on the TRAIN set
        # Fitting this model
        # GridSearchCV
        # TODO GERER LES NUMPY ARRAY ET LES DATAFRAME 
        # TODO POUR LE MOMENT QUEDES NUMPY ARRAY
        self.grid_search_cv.fit(self.X_train,self.y_hat_train)
        self.best_model = self.grid_search_cv.best_estimator_

        # ----- Showing some results on the test set
        if self.verbose:
            y_test_pred = self.predict(self.X_test)
            print("---------Some scores:---------")
            print("------------------------------")
            print(f"f1-score (macro): {f1_score(self.y_hat_test, y_test_pred, average='macro'):.10f}")
            print(f"         (micro): {f1_score(self.y_hat_test, y_test_pred, average='micro'):.10f}")
            # print(f"  accuracy_score: {accuracy_score(self.y_hat_test, y_test_pred):.10f}")
            print("------------------------------")
            print("")
                
    def predict(self, X):
        """Prediction function

        Args:
            X (np.array/pd.DataFrame): Dataset we want to predict

        Returns:
            np.array: that represents the list of predictions (labels)
        """
        return self.best_model.predict(X)

    def explain(self, X, feature_names=None):
        # TODO warning / try / catch: self.best_model
        
        # if feature_names == None:
        #     feature_names = ["A: "+str(i)for i in range(X.shape[1])]

        self.explainer = shap.Explainer(
            self.best_model.predict,
            self.X_train,
            feature_names=feature_names
        )

        self.shap_values = self.explainer(X)
        return self.shap_values
    

    def fit_explain(self, X, y, ids, feature_names=None):
        self.fit(X,y)
        return self.explain(X[ids])