import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

# 3rd party packages
import scipy.stats
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

# custom modules
from util import *



def exp(df_data, pretrained_tagger):
    """Without preprocessing"""
    dp = DatasetPreparator(df_data, pretrained_tagger)
    df = dp.check_post()
    
    
    
    getter = SentenceGetter(df)
    sentences = getter.sentences

    # feature extraction
    print(f"Feature extraction is in progress...")
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    print(f"Feature extraction is complete.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    classes = np.unique(df[["ne"]])
    classes = classes.tolist()
    classes.remove("O")
    new_classes = classes.copy()

    # training
    crf = CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False
    )
    params_space = dict(c1=scipy.stats.expon(scale=0.5),
                        c2=scipy.stats.expon(scale=0.05))
    f1_scorer = make_scorer(metrics.flat_f1_score,
                        average="macro", labels=new_classes)

    rs = RandomizedSearchCV(crf, params_space,
                            cv=5,
                            n_iter=10,
                            scoring=f1_scorer,
                            random_state=0)
    rs.fit(X_train, y_train)
    
    new_crf = rs.best_estimator_

    sorted_labels = sorted(new_classes,
                           key=lambda name: (name[1:], name[0]))
    y_pred = new_crf.predict(X_test)
    y_test_flat = [item for sublist in y_test for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]
    print(classification_report(y_test_flat, y_pred_flat, 
                                labels = sorted_labels, digits=3))