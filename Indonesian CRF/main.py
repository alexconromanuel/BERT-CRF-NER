# standard library
import sys
from datetime import datetime
import joblib

# 3rd party packages
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# custom modules
from util import *
from benchmarking import *

def main():
    path_train = "data/singgalang_train.tsv"
    path_test = "data/singgalang_test.tsv"
    pretrained_tagger = "/home/booster/CRF/pre-trained-model/all_indo_man_tag_corpus_model.crf.tagger"

    fr_train = FileReader(path_train)
    ext = fr_train.check_type()
    if ext == ".tsv":
        df_train = fr_train.read_tsv()
    else:
        df_train = pd.DataFrame()
        print(f"Currently, the given file cannot be processed.")
        sys.exit()
        
    fr_test = FileReader(path_test)
    ext = fr_test.check_type()
    if ext == ".tsv":
        df_test = fr_test.read_tsv()
    else:
        df_test = pd.DataFrame()
        print(f"Currently, the given file cannot be processed.")
        sys.exit()
    
    start_time = datetime.now()
    
    dp_train = DatasetPreparator(df_train, pretrained_tagger)
    df_train = dp_train.check_post()
    
    dp_test = DatasetPreparator(df_test, pretrained_tagger)
    df_test = dp_test.check_post()
    
    getter_train = SentenceGetter(df_train)
    sentences_train = getter_train.sentences
    
    getter_test = SentenceGetter(df_test)
    sentences_test = getter_test.sentences
    

    # feature extraction
    print(f"Feature extraction is in progress...")
    X_train = [sent2features(s) for s in sentences_train]
    y_train = [sent2labels(s) for s in sentences_train]
    
    X_test = [sent2features(s) for s in sentences_test]
    y_test = [sent2labels(s) for s in sentences_test]
     
    print(f"Feature extraction is complete.")
    
    classes = np.unique(df_train[["ne"]])
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
    
    filename = 'finalized_model.sav'
    joblib.dump(new_crf,filename)


    sorted_labels = sorted(new_classes,
                           key=lambda name: (name[1:], name[0]))
    y_pred = new_crf.predict(X_test)
    y_test_flat = [item for sublist in y_test for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]
    print(classification_report(y_test_flat, y_pred_flat, 
                                labels = sorted_labels, digits=3))
    end_time = datetime.now()
    print("--- %s seconds ---" % (end_time - start_time))
    print(f"\n")

if __name__ == "__main__":
    main()