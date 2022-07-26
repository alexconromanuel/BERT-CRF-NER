# standard library
import sys
from datetime import datetime
import joblib
import pandas as pd  
# 3rd party packages
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# custom modules
from util import *
from benchmarking import *

def main():
    path_test = "inference_data.tsv"
    pretrained_tagger = "/home/booster/CRF/pre-trained-model/all_indo_man_tag_corpus_model.crf.tagger"

    fr_test = FileReader(path_test)
    ext = fr_test.check_type()
    if ext == ".tsv":
        df_test = fr_test.read_tsv()
    else:
        df_test = pd.DataFrame()
        print(f"Currently, the given file cannot be processed.")
        sys.exit()
    
    start_time = datetime.now()
    
    dp_test = DatasetPreparator(df_test, pretrained_tagger)
    df_test = dp_test.check_post()
    
    getter_test = SentenceGetter(df_test)
    sentences_test = getter_test.sentences

    # feature extraction
    print(f"Feature extraction is in progress...")
    X_test = [sent2features(s) for s in sentences_test]
    y_test = [sent2labels(s) for s in sentences_test]
     
    print(f"Feature extraction is complete.")
    
    new_classes = ['PERSON', 'PLACE', 'ORGANISATION']

    #Load Model
    model = joblib.load('C:\Users\User\Desktop\TA-Elsaday 12S18060\finalized_model.sav')

    sorted_labels = sorted(new_classes,
                           key=lambda name: (name[1:], name[0]))
    y_pred = model.predict(X_test)
    y_test_flat = [item for sublist in y_test for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]
    print(classification_report(y_test_flat, y_pred_flat, 
                                labels = sorted_labels, digits=3))
        
    # dictionary of lists  
    dict = {'token': X_test, 'entity': y_pred}  
    df = pd.DataFrame(dict) 
        
    # saving the dataframe 
    df.to_csv('inf.tsv') 

    end_time = datetime.now()
    print("--- %s seconds ---" % (end_time - start_time))
    print(f"\n")

if __name__ == "__main__":
    main()