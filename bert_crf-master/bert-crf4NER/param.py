class PARAM(object):	
	apr_dir = '../model/'
	data_dir = '../corpus/'
	model_name = 'model_4.pt'
	epoch = 5
	bert_model = 'bert-base-cased'
	lr = 5e-5
	eps = 1e-8
	batch_size = 16
	training_data = 'ind_train.txt'
	val_data = 'ind_test.txt'
	test_data = 'ind_test.txt'
	test_out = 'test_prediction.csv'
	raw_prediction_output = 'raw_prediction.csv'
