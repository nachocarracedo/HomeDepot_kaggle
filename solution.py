#python 3

#imports
from ast import literal_eval
import warnings

#pandas, numpy, visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline
#NLP
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
#ML
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn import linear_model
from sklearn import cross_validation

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn import pipeline, grid_search
from sklearn.pipeline import FeatureUnion

from sklearn.ensemble import GradientBoostingRegressor #For Classification
from sklearn.ensemble import AdaBoostRegressor

import time

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id', 'product_title', 'product_uid', 'search_term','product_description', 'attributes', 'brand', 'material','attributes_tokens',
                      'search_term_tokens', 'product_title_tokens','product_description_tokens', 'brand_tokens', 'material_tokens', 'search_units',
                      'product_title_tokens_uu','product_description_tokens_uu', 'attributes_tokens_uu','search_term_tokens_uu', 'search_no_units', 
                      'title_no_units','descr_no_units', 'attr_no_units','len_search_no_units']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


if __name__ == "__main__":

	start_time = time.time()
	
	print ("----- loading files -----")
	print (round(((time.time() - start_time)/60),2))
	#load files
	train_df = pd.read_csv("C:\\\\Users\\carrai1\\Desktop\\Projects\\HomeDepot\\datasets\\train.csv",encoding="ISO-8859-1")
	test_df = pd.read_csv("C:\\\\Users\\carrai1\\Desktop\\Projects\\HomeDepot\\datasets\\test.csv",encoding="ISO-8859-1")
	products_df = pd.read_csv("C:\\\\Users\\carrai1\\Desktop\\Projects\\HomeDepot\\datasets\\product_descriptions.csv")
	attributes_df = pd.read_csv("C:\\\\Users\\carrai1\\Desktop\\Projects\\HomeDepot\\datasets\\attributes.csv")
	test_separation = train_df.shape[0]

	all_df = pd.read_csv("C:\\\\Users\\carrai1\\Desktop\\Projects\\HomeDepot\\datasets\\df_all_fixV8.csv",encoding="ISO-8859-1",index_col=False)

	all_df["search_term_tokens"] = all_df["search_term_tokens"].fillna("")

	train_df = all_df.iloc[:test_separation]
	test_df = all_df.iloc[test_separation:]

	ytrain = train_df['relevance']
	xtrain = train_df.drop('relevance',axis=1)
	xtest = test_df.drop('relevance',axis=1)

	print (xtrain['search_term_tokens'].head(5))


	print ("----- Pipeline -----")
	print (round(((time.time() - start_time)/60),2))

	
	rfr = GradientBoostingRegressor(n_estimators=500,random_state=23)
	#rfr = AdaBoostRegressor(n_estimators=500, random_state=23)
	tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
	tsvd = TruncatedSVD(n_components=25, random_state = 1)
	clf = pipeline.Pipeline([
			('union', FeatureUnion(
						transformer_list = [
							('cst',  cust_regression_vals()),  
							('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term_tokens')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
							('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title_tokens')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
							('txt3', pipeline.Pipeline([('s2', cust_txt_col(key='product_description_tokens')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
							('txt4', pipeline.Pipeline([('s3', cust_txt_col(key='brand_tokens')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
							('txt5', pipeline.Pipeline([('s3', cust_txt_col(key='material_tokens')), ('tfidf5', tfidf), ('tsvd5', tsvd)])),
							],
						transformer_weights = {
							'cst': 1.0,
							'txt1': 1.0,
							'txt2': 1.0,
							'txt3': 0.1,
							'txt4': 0.5,
							'txt4': 0.2,
							},
					n_jobs = -1
					)), 
			('rfr', rfr)])
			
	warnings.filterwarnings("ignore", category=DeprecationWarning)	
	
	print ("----- fitting grid cv-----")
	print (round(((time.time() - start_time)/60),2))
	param_grid = {'rfr__min_samples_leaf' : [50,100,250,500],'rfr__subsample':[0.7],"rfr__learning_rate":[0.01]}
	model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20)
	model_fit = model.fit(xtrain, ytrain)

	print("Best parameters found by grid search:")
	print(model.best_params_)
	print("Best CV score:")
	print(model.best_score_)
	
	print ("----- predicting -----")
	print (round(((time.time() - start_time)/60),2))
	y_pred = model.predict(xtest)
	
	
	print ("----- checks -----")
	plt.hist(ytrain - model_fit.predict(xtrain) )
	plt.show()
	plt.scatter(ytrain,train_df["relevance"] - model_fit.predict(xtrain) )
	plt.show()
	plt.scatter(ytrain,model_fit.predict(xtrain))
	plt.show()
	

	# PREDICTIONS FIX (for predictions over 3)
	def fix_predictions (predictions):
		predictions_fix=[]
		for i in predictions:
			if (i>3):
				predictions_fix.append(3)
			elif (i<1):
				predictions_fix.append(1)
			else:
				predictions_fix.append(i)
		return predictions_fix

	#predictions_lr2_fix = fix_predictions(predictions_lr2)
	predictions_rf_fix = fix_predictions(y_pred)
	#plt.hist(predictions_lr2_fix, bins=20)
	#plt.show()
	plt.hist(predictions_rf_fix, bins=20)
	plt.show()

	print ("----- Creating sub -----")
	print (round(((time.time() - start_time)/60),2))

	all_df = pd.read_csv("C:\\\\Users\\carrai1\\Desktop\\Projects\\HomeDepot\\datasets\\df_all_fixV8.csv",encoding="ISO-8859-1",index_col=False)
	train_df = all_df.iloc[:test_separation]
	test_df = all_df.iloc[test_separation:]
	pd.DataFrame({"id": test_df["id"], "relevance": predictions_rf_fix}).to_csv('C:\\\\Users\\carrai1\\Desktop\\Projects\\HomeDepot\\subs\\solution_gb.csv',index=False)

	print ("----- END -----")
	print (round(((time.time() - start_time)/60),2))
	
	
	
	rfr = RandomForestRegressor(n_estimators=3000)
	tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
	tsvd = TruncatedSVD(n_components=25, random_state = 1)
	clf = pipeline.Pipeline([
			('union', FeatureUnion(
						transformer_list = [
							('cst',  cust_regression_vals()),  
							('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term_tokens')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
							('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title_tokens')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
							('txt3', pipeline.Pipeline([('s2', cust_txt_col(key='product_description_tokens')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
							('txt4', pipeline.Pipeline([('s3', cust_txt_col(key='brand_tokens')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
							('txt5', pipeline.Pipeline([('s3', cust_txt_col(key='material_tokens')), ('tfidf5', tfidf), ('tsvd5', tsvd)])),
							],
						transformer_weights = {
							'cst': 1.0,
							'txt1': 1.0,
							'txt2': 1.0,
							'txt3': 0.0,
							'txt4': 0.3,
							'txt4': 0.2,
							},
					n_jobs = -1
					)), 
			('rfr', rfr)])
			
	warnings.filterwarnings("ignore", category=DeprecationWarning)	
	
	print ("----- fitting grid cv-----")
	print (round(((time.time() - start_time)/60),2))
	param_grid = {'rfr__min_samples_leaf' : [50,100,250,500]}
	model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 3, verbose = 20)
	model_fit = model.fit(xtrain, ytrain)

	print("Best parameters found by grid search:")
	print(model.best_params_)
	print("Best CV score:")
	print(model.best_score_)
	
	print ("----- predicting -----")
	print (round(((time.time() - start_time)/60),2))
	y_pred = model.predict(xtest)
	
	
	print ("----- checks -----")
	plt.hist(ytrain - model_fit.predict(xtrain) )
	plt.show()
	plt.scatter(ytrain,train_df["relevance"] - model_fit.predict(xtrain) )
	plt.show()
	plt.scatter(ytrain,model_fit.predict(xtrain))
	plt.show()
	

	# PREDICTIONS FIX (for predictions over 3)
	def fix_predictions (predictions):
		predictions_fix=[]
		for i in predictions:
			if (i>3):
				predictions_fix.append(3)
			elif (i<1):
				predictions_fix.append(1)
			else:
				predictions_fix.append(i)
		return predictions_fix

	#predictions_lr2_fix = fix_predictions(predictions_lr2)
	predictions_rf_fix = fix_predictions(y_pred)
	#plt.hist(predictions_lr2_fix, bins=20)
	#plt.show()
	plt.hist(predictions_rf_fix, bins=20)
	plt.show()

	print ("----- Creating sub -----")
	print (round(((time.time() - start_time)/60),2))

	all_df = pd.read_csv("C:\\\\Users\\carrai1\\Desktop\\Projects\\HomeDepot\\datasets\\df_all_fixV8.csv",encoding="ISO-8859-1",index_col=False)
	train_df = all_df.iloc[:test_separation]
	test_df = all_df.iloc[test_separation:]
	pd.DataFrame({"id": test_df["id"], "relevance": predictions_rf_fix}).to_csv('C:\\\\Users\\carrai1\\Desktop\\Projects\\HomeDepot\\subs\\solution_rf.csv',index=False)

	print ("----- END -----")
	print (round(((time.time() - start_time)/60),2))
