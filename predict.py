# LOAN PREDICTION PROBLEM - ANALYTICS VIDHYA COMPETETION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mytraining import *

train = pd.read_csv('train.csv')


''' REPLACE ALL THE MISSING VALUES'''
def replace_missing(df):
	#print df.apply(lambda x : sum(x.isnull()), axis=0)
	#print df['LoanAmount'].median()

	#print df['Gender'].value_counts()
	#df['Gender'].fillna('Male', inplace=True)

	#print df['Married'].value_counts()
	#df['Married'].fillna('Yes', inplace=True)

	#print df['Self_Employed'].value_counts()
	df['Self_Employed'].fillna('No', inplace=True)

	#print df['Dependents'].value_counts()
	#df['Dependents'].fillna(0, inplace=True)

	#print df['Credit_History'].value_counts()
	df['Credit_History'].fillna('1.0', inplace=True)

	#print df['LoanAmount'].value_counts()
	table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
	#print table
	# Define function to return value of this pivot_table
	def fage(x):
	    return table.loc[x['Self_Employed'],x['Education']]
	# Replace missing values
	#df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
	df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

	#print df['Loan_Amount_Term'].value_counts()
	table1 = df.pivot_table(values='Loan_Amount_Term', index='Self_Employed' ,columns='Education', aggfunc=np.median)
	#print table
	# Define function to return value of this pivot_table
	def fage1(x):
	    return table1.loc[x['Self_Employed'],x['Education']]
	# Replace missing values
	#df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
	df['Loan_Amount_Term'].fillna(df[df['Loan_Amount_Term'].isnull()].apply(fage1, axis=1), inplace=True)

	
	#print df.apply(lambda x : sum(x.isnull()), axis=0)

	

	''' NORMALIZING THE DISTRIBUTIONS '''
	df['LoanAmount_Log'] = np.log(df['LoanAmount'])
	#df.hist(column='LoanAmount_Log', bins=50)
	#plt.show()


	df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
	df['TotalIncome_log'] = np.log(df['TotalIncome'])
	#df.hist(column='TotalIncome_log', bins=20)
	#plt.show()

	df['LoanAmount/TotalIncome'] = df['LoanAmount_Log'] / df['TotalIncome_log']
	#new_df = pd.DataFrame({'LoanAmount_Log':df['LoanAmount_Log'], 'TotalIncome_log':df['TotalIncome_log'],'LoanAmount/TotalIncome':df['LoanAmount/TotalIncome']})
	#print new_df.head(20)
	#df.hist(column = 'LoanAmount/TotalIncome', bins=20)
	#plt.show()
	
	#df['ApplicantIncome_log'] = np.log(df['ApplicantIncome'])
	#df.hist(column='ApplicantIncome_log', bins=20)
	#plt.show()
	
	df['CoapplicantExist'] = np.where(df['CoapplicantIncome'] > 0, 1, 0)
	#print df['CoapplicantExist']
	
	df['DependentsExist'] = np.where(df['Dependents'] == '0', '0', '1')
	#print df['DependentsExist']
	
	df['LoanPerMonth'] = df['LoanAmount'] / df['Loan_Amount_Term']
	df['LoanPerMonth_log'] = np.log(df['LoanPerMonth'])
	#df.hist(column='LoanPerMonth_log', bins=20)
	#plt.show()
		
	return df

train = replace_missing(train)




''' ----------BUILDING THE PREDICTIVE MODEL--------------'''
''' ENCODING THE LEVEL OF CATEGORICAL VARIABLES INTO NUMERIC'''
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Credit_History']
le = LabelEncoder()
for i in var_mod :
	train[i] = le.fit_transform(train[i])




outcome_var = 'Loan_Status'


''' LOGISTIC REGRESSION '''
'''
print '\n------ LOGISTIC REGRESSION ------'
model = LogisticRegression()
#predictor_var = ['Credit_History']
predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome','Dependents','Property_Area']
classification_model(model, train, predictor_var, outcome_var)
'''


''' DECISION TREE '''
'''
print '\n------ DECISION TREE ------'
model = DecisionTreeClassifier()
#predictor_var = ['TotalIncome_log', 'Credit_History', 'LoanAmount_Log']
predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome','Dependents','Property_Area']
classification_model(model, train, predictor_var, outcome_var)
'''


''' RANDOM FOREST '''

print '\n------ RANDOM FOREST ------'
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area','LoanAmount_Log','TotalIncome_log','LoanAmount/TotalIncome','ApplicantIncome','CoapplicantIncome','LoanPerMonth_log','CoapplicantExist','DependentsExist']
classification_model(model, train, predictor_var, outcome_var)


#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print featimp



''' RANDOM FOREST (AFTER FEATURE IMP MEASURES)'''

print '\n------ RANDOM FOREST (AFTER FEATURE IMP MEASURES) ------'
#model = RandomForestClassifier(n_estimators=50, min_samples_leaf=100, max_features=0.7)
model = RandomForestClassifier(n_estimators=100, min_samples_split=3, max_depth=5, max_features='auto',random_state = 42)
#predictor_var = ['Credit_History', 'TotalIncome_log', 'LoanAmount_Log','LoanAmount/TotalIncome','Dependents','Property_Area']
predictor_var = ['Credit_History', 'LoanPerMonth_log', 'LoanAmount/TotalIncome', 'TotalIncome_log']
classification_model(model, train, predictor_var, outcome_var)


''' XGBOOST '''
'''
print '\n------ XGBOOST ------'
from xgboost.sklearn import XGBClassifier 
model = XGBClassifier()
#predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','Dependents','Property_Area']
predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome','Dependents','Property_Area']
classification_model(model, train, predictor_var, outcome_var)
'''


''' SVM '''
'''
print '\n------ SVM ------'
# SVC with RBF kernel
from sklearn import svm
model = svm.SVC(kernel='rbf', gamma=0.7, C=0.7)
predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','Dependents','Property_Area']
#predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome','Dependents','Property_Area']
classification_model(model, train, predictor_var, outcome_var)
'''


''' Naive Bayes '''
'''
print '\n------ Naive Bayes ------'
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','Dependents','Property_Area']
predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome']
classification_model(model, train, predictor_var, outcome_var)
'''


''' Neural Network '''
'''
print '\n------ Neural Network ------'
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','Dependents','Property_Area']
predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome']
classification_model(model, train, predictor_var, outcome_var)
'''


''' Ensemble - Gradient Boosting Classifier '''
'''
print '\n------- Gradient Boosting Classifier ---------'
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(loss='deviance',learning_rate=0.05,n_estimators=100,max_features=4)
predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','Dependents','Property_Area']
#predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome']
classification_model(model, train, predictor_var, outcome_var)
'''


''' ENsemble - Bagged Decision Tree'''
'''
print '\n------------ ENsemble - Bagged Decision Tree ----------'
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
#predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','Dependents','Property_Area']
predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome']
classification_model(model, train, predictor_var, outcome_var)
'''


''' ENsemble - Extra Tree '''
'''
print '\n------------ ENsemble - Extra Tree  ----------'
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=100, max_features=4)
#predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','Dependents','Property_Area']
predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome']
classification_model(model, train, predictor_var, outcome_var)
'''


''' ENsemble - AdaBoost Classifier'''
'''
print '\n------------ ENsemble - AdaBoost Classifier  ----------'
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100, random_state=7)
predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','Dependents','Property_Area']
#predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome']
classification_model(model, train, predictor_var, outcome_var)
'''


''' Voting Ensemble for Classification '''
'''
print '\n------ Voting Ensemble for Classification ----'
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
# create the sub models
estimators = []
model1 = XGBClassifier()
estimators.append(('xgb', model1))
model2 = AdaBoostClassifier(n_estimators=100, random_state=42)
estimators.append(('ada', model2))
model3 = RandomForestClassifier(n_estimators=100, min_samples_split=3, max_depth=5, max_features='auto',random_state = 42)
estimators.append(('rf', model3))
# create the ensemble model
model = VotingClassifier(estimators)
#predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','Dependents','Property_Area']
#predictor_var = ['TotalIncome_log','LoanAmount_Log','Credit_History','LoanAmount/TotalIncome']
predictor_var = ['Credit_History', 'LoanPerMonth_log', 'LoanAmount/TotalIncome', 'TotalIncome_log', 'CoapplicantIncome']
classification_model(model, train, predictor_var, outcome_var)
'''



''' ----- PREPROCESSING THE TEST DATA ----------'''
test = pd.read_csv('test.csv')
''' REPLACE ALL THE MISSING VALUES'''
test = replace_missing(test)


''' ENCODING THE LEVEL OF CATEGORICAL VARIABLES INTO NUMERIC'''
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Credit_History']
le = LabelEncoder()
for i in var_mod :
	test[i] = le.fit_transform(test[i])


''' ----- PREDICTION ----------'''

prediction = predict_using_model(model, test, predictor_var)
submission = pd.DataFrame({   
"Loan_ID": test["Loan_ID"],      
"Loan_Status": prediction
    })
submission.to_csv('submission-rf-9.csv', index=False)

