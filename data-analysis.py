import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')


#print train.head(10)

#print train.describe()

#print train['Property_Area'].value_counts()
#print train['Credit_History'].value_counts()

#train['ApplicantIncome'].hist(bins=50)
#plt.hist(train['ApplicantIncome'], bins=50)
#plt.show()

#plt.boxplot(train['ApplicantIncome'])
#train.boxplot(column='ApplicantIncome')
#plt.show()

#train.boxplot(column='ApplicantIncome', by = 'Education')
#train.boxplot(column='ApplicantIncome', by='Property_Area')
#plt.show()

#train.hist(column='LoanAmount', bins=50)
#plt.show()


#train.boxplot(column='LoanAmount', by = ['Education','Self_Employed'])
#plt.show()

'''
temp1 = train['Credit_History'].value_counts(ascending=True)
print temp1

temp2 = train.pivot_table(values='Loan_Status', index='Credit_History', aggfunc = lambda x : x.map({'Y':1, 'N':0}).mean())
print temp2
'''

'''
temp11 = train['Education'].value_counts(ascending=True)
print temp11

temp22 = train.pivot_table(values = 'Loan_Status', index='Education', aggfunc = lambda x : x.map({'Y':1 , 'N':0}).mean())
print temp22
'''

'''
temp33 = train['Property_Area'].value_counts(ascending=True)
#print temp33

temp44 = train.pivot_table(values = 'Loan_Status', index='Property_Area', aggfunc = lambda x : x.map({'Y':1 , 'N':0}).mean())
print temp44
'''

'''
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
temp2.plot(kind = 'bar')

plt.show()
'''

'''
temp3 = pd.crosstab(train['Credit_History'], train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','green'])
plt.show()
'''
