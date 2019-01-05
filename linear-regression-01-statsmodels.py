################################################################################################
#	name:	linear-regression-01-statsmodels.py
#	desc:	linear regression using statsmodels
#	date:	2018-07-14
#	Author:	conquistadorjd
#   reference: http://www.statsmodels.org/dev/examples/notebooks/generated/ols.html
################################################################################################
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd


print('*** Program started ***')
##########################################################################################################################
# Data downloaded from https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html
# Fire and Theft in Chicago
# In the following data pairs
# X = fires per 1000 housing units
# Y = thefts per 1000 population
# within the same Zip code in the Chicago metro area
# Reference: U.S. Commission on Civil Rights
# df = pd.read_excel('slr05.xls', sheet_name='slr05')
# x  = df['X'].tolist()
# y  = df['Y'].tolist()
# label = "Fires and thefts"


##########################################################################################################################
# Data downloaded from https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html
# In the following data
# X = chirps/sec for the striped ground cricket
# Y = temperature in degrees Fahrenheit
# Reference: The Song of Insects by Dr.G.W. Pierce, Harvard College Press
df = pd.read_excel('slr02.xls', sheet_name='slr02')
x  = df['X'].tolist()
y  = df['Y'].tolist()
label = "chirps per sec and temp"

print(df)

# # this is to preserve original x values to be used for plotting
x1=np.array(x)
y1=np.array(y)
print("std", np.std(x1))
print("std", np.std(y1))
# # This is needed as per statsmodel documentation
# print('x before : ' , x)
x=sm.add_constant(x)
# print('x after  : ' , x)
# ##################################### regression
model = sm.OLS(y,x)
results = model.fit()

print("summary : ",results.summary())
# # print('Parameters: ', results.params)
print( 'results.params : ',results.params)
# print(x1, y)
pc = stats.pearsonr(x1,y)
print('pc : ',pc)
tau = stats.kendalltau(x1,y)
# print(tau)
rho = stats.spearmanr(x1,y)
# print(rho)

# # creating regression line
xx= x1
# print(type(results.params[1]), results.params[1])
print('x1 ',type(x1), ' results.params[1] ', type(results.params[1]))
# print('calculations : ', pc*np.std(y1)/np.std(x1))
yy = results.params[0] + x1*results.params[1]
plt.scatter(x1,y,s=None, marker='o',color='g',edgecolors='g',alpha=0.9,label=label)
plt.plot(xx,yy)
plt.legend(loc=2)
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.title('C '+ "{:.3f}".format(results.params[0]) + ' M ' + "{:.3f}".format(results.params[1]) 
	   + ' PC '+ "{:.3f}".format(pc[0])
	   + ' tau ' + "{:.3f}".format(tau[0]) 
	   + ' rho ' + "{:.3f}".format(rho[0])
	   # + ' gamma ' + "{:.3f}".format(gamma)
	   , fontsize=10)

# Saving image
plt.savefig('linear-regression-01-statsmodels-'+ label+'.png')

# In case you dont want to save image but just displya it
plt.show()

print('*** Program ended ***')
