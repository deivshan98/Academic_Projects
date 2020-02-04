#!/usr/bin/env python
# coding: utf-8

# # Econometric 322 Lab \#7: Self-Identified Problem

# # <font color = red> Collaboration Policy </font>
# 
# The submitted assignmenst must be your work.  There is to be no collaboration.

# # <font color = red> Purpose </font>
# 
# The purpose of this lab is to allow you to:
# 
# 1. identify and define your own economic problem;
# 
# 2. collect your own data;
# 
# 3. analyze your data using the tools you learned;
# 
# 4. estimate a multiple linear regression model in Pandas;
# 
# 5. interpret key statistics;
# 
# 6. identify shortcomings in the proposed linear model;
# 
# 5. summarize the regression output;
# 
# 6. estimate elasticities and judge their reasonableness;
# 
# 7. build a model portfolio;
# 
# 8. interpret the model results.

# # <font color = red> Problem Statement</font> 
# 
# State your problem, why it's a problem, and what you expect to show.

# Homelessness has been a big problem in the United States, and around the world, for many years now. People from all walks of life and tax brackets can become homeless in almost a blink of an eye. This lab will attempt to ascertain the factors of homelessness, and which states seem to have the highest amount of homeless individuals. I expect to see a positive relationship between homelessness and drug abuse, alcohol abuse, and percent of individuals with mental health issues as these factors logically seem to put people on the streets at their worst. I expect to show a negative relationship between homelessness and wages and housing prices.

# # <font color = red> Documentation </font>
# 
# Provide the appropriate documentation.

# Abstract
# 
# The purpose of this lab was to determine the relationship between homelessness and many independent variables, such as drug abuse, alcohol abuse, average wages, average housing prices, and amount of people affected by mental health. The problem was to see which state had the highest level of homelessness, and how each independent variable played a part in homelessness. But most importantly, I wanted to create a model that would be encompassing and account for a majority of data. I used the OLS method to solve the problem, and learned that the resulting analysis was only explained in part by my model. In short, my model was not enough to explain a good chunk of the data, however, I realized that the relationships I was looking for between the variables were accurate by my original hypothesis. The resulting data led me to reject the null hypothesis, as the data was significant enough.

# | drugabuse | Number of individuals | Statistical Abstract of the US 2012 | drug abuse    |
# 
# | alcabuse  | Number of individuals | Statistical Abstract of the US 2012 | alcohol abuse |
# 
# | wages     | Dollars               | Statistical Abstract of the US 2012 | Average wage  |
# 
# | housing   | Dollars               | Statistical Abstract of the US 2012 | Average housing price |
# 
# | mentalhealth | Number of Individuals | Statistical Abstract of the US 2012 | Percent of population affected by mental health issues |
# 
# | fedaid | Number of Programs | Statistical Abstract of the US 2012 | Amount of Federal Aid programs by State |
# 
# | belowpovlevel | Number of Individuals | Statistical Abstract of the US 2012 | Percent of Population who are homeless |

# # <font color = red> Pre-lab </font>
# 
# Data description, testable hypotheses, statistical hypotheses.

# homelessness = $\beta_0$ + $\beta_1$ * drugabuse + $\beta_2$ * alcabuse + $\beta_3$ * wages + $\beta_4$ * housing + $\beta_5$ * mentalhealth + ε 
# 
# This hypothesis shows the estimated relationship between each independent variable and homelessness. I expect to see a positive relationship between drug abuse, alcohol abuse, and homelessness, as substance abuse can lead a person to lose their house and job. I also expect to see a negative relationship between wages, housing prices, and homelessness since the higher the average wage, the less likely for workers to stay above poverty levels, and the higher the prices for accomodations, the more likely an affected individual will be below the poverty level. Finally, I expect to see a positive relationship between mental health rates and homelessness, since many people who are currently homeless are affected by mental health issues, and the inability to treat their disability only leads to it worsening.

# | Null     | Alternative |
# 
# | $H_{O, 1}: \beta_1 = 0$  | $H_{A, 1}: \beta_1 > 0$ |
# 
# | $H_{O, 2}: \beta_2 = 0$  | $H_{A, 2}: \beta_2 > 0$ |
# 
# | $H_{O, 3}: \beta_3 = 0$  | $H_{A, 3}: \beta_3 < 0$ |
# 
# | $H_{O, 3}: \beta_4 = 0$  | $H_{A, 4}: \beta_4 < 0$ |
# 
# | $H_{O, 3}: \beta_5 = 0$  | $H_{A, 4}: \beta_5 > 0$ |

# # <font color = red> Tasks </font>
# 
# Data import, data examination, model estimation, elasticity calculations, portfolio construction.  Be sure to include all you learned this semester.

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

from __future__ import print_function
from statsmodels.compat import lzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

import statsmodels.formula.api as smf 
from statsmodels.iolib.summary2 import summary_col


# In[4]:


file = r'C:\Users\deivs\OneDrive\Desktop\lab7.xlsx'
df = pd.read_excel(file)
df.head()


# In[5]:


df.describe().T


# In[6]:


ax = sns.pairplot( df, kind = 'reg', diag_kind = 'kde')


# In[7]:


ax = sns.boxplot( y = 'belowpovlevel', x = 'fedaid' , data = df).set_title('Distribution of Homelessness')


# In[8]:


ax = sns.scatterplot ( y = 'belowpovlevel', x = 'State', data = df)
ax.set_title("Distribution of Homelessness")
ax.ylabel = 'Homelessness'
ax.xlabel = 'Number of Federal Aid programs'


# In[9]:


##OLS

formula = 'belowpovlevel ~ drugabuse + alcabuse + wage + mentalhealth + housing + fedaid'

mod = smf.ols( formula, data = df)

reg01 = mod.fit()

print (reg01.summary())


# In[11]:


sns.heatmap(corr_m)


# In[17]:


hypothesis = '( drugabuse = 0, alcabuse = 0, wage = 0, housing = 0, mentalhealth = 0 )'
f_test = reg01.f_test(hypothesis)
f_test.summary()


# In[20]:


##heteroskedasticity

x = reg01.model.data.orig_exog
print(x.head())
print('\n')
print( reg01.resid.head())
white = sm.stats.diagnostic.het_white( reg01.resid, x)

ret = ['Test Statistic', 'p-Value', 'F Statistic', 'p-Value']
xzip01 = zip(ret, white)

print( '\nWhites Test for Heteroskedasticity')
lzip(xzip01)


# In[29]:


##vif

indepvar = ['drugabuse', 'alcabuse', 'wage', 'mentalhealth', 'housing']

x = np.diag( np.linalg.inv( corr_m))

xzip = zip(indepvar, x)

lzip( xzip)


# In[31]:


anova_lm(reg01)


# In[32]:


##regression for restricted

formula = 'belowpovlevel ~ drugabuse'
mod = smf.ols( formula, data=df)
reg02 = mod.fit()
print (reg02.summary())


# In[33]:


anova_lm( reg02 )


# In[34]:


anova_lm( reg02, reg01 )


# In[35]:


formula = 'belowpovlevel ~ 1'
mod = smf.ols( formula, data = df)
reg03 = mod.fit()
print (reg03.summary())


# In[36]:


##regression for restricted

formula = 'belowpovlevel ~ alcabuse'
mod = smf.ols( formula, data=df)
reg07 = mod.fit()
print (reg07.summary())


# In[37]:


anova_lm(reg07)


# In[38]:


##regression for restricted

formula = 'belowpovlevel ~ wage'
mod = smf.ols( formula, data=df)
reg04 = mod.fit()
print (reg04.summary())


# In[39]:


anova_lm(reg04)


# In[40]:


##regression for restricted

formula = 'belowpovlevel ~ mentalhealth'
mod = smf.ols( formula, data=df)
reg05 = mod.fit()
print (reg05.summary())


# In[41]:


anova_lm(reg05)


# In[42]:


##regression for restricted

formula = 'belowpovlevel ~ housing'
mod = smf.ols( formula, data=df)
reg06 = mod.fit()
print (reg06.summary())


# In[43]:


anova_lm(reg06)


# In[44]:


constant = anova_lm( reg03 )
constant


# In[45]:


##regression standard error (s)

reg_stderr = np.sqrt(constant)
print ('Regression Standard Error' + str(constant))


# In[46]:


anova_lm(reg03, reg01)


# In[47]:


##
## create a variable to hold the model names; this is a list
## 
model_names = [ 'Model ' + str( i ) for i in range( 1, 8) ]
##
## create a variable to hold the statistics to print; this is a dictionary
##
info_dict = { '\nn': lambda x: "{0:d}".format( int( x.nobs ) ),
              'R2 Adjusted': lambda x: "{:0.3f}".format( x.rsquared_adj ),
              'AIC': lambda x: "{:0.2f}".format( x.aic ),
              'F': lambda x: "{:0.2f}".format( x.fvalue ),
}
##
## create the portfolio summary table
##
summary_table = summary_col( [ reg01, reg02, reg03, reg04, reg05, reg06,reg07 ],
            float_format = '%0.2f',
            model_names = model_names,
            stars = True, 
            info_dict = info_dict 
)
summary_table.add_title( 'Summary Table for House Price Models' )
print( summary_table )


# In[48]:


##elasticities

dYdX = reg02.params[1]
eta = dYdX * (df.belowpovlevel.mean()/df.drugabuse.mean())
print( 'eta = ', round(eta, 4))


# In[49]:


dYdX = reg07.params[1]
eta = dYdX * (df.belowpovlevel.mean()/df.alcabuse.mean())
print( 'eta = ', round(eta, 4))


# In[50]:


dYdX = reg04.params[1]
eta = dYdX * (df.belowpovlevel.mean()/df.wage.mean())
print( 'eta = ', round(eta, 4))


# In[51]:


dYdX = reg05.params[1]
eta = dYdX * (df.belowpovlevel.mean()/df.mentalhealth.mean())
print( 'eta = ', round(eta, 4))


# In[52]:


dYdX = reg06.params[1]
eta = dYdX * (df.belowpovlevel.mean()/df.housing.mean())
print( 'eta = ', round(eta, 4))


# In[53]:


##
## retrieve the SSR (labeled ess in statsmodels) and SSE (labeled ssr in statsmodels) values
## why is Model 5 zero?
##
ssr = np.array( [ reg01.ess, reg02.ess, reg03.ess, reg04.ess, reg05.ess, reg06.ess, reg07.ess ] )
sse = np.array( [ reg01.ssr, reg02.ssr, reg03.ssr, reg04.ssr, reg05.ssr, reg06.ssr, reg07.ssr ] )
vars = np.array( [reg01.df_model, reg02.df_model, reg03.df_model, reg04.df_model, reg05.df_model, reg06.df_model, reg07.df_model ])
df_ssr = pd.DataFrame( ssr, index = [ 'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6', 'Model 7' ], columns = [ 'SSR' ] )
df_ssr['SSE'] = sse
df_ssr['Vars'] = vars
print( df_ssr )
ax = sns.barplot( x = ssr, y = df_ssr.index, data = df_ssr )
ax.set_title( 'SSR Values' )
ax.set_xlabel( 'SSR' )


# In[54]:


ax = sns.barplot( x = sse, y = df_ssr.index, data = df_ssr )
ax.set_title( 'SSE Values' )
ax.set_xlabel( 'SSE' )


# # <font color = red> Analysis and Conclusions </font>
# 
# Apply the analysis tools you learned.

# ## <font color = blue> What is the relationship between Homelessness and the independent variables? Is this the relationship you expected?  Are your testable hypotheses supportable and why?</font>

# The correlation matrix shows that the relationship between the variables are actually not as strong as I previously thought. The first regression (unrestricted) shows that the R^2 value was actually 0.599, and the adjusted R^2 is lower at 0.533. When I go through each of the restricted regressions, I find that the R^2 for wage is low (0.244) and housing is even lower ( 0.027) and most likely does not make enough of an impact by itself. Unfortunately, the other restricted regressions also produce a low R^2 score, as wage has the highest among all of the variables.
# This is not the relationship I expected in terms of R^2, however, I did correctly assume whether the respective relationships would be positive or negative. My testable hypothesis are supportable, however, as the first regression states, there is a strong chance of multicollinearity.

# ## <font color = blue>Interpret the  dummy variables.  What do you conclude about their statistical significance?</font>

# The dummy variables group the number of federal aid programs in each state, and offers a bigger view of homelessness by state. The resulting F value is 9.72609637 which is higher than the F-stat of 8.97. The P value is 3.18e-06, which is very small in comparison. This leads us to conclude that the dummy variables are in fact statistically significant.

# ## <font color = blue> Interpret the $R^{2}$ and $R^{2}$ adjusted from the regression output. What do they say about your model? Which is the better measure and why?</font>

# The regression output yields a R-squared value of .599, however, the adjusted R-Squared value is .533. The low values indicates that the model is not the best way to analyze violent crimes. The fact that the Adjusted R-Squared value is lower than the R-squared value shows that the addition of the variables does not improve the model. The adjusted R-squared value is the better measure to ascertain the value of a model, as it only increases if the added variables improve the model, as the value will go down if the variable does not add any value, unlike the R-squared value, which increases with each variable, regardless if it actually improves the model or not.

# ## <font color = blue> How can you make the model better?  What additional variables can you think of that should be included? Defend your answer.</font>

# The best way to make this model better is to add variables which will increase the adjusted R-squared value. Variables such as recent job layoffs, and rate of domestic violence should be included, as recent job layoffs records the mood of the economy and how it's doing, and the rate of domestic violence would show how stable households in each state are. Other prospective variables could be general age of the population, as it's less likely for a senior citizen to become homeless, compared to a younger individual. These variables, however, should be tested once added to the model to make certain that they improve the model, and do not muddy the results.

# ## <font color = blue> Interpret the F-statistic from the regression output.  What hypothesis does it test?  What do you conclude about your model based on the F-statistic?  </font>

# The F-Statistic, a ratio of two measures of variance (regression mean square over mean square errors), from the Regression output was 8.977, and since it is higher than the F value, our data is significant enough to reject the null hypothesis. The p-value also supports this claim, as it is low enough to show that all the data is not significant. 

# ## <font color = blue> Is Homelessness elastic or inelastic with respect to each of the independent variables?  Do they make sense?  Is it what you expect? Defend your answer. </font>

# Homeless is actually inelastic with every variable except for mentalhealth (0.0) and housing (0.02), which does make sense, as the other independent variables do not explicitly lead to homelessness. For example, drug abuse and alcohol abuse are factors which only influence homelessness in part through the mental health and housing. Wages are also negatively unit elastic (-0.0), which was very suprising as I expected it to be extremely elastic. Wages being unitarily elastic shows that any change in wages should result in an identical change in homelessness. Mentalhealth is also unitarily elastic, which was initially hard for me to grasp as to the reason why. I then did some research, and I learned that there are many people with mental health problems that are currently homeless, due to many reasons such as being unable to pay for care, or unable to work and unable to treat their disability. 

# ## <font color = blue> Interpret the correlation matrix </font>

# The correlation matrix shows that the strongest positive correlation (0.407601) to homelessness is the alcohol abuse. This was expected, given the reasoning that abuse of a substance would lead to people slowly slipping in life and falling deeper in the bottle. The matrix also shows a positive correlation between the wages and percent of individuals with mental health problems (0.390142), which was expected, as states above. I would have thought that homelessness would have corresponded better with housing prices, since there is a higher chance of people affected by the other independent factors to not be able to afford increased housing prices, however, this is not the case. It is very apparent that a few essential factors are missing.

# ## <font color = blue> What is the best model?  Basis?  Explain your answer. </font>

# The best model was the unrestricted model, as it had a R^2 value of  0.533. The unrestricted model also had a highest SSR, 2.772306e+02, and lowest SSE, 185.300200. These two sum of squares measure how much of the data is explained by the model and disturbances, respectively. The F Stat was 8.977, which is very high given the fact that we are looking for a F stat that is very low. Comparatively, however, the first model is the most encompassing, and thus, the best model.

# In[ ]:




