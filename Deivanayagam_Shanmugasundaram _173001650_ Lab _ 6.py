#!/usr/bin/env python
# coding: utf-8

# # Econometric 322 Lab \#6: Basic Multiple Regression

# # <font color = red> Introduction </font>

# ## <font color = blue> Purpose </font>
# 
# The purpose of this lab is to allow you to build and analyze a multiple regression model for an actual problem.
# 
# At the end of this lab, you will be able to:
# 
# 1. specify a multiple linear regression model for a problem;
# 
# 2. estimate a multiple linear regression model in Pandas;
# 
# 3. use dummy variables and interpret their impact;
# 
# 3. interpret key statistics;
# 
# 4. identify shortcomings in the proposed linear model;
# 
# 5. summarize the regression output;
# 
# 6. estimate elasticities and judge their reasonableness;
# 
# 7. build a model portfolio.

# ## <font color = blue> Problem </font> 
# 
# Crime is a topic on everyones' mind.  Politicians historically have raised the issue at election time, pointing out that there is a serious crime problem and that only they can solve it - if elected.  We now, of course, have substituted the word "terriorism" for violent crime, but the effect is the same: we feel insecure in our own homes.  Crime is actually divided into categories, violent crime being just one.  In this lab, you will estimate a complex model to explain violent crime rates at the state level in the United States.

# ## <font color = blue> Assignment </font>
# 
# Using the Statistical Abstract of the U.S. (2012 edition), find the total violent crime rate data by state and collect data on each state for the total violent crimes in that state.  Collect data by state on the total unemployment rate,  Gross State Product (would you use Current or Real Dollars?), personal income (would you use Current or Real Dollars?), and one other variable of your choice that you believe affects violent crime.
# 
# The U.S is divided into four Census regions.  These can also be found online (google "MapStats: United States").  You previously used a file that mapped states to regions.  Merge this file with your DataFrame.  Create dummy variables for the regions and include the dummies in your models.
# 
# Be sure to graph your data and interpret the graphs. All graphs must be clearly labeled. 
# 

# # <font color = red> Documentation </font>

# ## <font color = blue> Abstract </font>

# The purpose of this lab was to determine the relationship between violent crimes and 4 other variables (high-school graduates, unemployment rate, personal income, and gross state product). The main problem was whether or not the model's were expansive enough to explain most, if not all, of the data, and which model did that the best. We used the OLS method and regressions to analyze the relationship between the data and the model. Through the resulting data, we concluded that our data was significant enough to reject the null hypothesis and determine the resulting relationship, however, the adjusted r-squared begged for more encompassing variables, in terms of improving the model.

# ## <font color = blue> Data Dictionary </font>

# | Variable | Values   | Source | Mnemonic |
# |----------|----------|--------|---------|
# |  amount of High school Graduates | population | Statistical Abstract of the US | amtgrad |
# 
# | unemployment rate | percent | Statistical Abstract of the US | amtunemp |
# 
# | personal income | Dollars | Statistical Abstract of the US | income |
# 
# | gross state product | Dollars | Statistical Abstract of the US | gsp |

# # <font color = red> Pre-lab Questions </font>
# 
# Before you do any work, please think about the relationship among these variables. In particular, think how you would answer the following if called on in class:

# ## <font color = blue> What type of data is this and why (i.e., source and domain)? </font>

# This data is a time-series data set, taken from multiple tables from the 2012 edition of the Statistical Abstract of the US, and therefore the data is from a secondary source.

# ## <font color = blue> What are good testable hypothesis?  Explain your testable hypotheses. </font>

# I believe that crime rates will be higher in states with high unemployment, low gross state product, low personal income, and a low rate of high school graduation. To simplify, I believe that crime rates share a positive relationship with unemployment, as the higher the unemployment rate, the higher the potential for crime, and a negative relationship with income, gross state product, and high school graduation rates. My reasoning for a negative relationship is because the higher each of these variables are, the less likely for individuals to commit violent crimes.

# ## <font color = blue> Write a tentative <u>specific</u> model.  Explain your model.  </font>

# crime = $\beta_0$ + $\beta_1$ * amtgrad + $\beta_2$ * amtunemp + $\beta_3$ * gsp + $\beta_4$ * income + ε 
# 
# 
# 
# This tentative specific model shows the estimated relationship between crime and the 4 variables. Amtgrad is the amount of high school graduates from the corresponding state, and is estimated to have a negative relationship with crime. The next variable is the unemployment rate among eligble workers, and should have a positive relationship with crime. Gross State Product is abbreviated as 'gsp', and like income, the final variable, has a positive relationship with crime.

# ## <font color = blue> Write the statistical hypotheses to go along with your testable hypothesis.  Explain what you wrote.</font>

# | Null     | Alternative |
# |----------|-------------|
# | $H_{O, 1}: \beta_1 = 0$  | $H_{A, 1}: \beta_1 < 0$  |
# | $H_{O, 2}: \beta_2 = 0$  | $H_{A, 2}: \beta_2 > 0$  |
# | $H_{O, 3}: \beta_3 = 0$  | $H_{A, 3}: \beta_3 < 0$ |
# | $H_{O, 3}: \beta_4 = 0$  | $H_{A, 4}: \beta_4 < 0$ |
# 
# 

# In[ ]:





# # <font color = red> Tasks </font>

# ## <font color = blue> Load the Pandas, Seaborn, and Statsmodels packages and give them aliases.  You will also need the Statsmodels formula API for formulas.</font>

# In[13]:


import pandas as pd
import numpy as np

import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.api import anova_lm

import matplotlib.pyplot as plt


# ## <font color = blue> Import the data.  Set the row index to the states. </font>

# In[14]:


file = r'C:\Users\deivs\OneDrive\Desktop\lab6.xlsx'

df2 = pd.read_excel(file, index_col = 'state')
df2.head()


# ## <font color = blue> Import the state/region data and merge with your DataFrame.  Merge on the indexes of both DataFrames as you did in a prior lab.</font>

# In[15]:


f = r'C:\Users\deivs\OneDrive\Desktop\statesRegionsMapping.xlsx'
df1 = pd.read_excel(f, index_col = 'state')


df = pd.merge(df1, df2, on = 'state')


# ## <font color = blue> Print the first five (5) records. </font>

# In[16]:


df.head()


# ## <font color = blue> Graph the data. </font>

# In[17]:


graph = sns.pairplot(df, kind = 'reg', diag_kind = 'kde')


# ## <font color = blue> Create a correlation matrix </font>

# In[67]:


x = df[ df.columns[ 0:5]].corr()
x.corr()


# ## <font color = blue> Graph the correlation matrix. </font>

# In[68]:


##plt.matshow(df.corr())

sns.heatmap(x).set_title('Heatmap of the Correlation Matrix')


# ## <font color = blue> Estimate a regression model with crime as the dependent variable.  Display the summary report.</font>

# In[56]:


formula = 'crime ~ amtunemp + amtgrad + income + gsp + Region'
mod = smf.ols(formula, data = df)
reg01 = mod.fit()
print(reg01.summary())


# ## <font color = blue> Test the null hypothesis that all dummies are statistically zero. </font>

# In[59]:


df.Region.value_counts()


# In[61]:


reg01.model.data.orig_exog.head()


# In[64]:


hypothesis = ' (C(Region)[T.Northeast] = 0,C(Region)[T.South] = 0,C(Region)[T.West] = 0) '

f_test = reg01.f_test(hypothesis)
f_test.summary()


# ## <font color = blue>Estimate two additional models.</font>

# In[22]:


formula = 'crime ~ gsp'
mod = smf.ols ( formula, data = df)
reg04 = mod.fit()
print ( reg04.summary())


# In[23]:


formula = 'crime ~ amtunemp'
mod = smf.ols ( formula, data = df)
reg03 = mod.fit()
print ( reg03.summary())


# ## <font color = blue>Estimate a constant-only model (the restricted model).</font>

# In[24]:


formula = 'crime ~ 1'
mod = smf.ols(formula, data = df)
reg02 = mod.fit()
print ( reg02.summary())


# ## <font color = blue>Compare your unrestricted model to the constant-only model (the restricted model).</font>

# In[63]:


anova_lm( reg01, reg02 )


# ## <font color = blue> Calculate elasticities.</font>

# In[43]:


dYdX = reg03.params[1]
eta = dYdX * (df.crime.mean()/df.amtunemp.mean())
print ('eta = ', round(eta, 4))


# In[42]:


dYdX = reg01.params[1]
eta = dYdX * (df.crime.mean()/df.amtgrad.mean())
print ('eta = ', round(eta, 4))


# In[44]:


dYdX = reg04.params[1]
eta = dYdX * (df.crime.mean()/df.income.mean())
print ('eta = ', round(eta, 4))


# In[74]:


dYdX = reg01.params[1]
eta = dYdX * (df.crime.mean()/df.gsp.mean())
print ('eta = ', round(eta, 4))


# ### Elasticities​ Summary Table

# | Variable | Model    | Estimate | Mean | Elasticity | Interpretation |
# |amtunemp |3 |2646.15|0.085|13255916.2504| Extremely Elastic|
# | amtgrad | 1   | 0.04 | 12024.98 | 93.5687 | Elastic |
# | gsp | 1    | -0.01 | 253814.66 | -0.271 | Relatively Inelastic |
# | income | 4    | 0.0 | 42270.58 | 0.0 | Unit Elastic |

# In[71]:


ssr = np.array( [ reg01.ess, reg02.ess, reg03.ess, reg04.ess ] )
sse = np.array( [ reg01.ssr, reg02.ssr, reg03.ssr, reg04.ssr ] )

vars = np.array( [reg01.df_model, reg02.df_model, reg03.df_model, reg04.df_model])
df_ssr = pd.DataFrame( ssr, index = [ 'Model 1', 'Model 2', 'Model 3', 'Model 4' ], columns = [ 'SSR' ] )
df_ssr['SSE'] = sse
df_ssr['Vars'] = vars
print( df_ssr )

ax = sns.barplot( x = ssr, y = df_ssr.index, data = df_ssr )
ax.set_title( 'SSR Values' )
ax.set_xlabel( 'SSR' )


# ## <font color = blue> Build a Model Portfolio.</font>

# In[54]:


model_names = ['Model' + str(i) for i in range ( 1,6)]

info_dict = { '\nn': lambda x: "{0:d}".format( int( x.nobs ) ),
              'R2 Adjusted': lambda x: "{:0.3f}".format( x.rsquared_adj ),
              'AIC': lambda x: "{:0.2f}".format( x.aic ),
              'F': lambda x: "{:0.2f}".format( x.fvalue ),
            }

summary_table = summary_col( [ reg01, reg02, reg03, reg04 ],
            float_format = '%0.2f',
            model_names = model_names,
            stars = True, 
            info_dict = info_dict 
                           )

summary_table.add_title ( 'Summary Table for House Price Models')
print(summary_table)


# # <font color = red> Post-lab Questions </font>

# ## <font color = blue> What is the relationship between violent crime and the independent variables? Is this the relationship you expected?  Are your testable hypotheses supportable and why?</font>

# The correlation matrix shows that all of the variables are positively correlated with crime, which was a surprise given the original hypothesis that amtgrad (amount of graduates) and gsp (gross state product) were negatively correlated with violent crime. I, however, accurately predicted that the other two variables would be positively correlated with violent crime. The fact that the R Squared value being quite low shows that the models are not fully accountable for total violent crime, and casts some doubt on the results. This hypothesis is supportable by the fact that it can be replicated and should yield the same results. The addition of other variables will lead to a higher R Squared value, and a more encompassing model. 

# ## <font color = blue>Interpret the regional dummy variables.  What do you conclude about their statistical significance?</font>

# The dummy variables group each of the states into their respective regions, and offers a bigger view of total violent crimes by region. The resulting F value is 3.48 which is considerably lower than the F stat, which is 4.361, and the p-value is 0.0157. This leads us to conclude that the dummy variables are statistically significant, and support the decision to reject the null hypothesis.

# ## <font color = blue> Interpret the $R^{2}$ and $R^{2}$ adjusted from the regression output. What do they say about your model? Which is the better measure and why?</font>

# The regression output yields a R-squared value of .433, however, the adjusted R-Squared value is .334. The low values indicates that the model is not the best way to analyze violent crimes. The fact that the Adjusted R-Squared value is lower than the R-squared value shows that the addition of the variables does not improve the model. The adjusted R-squared value is the better measure to ascertain the value of a model, as it only increases if the added variables improve the model, as the value will go down if the variable does not add any value, unlike the R-squared value, which increases with each variable, regardless if it actually improves the model or not.

# ## <font color = blue> How can you make the model better?  What additional variables can you think of that should be included? Defend your answer.</font>

# The best way to make this model better is to add variables which will increase the adjusted R-squared value. Variables such as general happiness, and police policy should be included, as general happiness shows the mood of the population, and police policy would see to it that a majority of violent crimes are either prevented or convicted promptly. Other prospective variables could be general age of the population, as it's less likely for a senior citizen to commit a violent crime compared to a younger individual. These variables, however, should be tested once added to the model to make certain that they improve the model. 

# ## <font color = blue> Interpret the F-statistic from the regression output.  What hypothesis does it test?  What do you conclude about your model based on the F-statistic?  </font>

# The F-Statistic, a ratio of two measures of variance (regression mean square over mean square errors), from the Regression output was 4.361, and since it is higher than the F value, our data is significant enough to reject the null hypothesis. The p-value also supports this claim, as it is low enough to show that all the data is significant. 

# ## <font color = blue> Are violent crimes elastic or inelastic with respect to each of the independent variables?  Do they make sense?  Is it what you expect? Defend your answer. </font>

# Violent crimes are elastic with each of the variables except for Gross State Product (-0.27), which was expected, as the output of a state doesn't have a huge impact on violent crimes by itself. The unemployment rate (13255916.2504) is, unsurprisingly, extremely elastic as a lack of job opportunities can lead to crimes being commited, along with the amount of graduates (93.6), as a GED is very important in securing most jobs. The variable personal income was actually unit elastic, which makes sense as the higher the income, the less the chance of the population has of committing crimes. 

# ## <font color = blue> Interpret the correlation matrix </font>

# The correlation matrix shows that the strongest positive correlation (0.370565) to violent crime is the unemployment rate. This was expected, given the reasoning that higher unemployment would lead to people committing crimes due to the lack of job opportunities. The matrix also shows a positive correlation between the Gross State Product and violent crime (0.202531), which was unexpected. I would have thought that violent crime would have corresponded with GSP if violent crimes were measured by total crimes, and since states with higher GSP's usually is populated with more people, increasing the chances of potential crimes.

# ## <font color = blue> What is the best model?  Basis?  Explain your answer. </font>

# The best model is the third model, as it has the highest Adjusted R-squared value (0.119), and SSR value (162651.139348) compared to its SSE value. This shows that the third model is the most encompassing out of the three, and offers an better explanation for  total violent crime. This model regressed the unemployment rate as the variable relating to crime. 

# ## <font color = blue> What is the practical significance of your model? In other words, what can this be used for, it for anything at all? How would you answer the question: "<i>So what?</i>" Defend your answer. </font>

# The practical significance of this model is in it's use as a way to pinpoint which states, and potentially which towns, will be the most likely to commit more violent crimes. With a more encompassing model, the government could find and properly take care of states with the most potential for violent crimes. By utilizing accurate data for the variables, the government could also make sure to prevent the variables from leading to the increase of potential violent crimes; for example, if the unemployment rate or amount of graduates drops to levels where it could influence violent crimes, the government would implement legislation to prevent that from happening.

# In[ ]:




