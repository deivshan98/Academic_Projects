#!/usr/bin/env python
# coding: utf-8

# # Econometric 322 Lab \#5: Basic OLS Modeling

# # <font color = red> Introduction </font>

# ## <font color = blue> Purpose </font>
# 
# The purpose of this lab is to introduce you to the application of the OLS methodology to an actual problem.
# 
# At the end of this lab, you will be able to:
# 
# 1. specify a linear model for a problem;
# 2. estimate a linear model in Statsmodel;
# 3. interpret key statistics;
# 3. identify shortcomings in the proposed linear model;
# 3. summarize the regression output;
# 1. estimate elasticities and judge their reasonableness;
# 1. build a model portfolio.

# ## <font color = blue> Problem </font> 
# 
# A few markets are key to the health of the economy. Autos and energy are two that most people would quickly cite. The housing market is another.  Housing data can be divided into housing permits, housing starts, new home sales, and existing home sales. In addition, housing can be viewed as single family or multifamily.
# 
# Existing home sales (as well as new home sales) are dependent on the future, expected state of the economy. For the future state, if there is a concern that the economy will go into a recession soon, people will be less willing to buy a new home for fear of losing their job, so housing sales will be weak or decline. But if the economy looks promising, then housing sales will be strong or increase.
# 
# Another factor important for the decision to buy a new home is the current mortgage rate or yield. Usually, people get a fixed rate which means they will be paying that same rate for many years, often 30 years. In essence, the mortgage rate adds to the price of a home.
# 
# In this lab, you will estimate a model to explain home sales as a function of the mortgage yield.

# ## <font color = blue> Assignment </font>
# 
# Locate annual data for U.S. New Houses Sold and New-home Mortgage Yields. I suggest using the following:
# 
# 1. Economic Report of the President: 2013, "Table B–56. New private housing units started, authorized, and completed and houses sold".  Use the "New houses sold" column.
#     
# 2. Economic Report of the President: 2013, "Table B–73. Bond yields and interest rates".  Use the "New home mortgage yields" column.
# 

# # <font color = red> Documentation </font>

# ## <font color = blue> Abstract </font>

# The purpose of this lab was to determine the relationship between New Houses Sold and Mortgage Yield, and the problem was whether or not our model was expansive enough to explain most, if not all, of the data. We used the OLS method to solve our problem, and through analysis of the resulting statistics learned that our model only explained a very miniscule amount of the data, however we concluded that there is a negative relationship between new houses sold and mortgage yield.

# ## <font color = blue> Data Dictionary </font>

# | Variable | Values   | Source | Mnemonic |
# 
# |  New Homes Sold | Thousands | Economic Report of the President, 2018 | new_homes_sold |
# 
# |Mortgage Yield| $ |Economic Report of the President, 2018|mort_yield|
# 

# # <font color = red> Pre-lab Questions </font>
# 
# Before you do any work, please think about the relationship among these macro variables. In particular, think how you would answer the following if called on in class:

# ## <font color = blue> What type of data is this and why (i.e., source and domain)? </font>

# This is a secondary source of data from the Economic Report of the President. Complied by collecting big data on the population, this data is a time series data set since it records the data for a mortgage yield and new houses sold throughout a range of multiple years. 

# ## <font color = blue> What is the mortgage yield?  Do not tell me that it's the yield on mortgage backed securities.  Think about the problem and then answer this question. </font>

# Mortgage yield is a measure of mortgage backed bonds, and is a useful tool to determine how successful the housing market is currently doing. If the mortgage yield is high, the amount of homes sold should be low.

# ## <font color = blue> How should mortgage yields or rates affect housing?  Positively?  Negatively? Explain. </font>

# Mortgage yields, while not the most influential factor for housing, should negatively affect housing prices, and indirectly, demand. All things considered, if there is a trend of rising mortgage rates over a period of time, demand will become weaker. This will lead to fewer new homes sold, thus negatively affecting housing.

# ## <font color = blue> In general, how do you think the housing market has behaved over the past, say, decade?  Explain. </font>

# The housing market has been steadily growing stronger since 2008, as a few key changes to the market have been made. For example, pre-crash loan products are mostly gone, and it is now required that everything must be fully documented with a down payment of at least 3.5% with most loan programs. These changes, among other, have led to the steady expansion of the housing market after the recession.

# ## <font color = blue> Write a tentative <u>specific</u> model.  Explain your model.</font>

# new_homes_sold = $\beta_0$ + $\beta_1$ * mort_yield + ε
# 
# new_homes_sold - new houses sold 
# 
# mort_yield - mortgage yield 
# 
#  ε - disturbances 

# ## <font color = blue> What is a good testable hypothesis?  Explain your testable hypothesis. </font>

# There is a negative correlation between the two paramaters, new houses sold and the mortgage yield, with the latter being the independent variable. When mortgage yield's go up, the amount of houses sold should go down, with all other factors being held constant and vice versa.

# ## <font color = blue> Write the statistical hypothesis to go along with your testable hypothesis.  Explain what you wrote.</font>

# First Testable Hypothesis:
# 
# $H_0$$mort $_$ yield$ : $\beta_1$ = 0
# 
# $H_A$$mort $_$ yield$ : $\beta_1$ < 0
# 
# mort_yield = New Home Mortgage Yield

# # <font color = red> Tasks and Questions </font>

# ## <font color = blue> Load the Pandas, Seaborn, and Statsmodels packages and give them aliases.  You will also need the Statsmodels formula API for formulas.  Please see Lesson \#4 for examples.</font>

# In[15]:



import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

import matplotlib.pyplot as plt


# ## <font color = blue> Import the data.  Set the row index to the years. </font>

# In[16]:



file = r'C:\Users\deivs\OneDrive\Desktop\ERP.xls'
df = pd.read_excel(file, index_col = "Year")


# ## <font color = blue> Print the first five (5) records. </font>

# In[17]:



df.head()


# ## <font color = blue> Graph the data </font>

# In[18]:




ax = df.plot(x = 'new_houses_sold', y = 'mort_yield', style = 'o', figsize = (12,6))
ax.set( xlabel = "New Homes Sold (Thousands)", ylabel = 'New Home Mortgage Yield', title = 'New Home Sales and Mortgage Yields')


# In[29]:


df.plot( kind = 'bar', x = 'new_houses_sold' , y = 'mort_yield')


# In[19]:


##sns.boxplot( x = 'new_homes_sold', y = "mort_yield", data = df)


# ## <font color = blue> When we build a regression model, we say we regress the dependent variable on the independent variable(s). For this lab, you will regress sales on yield.  Estimate an OLS model using sales as the dependent variable and yield as the independent variable.  Display the summary report.  See Lesson \#4 for an example.</font>

# In[20]:


formula = 'new_houses_sold ~ mort_yield'

mod = smf.ols(formula, data = df)

reg01 = mod.fit()

reg01.summary()


# ## <font color = blue> Retrieve and display the estimated parameters.  See Lesson \#4 for an example.</font>

# In[21]:


reg01.params


# In[22]:


reg01.ssr


# In[23]:


sxx = ((df.mort_yield - df.mort_yield.mean())**2).sum()
print (sxx)


# ## <font color = blue> Estimate a yield elasticity.  See Lesson \#4 for an example.</font>

# In[24]:


dYdX = reg01.params[1]

els = dYdX * (df.mort_yield.mean()/df.new_houses_sold.mean())
print( 'els =', round(els, 4))


# ## <font color = blue> Build a Model Portfolio.  See Lesson \#4 for an example.</font>

# In[25]:


summary_col([reg01], stars = True, info_dict = {'n': lambda x: "{0:d}".format(int(x.nobs)), 'R2': lambda x: "{:0.3f}".format(x.rsquared),})


# # <font color = red> Post-lab Questions </font>

# ## <font color = blue>  What is the relationship between home sales and mortgage yields? Is this the relationship you expected?  Plot the data as part of your analysis.</font>

# In[26]:


df.corr()
plt.matshow(df.corr())


# The relationship between home sales and mortgage yields is negative, yet very weak, as the correlation matrix lists the correlation as being -0.184. This is the relationship that I expected; I was very skeptical that the data would protray a positive relationship, as there are many factors that contribute towards the amount of new home sales. The model is missing a few key factors, and that is apparent in the results.

# ## <font color = blue> Test your statistical hypothesis about the effect of mortgage rates on new home sales.   What do you conclude?</font>

# In[27]:


from scipy.stats import anderson

result = anderson(df.mort_yield)

print('Test Stat: %.3f' % result.statistic)
print('Signifance Level: ', result.significance_level)
print( 'Critical Values: ', result.critical_values)


#  The data shows that the test statistic is in the range of critical values, thus there is not a statistical significance to force us to reject the null hypothesis. This supports the hypothesis that mortgage yield is not the only significant factor in determining new houses sold. Our model does not accurately reflect what we want to find, and only explains a small fraction of the new houses sold. To further test our statistical hypothesis, we need to look look at our P-value and compare it to 0.05. Since it is greater, we fail to reject the null hypothesis.

# ## <font color = blue> Interpret the $R^{2}$ from the regression output. What does it say about your model? </font>

# The R^2 value we derived from our model was 0.034, which shows that our model only accounts for 0.034 of the data. This shows that our model is not expansive enough to completely account for all of the data; there is another piece to the puzzle that we are missing to completely explain all of the data.

# ## <font color = blue> How can you make the model better?  What additional variables can you think of that should be included? Defend your answer.</font>

# In order to better optimize the model, economic growth and speculative demand should be additional variables to better account for the data set. These two factors should be able to explain more of the data set, as economic growth details the diposable income and consumer confidence in the current economy. If economic growth is high, the speculative demand for houses will be high as well, all factoring into the amount of new houses sold.

# ## <font color = blue> Interpret the F-statistic from the regression output.  What hypothesis does it test?  What do you conclude about your model based on the F-statistic?  </font>

# Our F Statistic is 1.542. This statistic tests our null hypothesis to see that all of the regression coefficients are zero. Since the F Statistic is the ratio of SSR to SSE, it tests the overall signifiance of the model; our F statistic shows a low signifiance at about 1.5.

# ## <font color = blue> Are existing home sales elastic or inelastic with respect to the mortgage yield?  Does this make sense?  Is it what you expect? Defend your answer.  Suppose there is a 1\% increase in mortgage rates. How much do home sales change in percentage terms? </font>

# Existing homes sales are inelastic with respect to the mortgage yield, as shown by the data; elasticity was calculated to be -0.2008. While this is not unitarily inelastic, with an 1% increase home sales would decrease by \beta 1 times

# ## <font color = blue> What is the practical significance of your model? In other words, what can this be used for, it for anything at all? How would you answer the question: "<i>So what?</i>" Defend your answer. </font>

# Based off of the model, this could be a weaker predictor for the housing market.It shows the negative relationship between mortgage yields and new homes sold, but attests that the relationship is very weak. The addition of other major factors can only strengthen the model, leading to a better assessment of the housing market, and indirectly, the status of current consumers, as house purchases are a big buy in a consumer's lifetime.

# In[ ]:




