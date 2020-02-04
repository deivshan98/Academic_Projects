#!/usr/bin/env python
# coding: utf-8

# # Econometric 322 Lab \#4

# ## <font color = blue> Assignment </font>
# 
# Use the water consumption data to estimate a simple regression model.  The water consumption data was introduced at the beginning of the semester and is available on Sakai.  The unknown parameters of a demand function have to be estimated.  Estimate a simple OLS model real per capita water consumption as a function of the real price per gallon.  No other variables are to be used since the purpose of this lab is just to have you become familiar with commands.

# # <font color = red> Documentation </font>

# ## <font color = blue> Abstract </font>

# In this lab, we attempted to use an OLS Model to estimate a simple regression model using the water consumption data. In this OLS model, the dependent variable was Per Capita Consumption, and the independent variable was real price. We then estimated the parameters of the OLS model by utilizing the statsmodel library to calculate the sum of sqaured residuals, and the standard error. Finally, we check that the sum of residuals equal to zero to finally conclude our lab.

# ## <font color = blue> Data Dictionary </font>

# | Variable | Values   | Source | Mnemonic |
#     |----------|----------|--------|---------|
#     | Aggregate Consumption | Millions of gallons, annual | Int'l Bottled Water | agg_consumption |
#     | Aggregate Revenue | Millions of dollars, annual/nominal | IBID. | agg_revenue |
#     | Per Capita Consumption | Gallons per person, annual |Calculated: agg\_consumption_pop |per_capita_consump |
#     | Nominal Price per Gallon | Nominal dollars | Calculated: agg_revenue/agg_cons. |price |
#     | Real Disposable Income per Capita | Real dollars, base = 2005, annual | Economic R. of Pres. 2010, Tbl. B-31 | real_dis_income |
#     | Food CPI | Index (Total Food & Beverages) | Economic R. of Pres. 2010, Tbl. B-60 |food_cpi |
#     | Population | Millions | Economic R. of Pres. 2010, Tbl. B-34 | pop |
#     | Real Price per Gallon | Real dollars, annual| Calculated: price/food_cpi | real_price  |

# # <font color = red> Tasks </font>

# ## <font color = blue> Load the Pandas and Statsmodels packages and give them aliases.  I recommend 'pd' and 'sm'.  You will also need the Statsmodels formula API for formulas.  Please see Lesson \#4 for examples.</font>

# In[13]:


##

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np


# ## <font color = blue> Import the water consumption data.  Set the row index to the years. </font>

# In[15]:


file = r'C:\Users\deivs\OneDrive\Desktop\water_con.xlsx'

df = pd.read_excel(file, index_col = 'year')


# ## <font color = blue> Print the first five (5) records. </font>

# In[16]:



df.head()


# ## <font color = blue> Estimate an OLS model using per capita consumption as the dependent variable and real price as the the independent variable.  Display the summary report.  See Lesson \#4 for an example.</font>

# In[19]:




formula = 'per_capita_cons ~ real_price'

mod = smf.ols( formula, data = df )

reg01 = mod.fit()

reg01.summary()


# ## <font color = blue> Retrieve and display the estimated parameters.  See Lesson \#4 for an example.</font>

# In[20]:




reg01.params


# In[21]:


sse = reg01.ssr
sse


# In[22]:


residuals_squared = reg01.resid**2

round( residuals_squared.sum(),2 )


# In[23]:


se_reg = np.sqrt( sse/( reg01.nobs-2))

round( se_reg,2 )


# In[24]:


round( reg01.resid.sum(), 4 )


# In[ ]:




