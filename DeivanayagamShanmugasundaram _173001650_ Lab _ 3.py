#!/usr/bin/env python
# coding: utf-8

# # Econometric 322 Lab \#3

# ## <font color = blue> Assignment </font>
# 
# Collect data on the following:
# 
# - Physicians and nurses by state
# 
# Use the reference in Lesson 3.

# # <font color = red> Documentation </font>

# ## <font color = blue> Abstract </font>

# The purpose of this lab was to manipulate data on the population and distribution of physicians and nurses across the country. The main problem was to see whether or not the population would be normally distributed throughout the country. We utilized the pandas and seaborn library to manipulate and visually represent the data in a clear, concise view. We came to the conclusion that there is not a normal distribution of physicians across the country, rather, there is a skewed distribution, leaning toward states with a higher population.

# ## <font color = blue> Data Dictionary </font>

# | Variable | Values   | Source | Mnemonic |
# |----------|----------|--------|---------|
# 
# | Total Physicians | -- | US Government Census |  tot_phys  |
# | Rate of Physicians | Individuals | US Government Census | rate_phys | 
# | Total Nurses | Individuals | US Government Census | tot_nurs |
# | Rate of Nurses | Individuals | US Government Census | rate_nurs |
# 

# # <font color = red> Pre-lab Questions </font>
# 
# Before you do any work, please think about the relationship among these macro variables. In particular, think how you would answer the following if called on in class:

# ## <font color = blue> What type of data is this and why (i.e., source and domain)? </font>

# This data is a secondary source from the online libraries of the US Census. It is a cross-sectional data set, as it provides multiple points of data for a single period of time.

# ## <font color = blue> What pattern do you expect to see for physicians by state?  Explain your answer. </font>

# I expect to see a greater concentration of physicians in states with either a high general population, or a population with a high percentage of minor's and seniors. While most would think that physicians would be normally distributed throughout the country, I believe otherwise because physicians would mostly concentrate in areas where there is a great demand for medical practicioners. Thus, states with high populated cities would be more densely populated with physicians. These states would also influence the distribution of pracitioners per region.

# ## <font color = blue> What should you do to the physician and nurse data before you do any analytical work? </font>

# The tables need to be reformatted in order to cleanly import the data through Anaconda, as Python requires simplified table titles and numbers to manipulate the data. There is also a technical problem with the data, as it include the District of Columbia, and displays a total of 51 states. This requires us to drop that datapoint by utilizing code. 

# # <font color = red> Tasks and Questions </font>

# ## <font color = blue> Load the Pandas and Seaborn packages and give them aliases.  I recommend 'pd' and 'sns'. </font>

# In[75]:



import pandas as pd
import seaborn as sns


# ## <font color = blue> Import the physician and nurse data.  Set the row index to the state names. </font>

# In[86]:



file = r'C:\Users\deivs\OneDrive\Desktop\lab3phys.xls'

df1 = pd.read_excel( file, index_col = 'State')


# ## <font color = blue> Print the first five (5) records. </font>

# In[87]:



df1.head()


# ## <font color = blue> Import the state information (see Lesson \#3).  Set the row index to the state names.</font>

# In[78]:



file = r'C:\Users\deivs\OneDrive\Desktop\lab3region.xlsx'
df2 = pd.read_excel( file, index_col = 'State')


# ## <font color = blue> Print the first five (5) records. </font>

# In[88]:



df2.head()


# ## <font color = blue> Merge the physician/nurse data and the state data as described in Lesson \#3. </font>

# In[102]:



x = pd.merge( df1, df2, right_index = True, left_index = True, how = 'inner')
x.shape
x.head()


# ## <font color = blue>Recall your answer to the question above regarding what you should do to the physician and nurse data before you do any analytical work.  Make the correction here.   Be sure to use the corrected data for the following tasks.</font>

# In[90]:




df1.drop(df1.index[10], inplace = True)


# ## <font color = blue> Create summary statistics for the physician and nurse data. </font>

# In[91]:



df1.describe().T


# ## <font color = blue> Plot the physician and nurse data using graphs you learned in Stat 101. </font>

# In[92]:




ax = df1.plot( x='tot_nurs', y = 'tot_phys', legend = False, style = 'o', figsize = (12,6))
ax.set(xlabel = 'Nurses', ylabel = 'Physicians', title = 'Physicians vs Nurses')


# ## <font color = blue> Plot the physicians by region.  What graph type would you use? </font>

# In[104]:



ax = x.plot.bar(x = 'Region', y = 'tot_phys', legend = False)
ax.set( xlabel = 'Region', ylabel = 'Physicians', title = 'Physicians by Region')


# ## <font color = blue> Print and graph a correlation matrix. </font>

# In[97]:



w = pd.merge ( df1, df2, left_index = True, right_index = True, how = 'inner')



x = w.corr()
sns.heatmap(x).set_title('Heatmap of the Correlation Matrix')
x.corr()


# ## <font color = blue> Plot a histogram of physicians. </font>

# In[95]:



ax = sns.distplot(df1.tot_phys).set_title('Histogram for Normality')


# ## <font color = blue> Perform a normality test of the physicians distribution. </font>

# In[96]:




from scipy.stats import anderson

result = anderson(df1.tot_phys)

print('Test Stat: %.3f' % result.statistic)
print('Signifance Level: ', result.significance_level)
print( 'Critical Values: ', result.critical_values)


# # <font color = red> Post-Lab Questions </font>

# ## <font color = blue> Interpret the summary statistics. </font>

# There seems to be a greater average amount of nurses than physicians for each 50 states; approximately 51,000 for nurses compared to just about 16,000 for the physicians. There is also a higher standard deviation for nurses is higher per state than the physicians, which also hints at the scarcity of physicians compared to nurses.

# ## <font color = blue> Interpret the graphs.  What do they tell you about the distribution of physicians around the country? </font>

# The graphs reaffirm my pre-lab hypothesis, which stated that there would be a greater percent of practioners in states with either a high general population, or a high minor/senior population. The distribution of physicians around the country are clustered around states such as California, New York, and Florida, all states which boast the higher average population among American states. These states are, however, outliers in regards to their population, but the majority of the graphs show that there is still a higher concentration of physicians in states with a higher population. This interpretation can be seen when one reads the middle of each graph, which does not usually contain outliers.

# ## <font color = blue> What are the Null and Alternative Hypotheses for the normality test of physicians? Explain your answer. </font>

# $H_0$$Physicians$ : $\beta_1$ = 0        
# 
# $H_A$$Physicians$ : $\beta_1$ > 0  
# 
# Our null hypothesis stated that the distribution of the population parameter (Physicians) would have no change, and the histogram would be normally distributed. Our alternative hypothesis claimed that the population parameter would show either a r positive skew as opposed to the null hypothesis. 

# ## <font color = blue> Are physicians normally distributed?  That is, do you reject or fail to reject your Null Hypothesis?  Explain you answer. </font>

# The data shows that physicians are not normally distibuted, as the test statistic is greater than the critical values, so we can reject the null hypothesis. We also see this theme in the histogram above. According to our null hypothesis, the histogram should have a normal distribution: obviously, it doesn't, which played a great role in our rejection of the null.

# ## <font color = blue> What can you observe about the correlation matrix? Explain. </font>

# There seems to be a lower correlation between physicians and population (0.965) than nurses and population (0.97), which is very interesting when one sees that there is also a low correlation between physicians and nurses (0.968). There seems to be a third correlating factor to explain a deeper relationship between the variables, and I believe that to be age. The heat map mirrors the correlation matrix, and visually displays how closely correlated the relationships are.
