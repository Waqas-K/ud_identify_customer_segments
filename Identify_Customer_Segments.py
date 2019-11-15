#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# #### Read Tables

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv',sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv',sep=';')

# Load in the Customer data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv',sep=';')

# Read Data Dictionary File and print
f = open('Data_Dictionary.md', "r")
print(f.read())


# #### Plot Table infos

# In[3]:


# Plot table info
print(azdias.info())
print(feat_info.info())
print(customers.info())


# #### Check Azdias table

# In[4]:


# Show shape of table
print('\n Azdias Table Size',azdias.shape)

#Plot first rows of table
azdias.head(5)


# #### Check Feature info table

# In[5]:


# Plot visual of missing values within table
plt.figure(figsize=(20,8))
sns.heatmap(data=feat_info.isnull(),cmap='viridis',yticklabels=False,cbar=False)

# Show shape of table
print('\n Feature Table Size',feat_info.shape)

#Plot first rows of table
feat_info.head(5)


# #### Check Customer table

# In[7]:


# Plot visual of missing values within table
plt.figure(figsize=(20,8))
sns.heatmap(data=customers.isnull(),cmap='viridis',yticklabels=False,cbar=False)

# Show shape of table
print('\n Customers Table Size',customers.shape)

#Plot first rows of table
customers.head(5)


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# #### Create a dataframe of missing data for statistics

# In[5]:


missing_data_stats=pd.DataFrame(azdias.isnull().sum(),columns=['Naturally_Missing_Values'])
missing_data_stats.head(5)


# #### Find missing or unknown values from each row in feat_info and replace corresponding values in azdias table with NaNs

# In[6]:


# Loop over all the rows/attributes in feat_info
for i in range(len(feat_info)):
    att=feat_info['attribute'][i]                                                         # Get name of attribute
    codes=feat_info['missing_or_unknown'][i].replace('[',"").replace(']',"").split(',')   # Get values from string
    
    # Loop over all the individual values extratcted from '	missing_or_unknown' column
    for j in range(len(codes)):
        if codes[j]!='':                                       # skip empty values as nothing needs to be replaced
            try:
                azdias.loc[azdias[att]==int(codes[j]),att] = np.nan       # replaces numeric values with NaNs
            except:
                azdias.loc[azdias[att]==(codes[j]),att] = np.nan          # replaces string values with NaNs


# In[7]:


# Merge and plot the naturally missing and overall missing values
missing_data_stats=pd.merge(missing_data_stats,pd.DataFrame(azdias.isnull().sum(),columns=['Overall_Missing_Values']),
                            left_index=True,right_index=True)
missing_data_stats.head(5)


# In[8]:


#Plot
sns.jointplot(missing_data_stats['Overall_Missing_Values'], missing_data_stats['Naturally_Missing_Values'],kind='hex')


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[9]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.
missing_data_stats['Overall_Missing_Percentage']=(missing_data_stats['Overall_Missing_Values']/len(azdias))*100
sorted_missing_data_stats=missing_data_stats.sort_values(by='Overall_Missing_Percentage',ascending=False)
sorted_missing_data_stats.head(5)


# In[11]:


# Investigate patterns in the amount of missing data in each column.
plt.figure(figsize=(15,30))
sns.barplot(x='Overall_Missing_Percentage', y=sorted_missing_data_stats.index, 
            data=sorted_missing_data_stats);
plt.title('Percentage of Missing Data by Attribute',size=20)


# In[12]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)

# Define criteria of outliers removal
outlier=30    # Outlier if % missing more than

#Get outlier columns and drop them from data
outlier_columns=sorted_missing_data_stats[sorted_missing_data_stats['Overall_Missing_Percentage']>outlier].index
for i in outlier_columns:
    azdias.drop(i,axis=1,inplace=True)

# View the table after dropping the outlier columns
azdias.head(2)


# In[13]:


# Calculate the count of wells missing similar percentage of data
sorted_missing_data_stats['Overall_Missing_Percentage_Rounded']=sorted_missing_data_stats['Overall_Missing_Percentage'].round()
sorted_missing_data_stats.groupby(by='Overall_Missing_Percentage_Rounded').count()


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# The amount of data missing in each column varies for 0% all the way to almost 100% (as seen from table above). There are about 25 columns with no missing data and around 6 with 1% missing data.Most of the other columns have 8-15% data missing. 6 columns have more than 30% data missing, these were classified as outliers and removed from the dataset.

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[14]:


# How much data is missing in each row of the dataset?

# Sum all the missing values by rows
nan_count_rows=azdias.isnull().sum(axis=1).sort_values(ascending=False)

# Plot and visualize the distribution of missing rows
plt.figure(figsize=(20,8))
sns.countplot(nan_count_rows)

plt.xlabel('Missing Values in each Row')
plt.ylabel('Count of Rows')
plt.title('Count of Rows vs Missing Values in each Row')


# In[15]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
# Define threshold value
row_nan_threshold=0

# Rows above threshold
at=azdias.loc[nan_count_rows>row_nan_threshold]

# Rows below threshold
bt=azdias.loc[nan_count_rows<=row_nan_threshold]


# In[16]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.

# Make a function to plot comparison between the two subsets
def make_countplot(column_name,data_at,data_bt):
    fig, ax = plt.subplots(1,2, sharex=True,figsize=(15,4))
    sns.countplot(column_name,data=data_at,ax=ax[0])
    sns.countplot(column_name,data=data_bt,ax=ax[1])
    ax[0].set_title('A lot of Missing Rows (Above Threshold)')
    ax[1].set_title('Few or No Missing Rows (Below Threshold)')


# In[120]:


# Make plots for several column to compare two subsets
# Randomly select n number of columns and plot the comparison
for i in azdias.sample(6,axis=1).columns:
    make_countplot(i,at,bt)
# ORTSGR_KLS9,KKK,ANZ_PERSONEN,LP_FAMILIE_FEIN,LP_LEBENSPHASE_GROB,KBA13_ANZAHL_PKW


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# Majority of the rows have 0 missing values. A small proportion has between 1 to 8 values missing per row. There are two spike at 43 and 47 missing values per row. From the charts above it seems that these columns have similar distributions in both data set and are thus qualitatively similar
# 
# #####################################################################################################################
# 
# (Double-click this cell and replace this text with your own text, reporting your observations regarding missing data in rows. Are the data with lots of missing values are qualitatively different from data with few or no missing values?)
# 

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[17]:


# How many features are there of each data type?
# Display count of features for each data type
feat_info.groupby(by='type')[['attribute']].count()


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[18]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?

# Get all categorical features
cat_att=feat_info[feat_info['type']=='categorical']['attribute']

# Loop through all the categorical features to identify binary numeric, binary non-numeric and multi-level categories
bin_num=[]         #binary and numeric     (type1)
bin_non_num=[]     #binary and non_numeric (type2)
ml_cat=[]          #multi-level categories (type3)

for i in cat_att:
    try:
        # get number of unique values
        uv=bt[i].nunique()
        dt=bt[i].dtype
        
        # For binary variable with numeric value
        if uv==2 and dt!='object':
            bin_num.append(i)       #append
                      
        # For binary variable with non-numeric value
        elif uv==2 and dt=='object':
            bin_non_num.append(i)       #append
         
        # For multi-level categories    
        elif uv>2:
            ml_cat.append(i)       #append

        
    # Exception for columns which are already dropped from data
    except Exception as e:
        print('Note=>  Column does not exist: ', e)

print('\nBinary_Numeric:\n', bin_num , '\n\nBinary_Non_Numeric:\n' , bin_non_num , '\n\nMulti-Level:\n' , ml_cat)


# In[19]:


# Re-encode categorical variable(s) to be kept in the analysis.

# Import Libraries
from sklearn.preprocessing import LabelEncoder

# Initialize encoder
le=LabelEncoder()

# columns list to encode
col_list=bin_non_num+ml_cat

# Create a dataframe of encoded features
bt_encoded=bt.copy(deep=True)

# Encode all columns in column list
for cols in col_list:
    bt_encoded[cols]=le.fit_transform(bt[cols])

# Show first few rows
bt_encoded.head(5)


# #### Discussion 1.2.1: Re-Encode Categorical Features
# Subset of data with no missing values was used to perform the encoding. Some of the columns were dropped in the previous steps so we printed a short note for those columns. At this stage we did not drop any additional columns and encoded all the remaing columns, in summary a total of 14 features were re-encoded using label encoder in sklearn (13 multi-level categoricals and 1 binary non-numeric).
# 
# #####################################################################################################################
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding categorical features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[20]:


feat_info[feat_info['type']=='mixed']


# In[21]:


# Create a list of all mixed features
mixed_att=feat_info[feat_info['type']=='mixed']['attribute']

for i in mixed_att:
    # Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
    if i=='PRAEGENDE_JUGENDJAHRE':
        bt_encoded.loc[(bt_encoded[i]<=2)  & (bt_encoded[i]>=1) ,i+'_Decade']=1
        bt_encoded.loc[(bt_encoded[i]<=4)  & (bt_encoded[i]>=3),i+'_Decade']=2
        bt_encoded.loc[(bt_encoded[i]<=7)  & (bt_encoded[i]>=5),i+'_Decade']=3
        bt_encoded.loc[(bt_encoded[i]<=9)  & (bt_encoded[i]>=8),i+'_Decade']=4
        bt_encoded.loc[(bt_encoded[i]<=13) & (bt_encoded[i]>=10),i+'_Decade']=5       
        bt_encoded.loc[(bt_encoded[i]<=15) & (bt_encoded[i]>=14),i+'_Decade']=6
        
        bt_encoded[i+'_Movement']=1
        bt_encoded.loc[(bt_encoded[i]==1)  | (bt_encoded[i]==3)   |
                       (bt_encoded[i]==5)  | (bt_encoded[i]==8)   |
                       (bt_encoded[i]==10) | (bt_encoded[i]==12) |
                       (bt_encoded[i]==14) ,i+'_Movement']=2
        
    # Investigate "CAMEO_INTL_2015" and engineer two new variables.
    elif i=='CAMEO_INTL_2015':
        bt_encoded[i+'_Wealth']=bt_encoded[i].apply(lambda x:int(x[0]))
        bt_encoded[i+'_Life_Stage']=bt_encoded[i].apply(lambda x:int(x[1]))
        
    # Investigate "LP_LEBENSPHASE_GROB" and engineer a new variables (for low,avg,high income groups)         
    elif i=='LP_LEBENSPHASE_GROB':
        #low income
        bt_encoded[i+'_income']=1 
        
        #avg income
        bt_encoded.loc[(bt_encoded[i]==3)  | (bt_encoded[i]==5)   |
                       (bt_encoded[i]==8)  | (bt_encoded[i]==11)  |
                       (bt_encoded[i]==12) ,i+'_income']=2   
        #high income
        bt_encoded.loc[(bt_encoded[i]==9) ,i+'_income']=3        
        
    
    # Investigate "WOHNLAGE" and engineer a new variables (for Rural and Non Rural Flag)   
    elif i=='WOHNLAGE':
        bt_encoded[i+'_Rural']=1
        bt_encoded.loc[(bt_encoded[i]==7)  | (bt_encoded[i]==8),i+'_Rural']=2
          

    # Investigate "PLZ8_BAUMAX" and engineer a new variables (for family and business buildings) 
    elif i=='PLZ8_BAUMAX':
        bt_encoded[i+'_Bldg_Type']=1
        bt_encoded.loc[bt_encoded[i]==5,i+'_Bldg_Type']=2


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# There were a total of 7 columns with mixed attributes 
# 
# (['LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE', 'CAMEO_INTL_2015', 'KBA05_BAUMAX', 'PLZ8_BAUMAX']).
# 
# Out of these 'KBA05_BAUMAX' was already dropped earlier since it was missing a lot of data. We will also drop LP_LEBENSPHASE_FEIN and only keep LP_LEBENSPHASE_GROB, since the latter contaied same information but at a coarse scale. Below is summary of engineering steps applied to the columns:
#  - LP_LEBENSPHASE_GROB was converted in to a column feature to differentiate low,average and high income groups
#  - WOHNLAGE was converted to a new variable to differetiate between Rural and Non Rural buildings
#  - PLZ8_BAUMAX was converted to new variable to distinguish between family and business buildings
# 
# #####################################################################################################################
# 
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding mixed-value features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[22]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)


# In[23]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.

#Remove original mixed columns from final dataset if new columns have been engineered
for i in mixed_att:
    try:
        bt_encoded.drop(i,axis=1,inplace=True)        
    except Exception as e:
        print('Note=>  Column does not exist: ', e)


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[24]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
################################################################################################################   
    missing_data_stats=pd.DataFrame(df.isnull().sum(),columns=['Naturally_Missing_Values'])
    
    # Loop over all the rows/attributes in feat_info
    for i in range(len(feat_info)):
        att=feat_info['attribute'][i]                                                         # Get name of attribute
        codes=feat_info['missing_or_unknown'][i].replace('[',"").replace(']',"").split(',')   # Get values from string

        # Loop over all the individual values extratcted from '	missing_or_unknown' column
        for j in range(len(codes)):
            if codes[j]!='':                                       # skip empty values as nothing needs to be replaced
                try:
                    df.loc[df[att]==int(codes[j]),att] = np.nan       # replaces numeric values with NaNs
                except:
                    df.loc[df[att]==(codes[j]),att] = np.nan          # replaces string values with NaNs
    
    # Merge and plot the naturally missing and overall missing values
    missing_data_stats=pd.merge(missing_data_stats,pd.DataFrame(df.isnull().sum(),
                                columns=['Overall_Missing_Values']),left_index=True,right_index=True)
    
################################################################################################################   
    # Perform an assessment of how much missing data there is in each column of the dataset.
    missing_data_stats['Overall_Missing_Percentage']=(missing_data_stats['Overall_Missing_Values']/len(df))*100
    sorted_missing_data_stats=missing_data_stats.sort_values(by='Overall_Missing_Percentage',ascending=False)
    
    # Investigate patterns in the amount of missing data in each column.
    plt.figure(figsize=(15,30))
    sns.barplot(x='Overall_Missing_Percentage', y=sorted_missing_data_stats.index, 
                data=sorted_missing_data_stats);
    plt.title('Percentage of Missing Data by Attribute',size=20)
    
################################################################################################################
    # remove selected columns and rows, ...
    # Define criteria of outliers removal
    outlier=30    # Outlier if % missing more than

    #Get outlier columns and drop them from data
    outlier_columns=sorted_missing_data_stats[sorted_missing_data_stats['Overall_Missing_Percentage']>outlier].index
    for i in outlier_columns:
        df.drop(i,axis=1,inplace=True)

################################################################################################################
    # Sum all the missing values by rows
    nan_count_rows=df.isnull().sum(axis=1).sort_values(ascending=False)

    # Plot and visualize the distribution of missing rows
    plt.figure(figsize=(20,8))
    sns.countplot(nan_count_rows)

    plt.xlabel('Missing Values in each Row')
    plt.ylabel('Count of Rows')
    plt.title('Count of Rows vs Missing Values in each Row')
    
    
    # Define threshold value
    row_nan_threshold=0

    # Rows above threshold
    at=df.loc[nan_count_rows>row_nan_threshold]

    # Rows below threshold
    bt=df.loc[nan_count_rows<=row_nan_threshold]
    
################################################################################################################    
    # select, re-encode, and engineer column values.

    # Assess categorical variables: which are binary, which are multi-level, and
    # which one needs to be re-encoded?

    # Get all categorical features
    cat_att=feat_info[feat_info['type']=='categorical']['attribute']

    # Loop through all the categorical features to identify binary numeric, binary non-numeric and multi-level 
    #categories
    bin_num=[]         #binary and numeric     (type1)
    bin_non_num=[]     #binary and non_numeric (type2)
    ml_cat=[]          #multi-level categories (type3)

    for i in cat_att:
        try:
            # get number of unique values
            uv=bt[i].nunique()
            dt=bt[i].dtype

            # For binary variable with numeric value
            if uv==2 and dt!='object':
                bin_num.append(i)       #append

            # For binary variable with non-numeric value
            elif uv==2 and dt=='object':
                bin_non_num.append(i)       #append

            # For multi-level categories    
            elif uv>2:
                ml_cat.append(i)       #append


        # Exception for columns which are already dropped from data
        except Exception as e:
            print('Note=>  Column does not exist: ', e)

    print('\nBinary_Numeric:\n', bin_num , '\n\nBinary_Non_Numeric:\n' , bin_non_num , '\n\nMulti-Level:\n' , ml_cat)
    
################################################################################################################
    # Import Libraries
    from sklearn.preprocessing import LabelEncoder

    # Initialize encoder
    le=LabelEncoder()

    # columns list to encode
    col_list=bin_non_num+ml_cat

    # Create a dataframe of encoded features
    bt_encoded=bt.copy(deep=True)

    # Encode all columns in column list
    for cols in col_list:
        bt_encoded[cols]=le.fit_transform(bt[cols])

################################################################################################################
    # Create a list of all mixed features
    mixed_att=feat_info[feat_info['type']=='mixed']['attribute']

    for i in mixed_att:
        # Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
        if i=='PRAEGENDE_JUGENDJAHRE':
            bt_encoded.loc[(bt_encoded[i]<=2)  & (bt_encoded[i]>=1) ,i+'_Decade']=1
            bt_encoded.loc[(bt_encoded[i]<=4)  & (bt_encoded[i]>=3),i+'_Decade']=2
            bt_encoded.loc[(bt_encoded[i]<=7)  & (bt_encoded[i]>=5),i+'_Decade']=3
            bt_encoded.loc[(bt_encoded[i]<=9)  & (bt_encoded[i]>=8),i+'_Decade']=4
            bt_encoded.loc[(bt_encoded[i]<=13) & (bt_encoded[i]>=10),i+'_Decade']=5       
            bt_encoded.loc[(bt_encoded[i]<=15) & (bt_encoded[i]>=14),i+'_Decade']=6

            bt_encoded[i+'_Movement']=1
            bt_encoded.loc[(bt_encoded[i]==1)  | (bt_encoded[i]==3)   |
                           (bt_encoded[i]==5)  | (bt_encoded[i]==8)   |
                           (bt_encoded[i]==10) | (bt_encoded[i]==12) |
                           (bt_encoded[i]==14) ,i+'_Movement']=2

        # Investigate "CAMEO_INTL_2015" and engineer two new variables.
        elif i=='CAMEO_INTL_2015':
            bt_encoded[i+'_Wealth']=bt_encoded[i].apply(lambda x:int(x[0]))
            bt_encoded[i+'_Life_Stage']=bt_encoded[i].apply(lambda x:int(x[1]))

        # Investigate "LP_LEBENSPHASE_GROB" and engineer a new variables (for low,avg,high income groups)         
        elif i=='LP_LEBENSPHASE_GROB':
            #low income
            bt_encoded[i+'_income']=1 

            #avg income
            bt_encoded.loc[(bt_encoded[i]==3)  | (bt_encoded[i]==5)   |
                           (bt_encoded[i]==8)  | (bt_encoded[i]==11)  |
                           (bt_encoded[i]==12) ,i+'_income']=2   
            #high income
            bt_encoded.loc[(bt_encoded[i]==9) ,i+'_income']=3        


        # Investigate "WOHNLAGE" and engineer a new variables (for Rural and Non Rural Flag)   
        elif i=='WOHNLAGE':
            bt_encoded[i+'_Rural']=1
            bt_encoded.loc[(bt_encoded[i]==7)  | (bt_encoded[i]==8),i+'_Rural']=2


        # Investigate "PLZ8_BAUMAX" and engineer a new variables (for family and business buildings) 
        elif i=='PLZ8_BAUMAX':
            bt_encoded[i+'_Bldg_Type']=1
            bt_encoded.loc[bt_encoded[i]==5,i+'_Bldg_Type']=2
            
################################################################################################################
    # Return the cleaned dataframe.
    #Remove original mixed columns from final dataset if new columns have been engineered
    for i in mixed_att:
        try:
            bt_encoded.drop(i,axis=1,inplace=True)        
        except Exception as e:
            print('Note=>  Column does not exist: ', e)
    return bt_encoded


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[26]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
print('Total Null Values: ',sum(bt_encoded.isnull().sum(axis=0)))


# In[27]:


# Apply feature scaling to the general population demographics data.
# Import Libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Initialize Scaler
scaler= StandardScaler()

# Apply feature scaling
X = scaler.fit_transform(bt_encoded)


# ### Discussion 2.1: Apply Feature Scaling
# All the null values were already taken care of in previous steps, to ensure there are no null values we printed sum of total null values in dataframe to confirm that it is 0. Then we performed Standard Scaling for all the features in data and saved the result as X
# 
# #####################################################################################################################
# 
# 
# (Double-click this cell and replace this text with your own text, reporting your decisions regarding feature scaling.)

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[28]:


def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(25, 12))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


# In[29]:


# Import libraries
from sklearn.decomposition import PCA

# Apply PCA to the data using
pca = PCA(80)
X_pca = pca.fit_transform(X)


# In[30]:


# Investigate the variance accounted for by each principal component.
scree_plot(pca)


# In[31]:


# Re-apply PCA to the data while selecting for number of components to retain.
# Apply PCA to the data using
pca = PCA(10)
X_pca = pca.fit_transform(X)
X_pca.shape


# ### Discussion 2.2: Perform Dimensionality Reduction
# Initially PCA is performed using n_components = 80 to be able to see the variance accounted by all the components.
# Scree_plot function is then used to plot the variance accounted by each component as well as the total variance. Using the plot as reference it is decided to keep 20 principal components since they explain more than 70% of the total variance
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding dimensionality reduction. How many principal components / transformed features are you retaining for the next step of the analysis?)

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[32]:


# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.
# Function to print and plot feture weights of a given component
def pca_results(full_dataset, pca,component):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions
    
    # Filter to specified component
    ev=variance_ratios.iloc[[component-1]]
    nc=components.iloc[[component-1]]
    
    # Sort weight values before concatenating
    nc_sorted=nc.sort_values(nc.last_valid_index(), axis=1)
    
    # Concatenate
    cdf=pd.concat([ev, nc_sorted], axis = 1)

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (30,15))

    # Plot the feature weights for selected components
    nc_sorted.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights",size=70)
    ax.set_xlabel('Dimension'+str(component),size=70)  
    ax.set_title('Explained Variance = '+str(pca.explained_variance_ratio_[component-1]),size=70)
    
    ax.legend(loc=(1.0, 0.01), ncol=2)
    plt.tight_layout()

    # Return a concatenated DataFrame
    return cdf


# In[33]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
pca_results(bt_encoded,pca,1)


# In[34]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.
pca_results(bt_encoded,pca,2)


# In[35]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.
pca_results(bt_encoded,pca,3)


# ### Discussion 2.3: Interpret Principal Components
# Below is the summary of the highest negative and positive weights of the first 3 principal components:
# 
# **Principal Component 1 (PC1)**:
# 
# Negative weights:
# - Movement patterns
# - Social status
# 
# Positive weights:
# - Wealth / Life Stage Typology
# 
# **Principal Component 2 (PC2)**:
# 
# Negative weights:
# - Religious personality
# - Decade of persons youth
# - Financial topology-money saver
# 
# Positive weights:
# - Age
# - Financial topology - be prepared
# - Event Oriented personality
# - Sensual-minded personality
# 
# **Principal Component 3 (PC3)**:
# 
# Negative weights:
# - Gender
# - Combative attitude personality
# - Dominant minded personality
# 
# Positive weights:
# - Dreamful personality
# - Social minded personality
# 
# From an overall perspective it seems that personality typology is a very important feature when it comes to principal components(PC). Majority of the highest and lowest values in PC-2 & PC-3 relate somehow to different personality types.
# 
# Also intersting to note are the:
# - Negative correlation between the religious and sensual minded personality in PC-2, which contrasts conservative and liberal personalities.
# - Positive correlation between Gender and combative/dominant personality, which could suggest that one of the gender (male/female) are generally more dominant minded.
# 
# #####################################################################################################################
# 
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[35]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.

#Import library
from sklearn.cluster import KMeans

# Instantiate fit, predict and score function
def kmean_score(data,number_of_clusters):
    kmeans = KMeans(number_of_clusters)
    km_model = kmeans.fit(data)
    km_labels = km_model.predict(data)
    km_model_score=abs(km_model.score(data))
    
    return km_model_score

# Over a number of different cluster counts...
# run k-means clustering on the data and...
# compute the average within-cluster distances.
    
# Loop to get all scores
num_clusters=list(range(1,11))
score_appended=[]
for i in num_clusters:
    score_appended.append(kmean_score(X_pca,i))
    
plt.plot(num_clusters,score_appended, linestyle='--', marker='o', color='green')


# In[36]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
# from sklearn.cluster import KMeans
kmeans=KMeans(4)
km_labels=kmeans.fit_predict(X_pca)


# ### Discussion 3.1: Apply Clustering to General Population
# 
# After defining a function which takes in the data and number of clusters, we run it through a number scenarios of incremental cluster numbers upto 10 clusters and then plot the model score (y-axis) against number of clusters (x-axis). It is hard to decide number of clusters to segment the population using the elbow method, however a value of 4 seems reasonalble based on the elbow plot
# 
# #####################################################################################################################
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[37]:


# Load in the customer demographics data.
customers.head(2)


# In[38]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.

# 1) Preprocessing
customers_cleaned=clean_data(customers)
customers_cleaned.head(2)

# 2) Feature Transformation
X_cust=scaler.fit_transform(customers_cleaned)
X_cust.shape

# 3) PCA
X_cust_pca=pca.fit_transform(X_cust)
X_cust_pca.shape

# 4) Clustering
km_cust_labels=kmeans.fit_predict(X_cust_pca)


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[72]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.

# CUSTOMER DATA
# Concatenate Cluster ID with cleaned table (after reseting its index) 
kmcl = pd.DataFrame(km_cust_labels, columns = ['ClusterID'])
cd=pd.concat([customers_cleaned.reset_index(drop=True), kmcl],axis=1)

# GENERAL DATA
# Concatenate Cluster ID with cleaned table (after reseting its index) 
kmgl = pd.DataFrame(km_labels, columns = ['ClusterID'])
gd=pd.concat([bt_encoded.reset_index(drop=True), kmgl],axis=1)


# In[96]:


# Make Countplot for comparison
fig, ax = plt.subplots(1,2, sharex=True,figsize=(15,4))
sns.countplot('ClusterID',data=cd, ax=ax[0])
sns.countplot('ClusterID',data=gd, ax=ax[1])

ax[0].set_title('Customer Data')
ax[1].set_title('General Data')


# In[ ]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?


# **Cluster 2 is overrepresented:**
# 
# People in this cluster genrally:
# - Have Low movement
# - Are consumption oriented middle class with average income
# - Have low propensity to save money
# - Have average to low religious affinity
# - Very high sensual-minded affinity
# - Had youth years in 80-90s
# - Not very socially or family minded
# 
# Below are the plots of the key PCA features for All data vs Data in Cluster 2

# In[116]:


# Make a list of key features from PCA analysis in previous exercise
key_feats=['MOBI_REGIO','LP_STATUS_FEIN', 'LP_STATUS_GROB',
           'CAMEO_INTL_2015_Wealth', 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015',
           'SEMIO_REL', 'PRAEGENDE_JUGENDJAHRE_Decade', 'FINANZ_SPARER',
           'SEMIO_LUST','SEMIO_ERL', 'FINANZ_VORSORGER', 'ALTERSKATEGORIE_GROB',
           'ANREDE_KZ', 'SEMIO_KAEM', 'SEMIO_DOM',
           'SEMIO_FAM', 'SEMIO_SOZ','SEMIO_VERT']

# Loop through the list of features and make plots for overall data and data in Cluster 2
for i in key_feats:   
    fig, ax = plt.subplots(1,2, sharex=True,figsize=(15,4))
    
    sns.countplot(i,data=gd, ax=ax[0])
    sns.countplot(i,data=gd[gd['ClusterID']==2], ax=ax[1])
    
    ax[0].set_title('All Data')
    ax[1].set_title('Cluster 2 Data')


# In[117]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?


# **Cluster 3 is underrepresented:**
# 
# People in this cluster genrally:
# - Are the top earners and have the highest social status with wealthy and prosperous households
# - Have high to average religious affinity
# - Had youth years in 60/70s
# - Are high monery saver
# - Have low sensual-minded affinity
# 
# Below are the plots of the key PCA features for All data vs Data in Cluster 3

# In[118]:


# Make a list of key features from PCA analysis in previous exercise
key_feats=['MOBI_REGIO','LP_STATUS_FEIN', 'LP_STATUS_GROB',
           'CAMEO_INTL_2015_Wealth', 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015',
           'SEMIO_REL', 'PRAEGENDE_JUGENDJAHRE_Decade', 'FINANZ_SPARER',
           'SEMIO_LUST','SEMIO_ERL', 'FINANZ_VORSORGER', 'ALTERSKATEGORIE_GROB',
           'ANREDE_KZ', 'SEMIO_KAEM', 'SEMIO_DOM',
           'SEMIO_FAM', 'SEMIO_SOZ','SEMIO_VERT']

# Loop through the list of features and make plots for overall data and data in Cluster 3
for i in key_feats:   
    fig, ax = plt.subplots(1,2, sharex=True,figsize=(15,4))
    
    sns.countplot(i,data=gd, ax=ax[0])
    sns.countplot(i,data=gd[gd['ClusterID']==3], ax=ax[1])
    
    ax[0].set_title('All Data')
    ax[1].set_title('Cluster 3 Data')


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# Yes, we were able to identify segments popular and unpopular with the mail-order company. Cluster 2 which was overrepresented is the popular segment with the company and should be its target audience, whereas cluster 3 is underrepresented and is the unpopular segment.
# 
# Popular segment generally comprises of consumption oriented middle class, brought up in 80/90s whereas unpopular segment comprises of top earners from wealthy households brought up in 60/70s and are high money-savers
# 
# #####################################################################################################################
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




