#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import os
main_start = time.time()

print('time at stage {:.6}'.format(time.time() - main_start))

from itertools import chain
import math
import difflib
import builtins
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from shutil import move
import re
import argparse
import random
import traceback
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
# end = time.time()

#pyspark sql functions and window import
from pyspark.sql import functions as F
from pyspark.sql import Window as W

# pyspart sql functions import
from pyspark.sql.functions import * # everything, then why above?

# pyspark sql types import
from pyspark.sql.types import * # we are importing everything, why above? Krishana

from functools import reduce  # For Python 3.x
# from pyspark.sql import DataFrame

# from azureml.widgets import RunDetails
import azureml.core
from pyspark.context import SparkContext
from pyspark.sql import SparkSession, DataFrame
from azureml.core.compute import ComputeTarget, AmlCompute, ComputeInstance
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace, Datastore, Dataset, Keyvault, Run
from azureml.core import ScriptRunConfig, Environment, Experiment
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import MpiConfiguration
from azureml.core.workspace import Workspace
from azureml.core.authentication import AzureCliAuthentication 

from azureml.data.datapath import DataPath

from azureml.telemetry import set_diagnostics_collection

# from azureml.contrib.dataset import FileHandlingOption
# from pyspark.sql import DataFrame
#from pyspark.sql.functions import  max as max_
# master_check = os.getenv('AZUREML_CONDA_ENVIRONMENT_PATH')
# os.environ['PYSPARK_DRIVER_PYTHON'] = os.getenv('AZUREML_CONDA_ENVIRONMENT_PATH')+'/bin/python'
# os.environ['PYSPARK_PYTHON'] = os.getenv('AZUREML_CONDA_ENVIRONMENT_PATH')+'/bin/python'
os.environ['RSLEX_DIRECT_VOLUME_MOUNT'] = 'true'

cli_auth = AzureCliAuthentication()
ws = Workspace.from_config(auth=cli_auth)

#Initiate Spark
os.environ['PYSPARK_PYTHON'] = '/anaconda/envs/azureml_py38/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/anaconda/envs/azureml_py38/bin/python'

def get_compute_stats():
    baseplace = os.popen('pwd').read()
    vm_name = baseplace.split('/')[8]
    compute = ComputeInstance(ws, vm_name)
    vms = AmlCompute.supported_vmsizes(ws, location=None)
    vmsize = compute.get_status().serialize()['vmSize'].lower()
    vms_df = pd.DataFrame(vms)
    vm_cores = vms_df.loc[vms_df['name'].str.lower()==vmsize, 'vCPUs'].item()
    vm_ram = int(vms_df.loc[vms_df['name'].str.lower()==vmsize, 'memoryGB'].item())
                                       
    return vm_name, vm_cores, vm_ram

vm_name, vm_cores, vm_ram = get_compute_stats()

# Enter RAM of current compute
driver_mem = int(vm_ram * 0.9)
partitions = vm_cores * 3

print("Connect to Spark")
# Start Spark session
spark = SparkSession.builder.appName("SparkApp")     .master('local[*]')     .config('spark.driver.memory', f'{driver_mem}g')     .config('spark.sql.shuffle.partitions', f'{partitions}')     .config("spark.driver.maxResultSize", "0")     .config("spark.executor.memory","8g")     .config("spark.executor.cores","1")     .config("spark.python.worker.memory","8g")     .getOrCreate()

print("spark con", spark)

sc = spark.sparkContext

print(spark.sparkContext.getConf().getAll())


# ##### Reading names dataset and cleaning it

# In[ ]:


# A file with thousands of people's names (Anush compiled from internet)
# these three lines will be useful 
# when we reference the data from other user's workspace
users= "/mnt/batch/tasks/shared/LS_root/mounts/clusters/vacCPU-gitv3/code/Users/"
data = "Sruti.Kanthan/rockies-datascience/NLP/AzureML/TIU_PII_NER/data/"
data_path = os.path.join(users, data)

# reading data in spark data frame, delimiter command, and header true
names=spark.read.option('delimiter', ',').option('header', 'true').csv("data/refined_namelist.csv") 
# Formatting the names so that there will not be any duplicate, convert spark df to list, and they are all capitalized
distinct_names = names.drop_duplicates(['name'])
name_title  = udf(lambda x: str(x).title(), StringType())
distinct_names = distinct_names.withColumn("name", name_title('name'))

# convert names to list
distinct_names = distinct_names.rdd.map(lambda x: x.name).collect()


# ##### List of States

# In[ ]:


# adding header to the spark dataframe
schema = StructType([    StructField("state", StringType(), True)])
# reading the data to create spark df
states=spark.read.option('delimiter', ',').schema(schema).option('header', 'False').csv("data/states.txt") 

#get distinct states 
states_df=states.select('state').distinct()

# We also need to add US territories
us_territories=["Guam", "Puerto Rico", "Virgin Islands", "American Samoa"]
df_us_territories = spark.createDataFrame(us_territories, StringType()).toDF('state')

#merged two df US states and UST-United States territories. 
ust_n_states_df=states_df.union(df_us_territories)

# filter the state name which has more than 2 characters (done at later stage now)
# filter_st_length=ust_n_states_df.filter(length('state')>2)

# If state abbreviation, keep uppercase; if not, then lowercase and title - each word's first letter capital, reused the name_title function
total_ust_n_states_df = ust_n_states_df.withColumn("state", when((length(col('state')) > 2) == 'true', name_title('state')).otherwise(ust_n_states_df.state))

# State Abbreviations
states_abb = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", 
"KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND",
 "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "GU", "PR", "VI", "AS"]


# ##### Zip codes

# In[ ]:


# Can't use regex to identify 5 digit sequences because some are medical codes
# reading the data to create spark df and this dataset has the header available
zip_codes = spark.read.option('delimiter', ',').option('header', 'True').csv("data/us_zips.csv") 
#select only distinct zip codes
dist_zip_df = zip_codes.select('ZIP').distinct()
# Renaming "ZIP" col to "zipcode" to avoid confusion with builtin zip method
dist_zip_df = dist_zip_df.withColumnRenamed("ZIP", "zipcode")
# list of distinct zip codes
zips_list = dist_zip_df.rdd.map(lambda x: x.zipcode).collect()


# ##### Cities and Counties

# In[ ]:



# Read the file with US city names and their corresponding states and counties
city_county_df=spark.read.option('delimiter', '|').option('header', 'true').csv('data/us_cities_states_counties.csv')

# Renaming the column names of the spark DF
new_cols = ["City", "State_code", "State_full", "County", "City alias"]
city_county_df = city_county_df.toDF(*new_cols)
city_county_df = city_county_df[city_county_df['State_code'].isin(states_abb)]
city_county_df = city_county_df[city_county_df['City'] != '88'] # Unnecessary city code
#city_county_df = city_county_df[['City', 'State_code', 'State_full']]

# convert Row list or Pyspark Column to list
states_list = total_ust_n_states_df.rdd.map(lambda x: x[0]).collect()

# Making sure information is limited to US states only
city_county_df = city_county_df.filter(~col("City").isin(states_list))

# File with additional US cities information-first file didn't include cities from US territories
additional_cts_df=spark.read.option('delimiter', ',').option('header', 'True').option("encoding", "latin1").csv('data/us_cities2.csv')
#renaming the columns and followed the same code as city_county_df mentioned above
new_cols = ["Id", "State_code","State_full","City","County","LATITUDE","LONGITUDE"]
additional_cts_df = additional_cts_df.toDF(*new_cols)
additional_cts_df = additional_cts_df.filter(~col("City").isin(states_list))
additional_cts_df = additional_cts_df[['City', 'State_code', 'State_full']]
additional_cts_df = additional_cts_df[additional_cts_df['State_code'].isin(states_abb)]

# Combining city file information and deleting duplicate information
keep = additional_cts_df.select("City").subtract(city_county_df.select("City"))
additional_cts_df = additional_cts_df[additional_cts_df['City'].isin(keep['City'])] # subsetting more_cities to only have cities from "keep"
#concatinating two dataframes, and dropping the duplicates
# add columuns in city_county_df from additional_cts_df
for column in [column for column in additional_cts_df.columns               if column not in city_county_df.columns]:
    city_county_df = city_county_df.withColumn(column, lit(None))
    
# add columuns in additional_cts_df from city_county_df
for column in [column for column in city_county_df.columns                if column not in additional_cts_df.columns]:
    additional_cts_df = additional_cts_df.withColumn(column, lit(None))

# Union all in both dataframes
city_county_df = additional_cts_df.unionAll(city_county_df)
city_county_df = city_county_df.dropDuplicates()


# Final dataframe with city, state abbreviation, and full state name information
city_state_df = city_county_df.dropDuplicates(subset=['City', 'State_code', 'State_full']) # dropping County duplicates (by state)
city_state_df = city_county_df[['City', 'State_code', 'State_full']]

# dropping City duplicates (by state)
county_state_df = city_county_df.dropDuplicates(subset=['County', 'State_code', 'State_full']) 
county_state_df = city_county_df[['County', 'State_code', 'State_full']]

#dropping null columns from the both dataframes
city_state_df = city_state_df.na.drop()
county_state_df = county_state_df.na.drop()

#group the dataframe by City
city_state_df = city_state_df.groupBy("City").agg(array_distinct(collect_list(struct("State_full", "State_code"))).alias("state_full_n_code"))

#convert sparkdf to Dictionary 
city_state_dict = city_state_df.select([col for col in city_state_df.columns]).rdd.map(lambda x: x.asDict()).collect()

# Creating a dictionary with city name as key and all states where this city name exists as value.
# Ex: {'City A': ['AL', 'Alabama', 'NJ', 'New Jersey']}. Added this because city names were being 
# mistagged as names and vice versa; much more accurate when searching for corresponding state in the sentence
ct_states_dict = {}
for x in city_state_dict: 
    if isinstance(x, dict):
        key = x['City']
        value = []
        for state in x['state_full_n_code']:
            value.append(state.State_full)
            value.append(state.State_code)
        ct_states_dict[key] = value
        
ct_states_dict = {key: value for key, value in ct_states_dict.items() if value}

#group the dataframe by County, similar as above
county_state_df = county_state_df.groupBy("County").agg(array_distinct(collect_list(struct("State_full", "State_code"))).alias("state_full_n_code"))

#convert sparkdf to Dictionary 
county_states_dict = county_state_df.select([col for col in county_state_df.columns]).rdd.map(lambda x: x.asDict()).collect()

# Creating a dictionary with county name as key and all states where this city name exists as value.
# Ex: {'County A': ['AL', 'Alabama', 'NJ', 'New Jersey']}. Added this because county names were being 
# mistagged as names and vice versa; much more accurate when searching for corresponding state in the sentence
county_sts_dict = {}
for x in county_states_dict: 
    if isinstance(x, dict):
        key = x['County']
        value = []
        for state in x['state_full_n_code']:
            value.append(state.State_full)
            value.append(state.State_code)
        county_sts_dict[key] = value
        
county_sts_dict = {key: value for key, value in county_sts_dict.items() if value}

# Specifying that states_list should only contain unabbreviated states; we have a separate list, states_abb, for abbreviated states
states_list = [state for state in states_list if len(state) > 2]


# - can add the key reference in sub mult (from 4/4 commit) back later; it's adding too much time right now

# In[ ]:


us_cities_all = list(set(ct_states_dict.keys()))

exclude = ['Enterprise', 'Call', 'Veteran', 'Clear', 'Section', 'Check', 'Standard', 'Home', 'Telephone', 'Blue', 
'Points', 'Florida','Center', 'Only', 'Post', 'Advance', 'Onset', 'Light', 'Range', 'Transfer',
 'Foley', 'Hygiene', 'Given', 'Normal', 'Start', 'Comfort','Falls', 'Likely', 'Home', 'Day', 'Money',
  'Calcium', 'Felt', 'Only', 'Colon', 'Mountain', 'Tell', 'Force', 'Standard', 'Gateway', 'Vista', 'Morse', 
  'Monitor', 'Muse', 'Drain', 'Agency', 'Street', 'Success', 'Cope', 'Wells', 'Effort', 'Issue', 'Hurt', 'Power',
   'Hill', 'House', 'Cataract', 'Deputy', 'Accident', 'Oral', 'Index', 'Reading', 'United', 'Early',
    'History', 'Diabetes', 'Tobacco','Use', 'SNOMED', 'of', 'Exposure', 'Diverticulosis', 'Happy', 'May', 'Green',
     'VA', 'Iron', 'Supply', 'Wake', 'Liberal', 'Saline', 'Intercourse','Braden', 'Paneled', 'Mode', 'Max', 'Bath', 
     'Brush', 'White', 'Vera', 'University', 'Man', 'Levels', 'paneled', 'readings', 'Rescue', 'Orient','Spray', 'Ord', 
     'Institute', 'Universal', 'Barrett', 'Sun', 'State', 'Park', 'State Park', 'State park', 'Vial', 'Ideal']

# City list minus words from exclude list
rem = [city for city in us_cities_all if city not in exclude]
#exclude.extend(med_exclude)

def sub_mult_regex(text, keys, tag_type, word_bounds):
    '''
    Used in situations where we want to mask multiple patterns at the same time; mass replacement with regex

    Parameters:
        text: string (eg, TIU note)
        keys: a list of words to be replaced by the regex
        tag_type: string you want the words to be replaced with. This will either be PersonName or PAddress
        word_bound: bool, True if replacing cities or NON-ABBREVIATED states (eg., "New Jersey", not "NJ"), False if replacing anything else

    Creates a replacement dictionary of keys and values 
    (values are the length of the key, preserving formatting).
    Eg., {68 Oak St., PAddress PAddress PAddress.,}. Note how punctuation are preserved

    Returns text with relevant text masked, and a list containing the words that were masked

    Sample call: sub_mult_regex("The patient lives in New Jersey; and the Dr is from North Dakota", ['New Jersey', 'North Dakota'],
    'PAddress', city_or_state=True)

    Sample return output: ('The patient lives in PAddress PAddress; and the Dr is from PAddress PAddress',
    ['New', 'Jersey', 'North', 'Dakota'])
    '''
    # List of words to exclude; ie, don't tag them
    excludes = ['Intern', 'Nurse', 'Doctor', 'Physician', 'Surgeon']

    if word_bounds:
        # If we're masking a city or state, handle word boundaries
        keys = [r"\b"+key+r"\b" for key in keys if key in text or key.upper() in text] # add word boundaries for each key in list
        add_vals = []
        for val in keys:
            # Create dictionary of city/state word:PAddress by splitting the city on the '\\b' char that remains and then adding one tag per word
            # Ex: '\\bDeer Island\\b' --> split('\\b') --> ['', 'Deer Island', ''] --> ''.join --> {Deer Island : PAddress PAddress}
            add_vals.append(re.sub(r'\w{1,100}', tag_type, ''.join(val.split('\\b')))) # To preserve the precise punctuation, etc. formatting of the keys, only replacing word matches with tags
        add_vals = [re.sub(r'\\b', "", val) for val in add_vals]

        # Addressing cases where the city or state name might actually be a person's name
        # If the word after the city or state is in total_names, then don't consider the city or state word to be an address 
        # (eg., Virginia Wilson - Virginia not counted since Wilson is name, normally it would be counted as a state
        for i, key in enumerate(keys):
            # Identifying key in text, if it's present. 
            findall = re.findall(rf'{key}\s[A-Z][a-z]+', text, re.IGNORECASE)
            # Identifying the word after the key *may need to add index exception if key is the last word in a note*
            next_word = re.sub(r'[^\w\s]','', ' '.join(findall).split(' ')[-1])
            # If the word after the key is in the total_names list, then don't mask it as PAddress
            if next_word.lower().capitalize() in distinct_names:
                print("Possible name after city or state:", next_word)
                keys = [m for m in keys if m != key]


    elif not word_bounds:
        # If we're not masking a city or unabbreviated state, we don't do the word boundary step
        # First, look for instances with exclude words - then remove the exclude words, keeping the rest of the name.
        # This is really only relevant when masking names (PersonName). Sometimes notes will contain information like 
        # "Intern, Laura S" and the name tagger would mistake "Intern" for a name, so we cut it off here 
        for i,key in enumerate(keys):
            for j,e in enumerate(excludes):
                if e in key or e.upper() in key:
                    # Remove the exclude word and everything until the next word that shows up (eg., Intern, Laura S. would remove 'Intern,')
                    keys[i] = re.sub(e+r"[.,!-()\/\\@#?'](.*?)[\s]", '', key)

        add_vals = []
        # Now that we have our list of keys, we create the corresponding values. Again, format is eg., {Chu, Sarah F: PersonName, PersonName PersonName}
        for val in keys:
            add_vals.append(re.sub(r'\w{1,100}', tag_type, val)) # To preserve the precise punctuation, etc. formatting of the keys, only replacing word matches with tags

    keys = [re.sub(r'\)|\(','', word) for word in keys]
    # Zipping keys and values together as dictionary
    add_dict = dict(zip(keys, add_vals))

    if len(add_dict) > 0:
        print("Zipped keys and values:", add_dict)

    # Compiling the keys together (regex)
    add_subs = re.compile("|".join("("+key+")" for key in add_dict), re.IGNORECASE)

    # This is where the multiple substitutions are happening
    # Taken from: https://stackoverflow.com/questions/66270091/multiple-regex-substitutions-using-a-dict-with-regex-expressions-as-keys
    group_index = 1
    indexed_subs = {}
    for target, sub in add_dict.items():
        #print("target:", target, "sub:", sub)
        indexed_subs[group_index] = sub
        group_index += re.compile(target).groups + 1
    if len(indexed_subs) > 0:
        text_sub = re.sub(add_subs, lambda match: indexed_subs[match.lastindex], text) # text_sub is masked text
    else:
        text_sub = text # Not all texts have names, so text_sub would've been NoneType and broken funct otherwise

    # Information on what words were changed pre and post masking (eg., would return 'ANN ARBOR' if that city was masked here)
    case_a = text # Original input text
    case_b = text_sub # Modified text

    diff_list = [li for li in difflib.ndiff(case_a.split(), case_b.split()) if li[0] != ' ']
    diff_list = [re.sub(r'[-,+]', "", term.strip()) for term in diff_list if (term[0] == '-' or term[0] == "+") and 'PAddress' not in term]
    diff_list = [word.strip() for word in diff_list]

    # Returning both the modified text and a list recording what was masked
    return text_sub, diff_list
    
def mask_multiword_cities(text_string):
    multi_word_cities = list(set([city for city in us_cities_all if '-' in city or len(city.split(' ')) > 1 and len(city) > 3 and "Mc" not in city and    "State " not in city and city != 'Mary D' and city != 'Cut Off' and city != 'D Lo']))
    return sub_mult_regex(text_string, multi_word_cities, "PAddress", word_bounds=True)

def single_city_abb_state(text):
    '''
    Replacing the original single-city approach that looked ahead of the city (token[i]) to
    token[i+1] and token[i+2] concatenated them to search for a state or clinic as context before
    masking; that approach was prone to issues because window may have been too small and didn't account
    for chunks of puncutation. Eg., "Lincoln , NE" vs "Lincoln \r\r , NE" (which wouldn't have been caught).
    Following approach uses a wider window of context for each city occurence.

    Input: Text (eg., "Dallas Memorial Hospital and Lincoln , NE")
    Output: Text with single cities and abbreviated states associated with them redacted
            (eg,: "PAddress Memorial Hospital and PAddress , PAddress")

    Logic: Cities are easily confused with person's names, as many cities are named after people. So, we
        need a way to consider context before masking as a city, and we can't mass-mask a single city
        throughout the text because context will vary at each occurence. Context here is found by 
        turning the text into chunks at each city instance.
        Single-word city is more likely to be a city, and not a trivial word or person's name,
        if there is a state or clinic-related term (from clinic_list) after it. Ex: Lincoln , NE 
        vs. Lincoln Smith, vs. Lincoln Memorial Hospital. 

    Function narrows down possible cities in text (capitalized words that show up in the set cities list). Splits
    the text on these cities (eg., earlier example becomes the chunks: ["Dallas", " Memorial Hospital and ", "Lincoln", " NE"]). 
    Looping through to see if city tokens are followed by a chunk that contains either a state name or clinic-like term 
    as its first word (using regex to isolate first word within the chunk); if so, (like the Dallas example), mask the city 
    (and the state ahead of it if relevant). Then, join the tokens back together as a string and return it.
    Also checks if "PAddress" follows the city (if so, masks as city only if the word before the city isn't in the names list).
    Also checks if there are unabbreviated states preceded by "PAddress", masks those as states.
    If conditions are not met, then the original string is returned
    '''
    # Isolating capitalized words in the text (city contenders)
    cap_words = [word for word in text.split(' ') if len(word) > 0 and word[0].isupper()]
    # Comparing cap_words words against city list, keeping those that match
    poss_cities = [re.sub(r'[^\w\s]', '', ele) for ele in cap_words if re.sub(r'[^\w\s]', '', ele).strip().lower().capitalize() in rem]
    # Adding word boundaries around the cities
    bounds = [r'\b'+city+r'\b' for city in poss_cities]
    # Initializing these lists for later 
    csz = []
    poss_state = []
    poss_clinic = []
    first_word_after_city = []
    # If the poss_cities list is 0, then return the text as is. Otherwise, continue
    #print("len poss cities:", 0)
    if len(poss_cities) > 0:
        # Split the text at each city instance (case insensitive)
        splitty = re.split("("+'|'.join(bounds)+")", text, flags=re.I)
        print(splitty)
        # Iterate through each chunk
        for i,chunk in enumerate(splitty):
            # If the chunk is a city name - staying in this level of the loop from now on since we're only interested in evaluating possible cities
            if any(city in chunk for city in poss_cities):
                print("possible city:", chunk)
                # Index restriction not needed since empty string always follows city when split
                first_word_after_city = re.findall(r'[a-zA-Z]+', splitty[i+1], re.IGNORECASE) # List of word-strings in the chunk after present city (excludes numbers, punctuation)
                if len(first_word_after_city) == 0: # This happened occasionally where there was actually no word in the next chunk
                    try:
                        first_word_after_city = re.findall(r'[a-zA-Z]+', splitty[i+2], re.IGNORECASE)
                    except:
                        print(first_word_after_city)
                if len(first_word_after_city) > 0:
                    if re.sub(r'[^\w\s]','', chunk).strip().lower().capitalize() in ct_states_dict.keys(): # Check if the city is in the city-state dict keys
                        poss_state = [ele for ele in first_word_after_city[:1] if ele.strip().lower().capitalize() in ct_states_dict[re.sub(r'[^\w\s]','', chunk).strip().lower().capitalize()]                        or ele.strip().upper() in ct_states_dict[re.sub(r'[^\w\s]','', chunk).strip().lower().capitalize()]] # Identify states by dict lookup, store in list (only looking at first word after city)
                        csz.extend(poss_state)
                        print("possible city:", chunk, "poss state:", poss_state)
                    # Checking for clinic-related words after the city word and creating a list of any results
                    poss_clinic = [re.sub(r'[^\w\s]','', ele) for ele in first_word_after_city[:1] if ele.upper() in clinic_list or ele.strip().lower().capitalize() in clinic_list]


                    # For cases where word after city is state (then mask city and state)
                    if poss_state != []: # If there was at least one state term found in the chunk after the city
                        print("this is the current chunk; got to poss_state:", repr(chunk), "\n")
                        csz.extend([re.sub(r'[^\w\s]','', chunk)]) # Add the city name to the csz list
                        splitty[i]= re.sub(r"\b" + re.sub(r'[^\w\s]','', chunk) + r"\b", 'PAddress', chunk) # Mask the city
                        try:
                            splitty[i+1] = re.sub(r"\b" + re.sub(r'[^\w\s]','', poss_state[0]) + r"\b", 'PAddress', splitty[i+1]) # Mask the state since it's the word after the city (try/except in case of index error)
                        except:
                            pass
                    # For cases where the word after city is a clinic-like term (then mask just the city)
                    if poss_clinic != [] or "PAddress" in first_word_after_city[0]: # If there was at least one clinic term in the chunk ahead of the city, or the word after the city is "PAddress"
                        print("got to poss clinic/PAddress", poss_clinic, first_word_after_city[0])
                        poss_name = "" # initializing poss_name in case loop doesn't give result
                        for c in reversed(splitty[:i]):
                            if len(re.findall(r'[a-zA-Z]+', c)) > 0: # Finding the first word
                                poss_name = c
                                poss_name = poss_name.strip()
                                break
                        print("poss name for", chunk, "before clinic:", poss_name)
                        try:
                            if poss_name[0].isupper() and (poss_name.lower().capitalize() in distinct_names or poss_name.upper() in distinct_names or len(poss_name)==1): # Check if the word right before the city is a person's name/initial
                                pass # if so, pass (eg., "Michael Henry Butler , \r \n PAddress"); 'Butler' is a city and 'PAddress' is technically the first word after it, but still a person's name
                            elif poss_name not in distinct_names: # If word after city is not a name, then mask city 
                                csz.extend([re.sub(r'[^\w\s]','', chunk)]) # Add the city name to the csz list
                                splitty[i]= re.sub(r"\b" + re.sub(r'[^\w\s]','', chunk) + r"\b", 'PAddress', chunk) # Mask the city
                        except:
                            pass
                    
                else:
                    print("got to else for", chunk)
                    pass
            #print("csz", csz)
            # Checking for cases like PAddress , AL ; want to mask 'AL' since it's an abbreviated state. Useful also when city name = a state name. 
            if any(state in chunk for state in states_abb):
                #print('state abb chunk:', chunk)
                poss_add = '' # initializing poss_add in case loop doesn't give result
                for c in reversed(splitty[:i]):
                    if len(re.findall(r'[a-zA-Z]+', c)) > 0: # Finding the first word
                        poss_add = c
                        poss_add = poss_add.strip()
                        break
                print("poss_add:", poss_add)

                if r'\b' + re.escape(poss_add) + r'\b' in "PAddress":
                    print("poss_add made it to regex:", poss_add)
                    splitty[i]= re.sub(r"\b" + re.sub(r'[^\w\s]','', poss_add) + r"\b", 'PAddress', chunk) # Mask the city

            #print(splitty)
                            
                
                
        assert len(''.join(splitty).split(' ')) == len(text.split(' ')) # Length of the masked string split on spaces must equal length of original text split on spaces

        return ''.join(splitty), csz
    else:
        return text, csz


# - Relatively rare scenario where the Dr's last name is also a city in Maryland, because word before MD suffix

# In[ ]:


nameTag_test(tag_address_test("The doctors are LELEY , ANJALI MD and PATRICIA S. WAKEFILED , RN \r and Sarah Fawcett Waters MD and Clark , DeAnne Mills MD"))



# List of words that indicate a hospital/clinic (helps give context to whether a word is a city)

clinic_list = ['VA', 'CBOC', 'Hospital', 'Center', 'Clinic', 'HOSPITAL', 'CLINIC', 'CENTER', 'CLC', 'VISN', 'County', "COUNTY", 
'MEDICAL', 'Medical', 'HEALTHCARE', 'Healthcare', 'Health', 'HEALTH', 'Division', "DIVISION", 'System', "SYSTEM", 'Care', 'CARE',
'Outpatient', 'OUTPATIENT', 'Primary', 'PRIMARY', 'Memorial', 'MEMORIAL']

def tag_address(text):
    '''
    Address identifier/masker

    1. Streets are tagged using regex, masked in text
    2. Multi-word cities are masked
    3. Unabbreviated states are masked
    4. Single-word cities masked (and their corresponding abbreviated states); also checks for clinics with city names attached
    5. Any lingering abbreviated states masked
    6. Zip codes masked

    Eg: "Patient lives at 123 Oak Terrace , West Orange, NJ, 07052" -> "Patient lives at PAddress PAddress PAddress , PAddress PAddress, PAddress, PAddress"
    '''
    try: # Try and except for full address function, will skip to next note if error 
        # Regex for capturing street addresses (eg 238 Walker Ave). Consolidating information and narrowing it down to streets with capitalized street names
        street_regex = r"(\d{1,5}\s?\w?.?[\w\s]{1,16}(?:street|st|avenue|ave|road|rd|terrace|lane|ln|highway|hwy|square|sq|trail|trl|way|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd|place|pl)[.,]?\W?(?=\s|$))|(\s(APT|Apt)\s[a-zA-Z0-9_.-]*)"
        street_store = re.findall(street_regex, text, flags=re.I)
        # Extracting matches with length > 1 (exclude false matches or empty strings)
        street_store = [builtins.max(t, key=len) for t in street_store if len(t) > 0]
        street_store = [street.strip() for street in street_store]
        # Limiting to streets with capitalized words only, all uppercase words only, and doesn't exclude 1st-10th 
        street_store = [street for street in street_store if re.sub(r'[^\w\s]','', street).istitle() or street.isupper() or len(re.findall(r'\d{1,10000}(st|rd|nd|th)',street,re.IGNORECASE)) > 0]
        # Limiting to streets that end with traditional street name ending (some false ones were slipping through the cracks)
        street_store = [street for street in street_store if re.sub(r'[^\w\s]','', street.split(' ')[-1]).strip().lower().capitalize() in ['Street', 'St', 'Road', 'Rd', 'Parkway', 'Pkwy',
        'Terrace', 'Avenue', 'Ave', 'Lane', 'Ln', 'Highway', 'Hwy', 'Square', 'Sq', 'Trail', 'Trl', 'Way', 'Drive', 'Dr', 'Court', 'Ct', 'Circle', 'Cir', 'Boulevard', 'Blvd', 'Place', 'Pl']]
        print("street store:", street_store)

        # Masking streets
        text_sub = sub_mult_regex(text, street_store, "PAddress", word_bounds=False)[0]
        # Keeping a running log of all address information that is captured and masked in csz
        csz = []
        if len(street_store) > 0:
            csz.extend(street_store)

        # Masking multi-word cities
        multiword_cities = mask_multiword_cities(text_sub)

        text_sub =  multiword_cities[0]
        csz.extend(multiword_cities[1])

        # Masking single-word cities and their corresponding abbreviated states (if any)
        single_city_ = single_city_abb_state(text_sub)
        csz.extend(single_city_[1])

        
        # Replacing all unabbreviated states; two-name states tended to be missed before (no conditions, tagged if word boundary match)
        states_ua = sub_mult_regex(single_city_[0], states_list, "PAddress", word_bounds=True)
        just_text = states_ua[0]
        if len(states_ua[1]) > 0:
            #print("masked states:", states_ua[1])
            csz.extend(states_ua[1])

        text_sub_split = just_text.split(' ')

        for i, val in enumerate(text_sub_split):
            # Zip code regexes
            long_zip_check = re.search(r'\d{5}-\d{4}', str(val)) 
            reg_zip_check = re.findall(r'\d{5}', str(val))
            #print(repr(val))
            # CHECKING ZIPS
            if len(reg_zip_check) == 1 and reg_zip_check[0] in val and re.sub(r'[^\w\s]','', reg_zip_check[0]) in zips_list:
                sub = re.sub(r'[^\w\s]','', val)
                #print("zip code:", reg_zip_check)
                # If word or two before zip has PAddress in it, then tag as zip
                if 'PAddress' in text_sub_split[i-1] or 'PAddress' in text_sub_split[i-2] or                 re.sub(r'[^\w\s]','', text_sub_split[i-1]).strip().lower().capitalize() in csz or re.sub(r'[^\w\s]','', text_sub_split[i-2]).strip().lower().capitalize() in csz:
                    csz.extend([sub])
                    text_sub_split[i] = re.sub(r'\d{1,100}', 'PAddress', sub)
            if long_zip_check != None and len(long_zip_check.group(0)) > 0:
                # If word or two before zip has PAddress in it, then tag as zip
                sub = re.sub(r'[^\w\s-]','', val)
                if 'PAddress' in text_sub_split[i-1] or 'PAddress' in text_sub_split[i-2] or re.sub(r'[^\w\s]','', text_sub_split[i-1]).strip().lower().capitalize() in csz or re.sub(r'[^\w\s]','', text_sub_split[i-2]).lower().capitalize() in csz:
                    #print("long zip code:", long_zip_check)
                    csz.extend([long_zip_check.group(0)])
                    text_sub_split[i] = re.sub(r'\d{1,100}', 'PAddress', sub)

        # Making sure that full states tagged in case some cities share a word
        # Formatting: ensuring that all PAddress tags are standalone, no buggy text attached (would happen with partial matches)
        for i,v in enumerate(text_sub_split):
            if v == 'PAddress' and text_sub_split[i-1].strip().lower().capitalize() in ['New', 'North', 'South', 'West', 'Puerto', 'American']:
                text_sub_split[i-1] = 'PAddress'
            if 'PAddress' in v:
                text_sub_split[i] = re.sub(r'\w{1,100}', 'PAddress', text_sub_split[i])

        print('csz:', set(csz))
        print("NEXT NOTE:", "\n\n")
        all_add_sub = [val for val in text_sub_split]
        all_add_sub = ' '.join(all_add_sub)
        #print("NEXT NOTE:")
        if len(all_add_sub.split(' ')) != len(text.split(' ')):
            raise Exception("Lengths of all_add_sub and text do not match:", len(all_add_sub.split(' ')), len(text.split(' ')), "\n", all_add_sub, "\n", text)

        return all_add_sub

    except Exception:
        print("EXCEPTION IN ADDRESS FUNCT AT TEXT:", "\n", text)
        #template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        #message = template.format(type(ee).__name__, ee.args)
        print(traceback.format_exc())
        return text



address_replacerUDF = F.udf(tag_address, StringType())


def hyphen_names(name):
    '''
    Helper function called by nameTag
    Searches for hyphenated names and tags them.
    If one half of the name is found in total_names and tagged, other half is tagged for model training
    (Anush suggestion)
    Returns text (called name param here) and list of hyphenated names found
    '''
    hyphen = r'\w{1,100}-\w{1,100}'
    matches = re.findall(hyphen, name)
    name_list = []
    if len(matches) > 0:
        for word in matches:
            if word.split('-')[0].strip().lower().capitalize() in distinct_names and word.split('-')[0].strip().lower().capitalize() not in exclude: #or word.split('-')[0] in only_names:
                name = re.sub(word.split('-')[0], 'PersonName', name)
                name_list.append(word.split('-')[0])
            if word.split('-')[1].strip().lower().capitalize() in distinct_names and word.split('-')[0].strip().lower().capitalize() not in exclude: # or word.split('-')[1] in only_names:
                name = re.sub(word.split('-')[1], 'PersonName', name)
                name_list.append(word.split('-')[1])
    inc_hyph = r"PersonName-\w{1,100}|\w{1,100}-PersonName"
    m_inc_hyph = re.findall(inc_hyph, name)
    m_inc_hyph = [val for val in m_inc_hyph if val != 'PersonName-PersonName']
    if len(m_inc_hyph) > 0:
        for found in m_inc_hyph:
            name =  re.sub(found, 'PersonName-PersonName', name)
    return name, name_list

# what output should look like: name = PersonName PersonName-PersonName 

def nameTag(address_text): 
    '''
    The current order of the names function is:
        1. Check for MD/Nurse names  (e.g., "Calvin McKinskey , RN")
        2. Check for names in the format Last, First Middle (e.g., KLEIN , MIRANDA S.)
        3. Check for names with Dr. or Mrs/Mr/Ms prefix
        3. Check for straggler names that show up in the total names list (from lastname/firstname cols and names txt file; e.g., "Deirdre")
        4. Check for middle names/initials sandwiched between two existing name tags (e.g., "(PersonName) M. (PersonName)")
        5. Check for hyphenated names
        6. Correct any possible mis-formatted name tags 

    Returns text with all names tagged (from whatevertext  was output from the address tagger function)
    '''
    try:
        #print(address_text)
        # Stores all identified names as text is iterated through
        only_names = [] 
        ###################### LAST MIDDLE FIRST ##################################

        # Regex to identify names in the Last , First Mid format (LFM)
        last_first_m = ":?\s?([A-Z][a-zA-Z]+)(\s?)(,)(\s)([A-Z][a-zA-Z]+)(.?)(\s?)(([A-Z][a-zA-Z]+)|([A-Z].?))?(\s)?(M?\.?D?\.?)|(R?\.?N?\.?A?\.?)"
        matches = re.finditer(last_first_m, address_text, re.MULTILINE)
        words = []
        for match in matches:
            words.append(match.group())
        print("lfm:", words)
        # Excluding LFM names with Dr suffixes
        # words = [name for name in words if " MD " not in name and "M.D." not in name]
        # Appending name up until month (eg., CLARK , JULIE APR); APR is actually the start of next sentence's date
        cutoff = ['\\bJAN\\b', '\\bFEB\\b', '\\bMAR\\b', '\\bAPR\\b', '\\bMAY\\b', '\\bJUNE\\b', '\\bJUL\\b', '\\bAUG\\b', '\\bSEP\\b', '\\bOCT\\b',
        '\\bNOV\\b', '\\bDEC\\b', '\\bVA\\b', '\\bSTAFF\\b', '\\bJANUARY\\b', '\\bFEBRUARY\\b', '\\bMARCH\\b', '\\bAPRIL\\b',
        '\\bJULY\\b', '\\bAUGUST\\b', '\\bSEPTEMBER\\b', '\\bOCTOBER\\b', '\\bNOVEMBER\\b', '\\bDECEMBER\\b', '\\bINTERN\\b', '\\bDOCTOR\\b',
        '\\bSURGEON\\b', '\\bNURSE\\b']
        for i,x in enumerate(words):
            for m,y in enumerate(cutoff):
                check = re.findall(y, x, re.IGNORECASE)
                if len(check) > 0:
                    print(check)
                    words[i] = re.sub(check[0], '', x)
        ############################ MD AND NURSES ######################################

        # Regex to identify Drs and nurses - suffixes: MD/M.D., RN/R.N., and RNA/R.N.A and D.O./DO

        md = r"([A-Z][a-zA-Z]+(\s+)([A-Z][a-zA-Z]+|[A-Z]\.?)?(\s+)?([A-Z][a-zA-Z]+)(\s+)?(\,)?(\s+)?)(M\.?D\.?|R\.?N\.?A?\.?)"
        rmd = re.finditer(md, address_text, re.MULTILINE)
        docs = []
        for matchNum, match in enumerate(rmd, start=1):
            docs.append(match.group()) 
        print("docs", docs)
        # Filtering out cases that aren't actually names
        only_docs = [string for string in docs if 'asst' not in string.lower() and 'neuro' not in string.lower()]
        print("removing asst and neuro:", only_docs)
        only_docs = [doc for doc in only_docs if ' '.join(doc.split(' ')[:-1]).istitle() or doc.isupper()]
        print("docs filt:", only_docs, "should have been title or doc should've been upper")
        #only_docs = [name for name in only_docs if name != 'MD' and name != 'M.D.' and name != 'M.D']
        only_names.extend(only_docs)
        # This section looks at all LFM names and checks to see if the capitalization for each word matches.
        # Eg., "CLARK , SANDY M." is more likely to be a name; "Clark, Sandy MARGARET" is much more likely to be
        # a name where part of the next sentence accidentally got attached. A common example of this is 
        # "JONES , MYNA At:" ("At" is not a middle name in this case). If capitalization doesn't match,
        # the first two words of the name will be appended and the last word ignored. If the name is only
        # two words, they'll only be appended if both have matching capitalization
        new_words = []
        for index, string in enumerate(words):
            # Replacing periods in the string to avoid name tagger tagging them
            string = string.replace(".", "")
            # Identifying the last word in the string (typically Middle Name)
            find_last = re.findall(r'^.*\b(\w+).*$', string)
            if len(find_last) > 0:
                last = find_last[0]
            name_split = string.split()
            name_split = [re.sub(r'[^\w\s]','', word) for word in name_split]
            name_split = [i for i in name_split if i.isalnum()]
            # If the name has three words:
            if len(name_split) == 3:
                # If all three names in the string are uppercase, then append the string as is (covers situation w/ middle initial since it would be capitalized)
                if name_split[0].isupper() and name_split[1].isupper() and name_split[2].isupper(): # should cover capitalized names with uppercase middle initials
                    new_words.append(string)
                # If all three names are capitalized but lowercase, append as is (unless the third word is "At", in which case just append the first two words)
                elif (name_split[0][0].isupper() and name_split[0][1:].islower()) and (name_split[1][0].isupper() and name_split[1][1:].islower()) and (name_split[2][0].isupper() and name_split[2][1:].islower()):
                    if name_split[2] == 'At' and len(find_last) > 0:
                        new_name = re.sub(last, "", string)
                        new_words.append(re.sub(r'[^\w]+$', "", new_name)) # append w/o punctuation
                    else:
                        new_words.append(string)
                # If the first two names are capitalized but lowercase, but the third word is an initial (capitalized), append the string as is
                elif (name_split[0][0].isupper()) and name_split[1][0].isupper() and len(name_split[2]) == 1:
                        if name_split[2].isupper():
                            new_words.append(re.sub(r'[^\w]+$', "", string)) 
                else:    
                    if len(find_last) > 0:
                        new_name = re.sub(last, "", string)
                        name_split = new_name.split()
                        name_split = [re.sub(r'[^\w\s]','', word) for word in name_split]
                        name_split = [i for i in name_split if i.isalnum()]
                        if (name_split[0].isupper() and name_split[1].isupper()) or (name_split[0][0].isupper() and name_split[0][1:].islower() and name_split[1][0].isupper() and name_split[1][1:].islower()):
                            new_words.append(re.sub(r'[^\w]+$', "", new_name))
            # If the name has two words, append both words only if they're of the same capitalization format
            elif len(name_split) == 2:
                if (name_split[0].isupper() and name_split[1].isupper()) or (name_split[0][0].isupper() and name_split[0][1:].islower() and name_split[1][0].isupper() and name_split[1][1:].islower()):
                    new_words.append(string)

        # Further narrowing LMF. Of the filtered list above, only keep strings where
        # at least one of the words in that string is a name that shows up in the 
        # total names list
        print("lfm names filtered:", new_words)
        # 

        for string in new_words:
            # Splitting string into words to see which ones are actually names
            split = re.split('([\s.,;()]+)', string)
            for l in split:
                if l.strip().lower().capitalize() in distinct_names and l.strip().lower().capitalize() not in exclude:
                    # If substring in name string is name, append entire string (w/o punctuation)
                    only_names.append(re.sub(r'[^\w]+$', "", string))
                    # Now that the string is appended to only_names, remove from new_words list
                    new_words = [word for word in new_words if word != string]
        print("lfm filt if name:", new_words)
        # RN/MD etc masking
        text_sub = sub_mult_regex(address_text, only_docs, "PersonName", word_bounds=False)[0]
        # Last, First Mid masking
        text_sub = sub_mult_regex(text_sub, only_names, "PersonName", word_bounds=False)[0]
        #print("only docs:", only_docs)
        #print("lfm:", only_names)
        only_names.extend(only_docs)


        ######### DR. AND MR/MS/MRS PREFIXES ####################

        pref_reg = r'(Dr\.?|Mr\.?|Ms\.?|Mrs\.?)\s([a-zA-Z]+)'
        r_pref = re.compile(pref_reg)
        # Listing all the matches, group by group
        res_pref = r_pref.findall(text_sub) 
        # Unpacking tuples into strings, one per LFM name
        words_pref = [''.join(tup[1]) for tup in res_pref if tup[1] != 'Person']
        print("dr/mr/mrs:", words_pref)

        text_sub = sub_mult_regex(text_sub, words_pref, "PersonName", word_bounds=False)[0]
        only_names.extend(words_pref)
        # Now, splitting the text into chunks for total_names and middle name check (so far, MD/RN/LFM should have already been tagged)
        text_chunks1 = text_sub.split(' ') # <<<<<<<<< this is what fixed the length error in the replacereportText
        temp_tag1 = ' '.join([str(i) for i in text_chunks1])

        #################### NAMES IN TOTAL NAMES LIST (FIRST/LAST IN DF AND NAMES TXT FILE) ##################

        all_tagged_nl = []
        for i,v in enumerate(text_chunks1):
            if len(v) > 1:
                if v[0].isupper():
                    if (v.strip().lower().capitalize() in distinct_names or re.sub(r'[^a-zA-Z\d\s]', "", v).strip().lower().capitalize() in distinct_names):
                        if v.strip().lower().capitalize() not in exclude: #or 'mike' in v.lower().capitalize() or 'james' in v.lower().capitalize():
                            print("single word name:", v)
                            only_names.append((text_chunks1[i]))
                            all_tagged_nl.append((text_chunks1[i]))
                            text_chunks1[i] = re.sub(r'\w{1,100}', 'PersonName', text_chunks1[i])
        temp_tag3 = ' '.join([str(i) for i in text_chunks1])
        #print('all tagged in total names:', all_tagged_nl)


        ################### MIDDLE NAMES ######################

        all_tagged_mn = []
        for i,v in enumerate(text_chunks1):
            if v.isalpha() or "." in v: # checking for middle initials/names without punct besides '.'
                if v != 'PersonName' and v[0].isupper() and i not in [0, len(text_chunks1)-1]:
                    if "PersonName" in text_chunks1[i+1] and "PersonName" in text_chunks1[i-1]:
                        print("in middle names:", v)
                        v = re.sub(r'[.,]', "", v) # for cases where there is punctuation in the string (doesn't remove it permanently)
                        only_names.append(text_chunks1[i])
                        all_tagged_mn.append(text_chunks1[i])
                        text_chunks1[i] = re.sub(r'\w{1,100}', 'PersonName', text_chunks1[i])


        text_tag4 = ' '.join([str(i) for i in text_chunks1])

        ############ HYPHENATED NAMES #############################

        hyphen_mask = hyphen_names(text_tag4)
        print("hyphen names:", hyphen_mask[1], "\n")
        only_names.extend(hyphen_mask[1])

        ########## FORMATTING PERSONNAME TAGS WHERE NEEDED ############

        final_check = hyphen_mask[0].split(' ')
        for i,v in enumerate(final_check):
            if 'PersonName' in v:
                final_check[i] = re.sub(r'\w{1,100}', 'PersonName', final_check[i])

        name_bound = ' '.join([str(i) for i in final_check])

        #############################################################
        # For testing purposes:
        ##only_names = [re.sub(r'[^\w]+$', "", word) for word in only_names]
        #only_names = [re.split('([\s.,;()]+)', word) for word in only_names]
        #only_names = sum(only_names, [])
        #punct = ['.', ',', '\\', '/', '!', '#', '$', '%', '(', ')', '_', '-']
        #tags = ['PersonName', 'PAddress']
        #only_names = [name for name in only_names if name not in tags]
        #only_names = [re.findall(r'\w+', name) for name in only_names]
        #only_names = sum(only_names, [])
        print("all names:", set(only_names), "\n")
        if len(name_bound.split(' ')) != len(address_text.split(' ')):
            raise Exception("Lengths of name_bound and text do not match:", len(name_bound.split(' ')), len(address_text.split(' ')), "\n", name_bound, "\n", address_text)
        return name_bound
    except Exception:
        print("EXCEPTION FOR NAMES AT:", repr(address_text))
        print(traceback.format_exc())
        #template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        #message = template.format(type(e).__name__, e.args)
        return address_text
        
    ################################################################


names_replacerUDF = F.udf(nameTag, StringType())


def replaceSSN(df):
    ssn_pattern = '(\d{3}-\d{2}-\d{4}|\b\d{9}\b)'
    df = df.withColumn('ExtractedSSNArray', F.expr(f"regexp_extract_all(ReportTextMasked, '{ssn_pattern}', 0)"))             .withColumn('ExtractedSSNCount', F.size('ExtractedSSNArray'))             .withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', r'\d{3}-\d{2}-\d{4}','SSNumber'))             .withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', r'\b\d{9}\b','SSNumber')) 

    return df

def phoneNumberSplitter(word):
    phone_nums = re.findall("(?:\d-)?[\+]?[(]?[0-9]{3}[)]?[-\s\.]{0,2}[0-9]{3}[-\s\.]?[0-9]{4,6}[)]?", word)
    subs = [(x, (len(x.split()) * ' PhoneNumber').strip()) for x in phone_nums] #creates the P substitution string

    for num, sub in subs:
        word = word.replace(num, sub)

    return word

def replacePhone(df):
    phone_replacerUDF = udf(lambda q : phoneNumberSplitter(q), StringType())

    df = df.withColumn("ReportTextMasked", phone_replacerUDF("ReportTextMasked"))

    return df

def replaceEmail(df):
    email_pattern = r'\S{1,}@\S{2,}\.\S{2,}'
    df = df.withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', email_pattern, 'EAddress')) 
    return df

def replaceURL(df):
    url_pattern = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    df = df.withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', url_pattern, 'URLAddress'))

    return df

def replaceIP(df): 
    ip_pattern = r'(\b25[0-5]|\b2[0-4][0-9]|\b[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}'
    df = df.withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', ip_pattern, 'IPAddress'))

    return df

def replaceAccountNumber(df):
    account_pattern = '\b\d{8}\b'
    df = df.withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', account_pattern, 'AccountNumber'))

    return df

def reportTextSpacer(reportText, reportTextMasked):
    '''
    Report text is spaced (reportTextSpacer)

    reportTextSpacer (text_replacerUDF) → Goes in and looks at whether reportText and ReportTextMasked
    have the same lengths when split on space. If same length, spaces out special characters in untagged text
    Puts the output of the reportText text with special characters spaced out into a new column, ReportTextCleaned
    (reportText and ReportTextMasked don't get edited, new column created). This function is run twice - spaces on reportText (ReportTextCleaned)
    and ReportTextMasked (text_replacer2UDF, called in outsideTagging()). We pass ReportTextMasked into IOB function (outsideTagging), 
    which becomes labels column; the spacing on reportText (ReportTextCleaned) becomes the text column, and we need both of these columns as input to the NER model
    '''
    rt_arr = reportText.split(' ')
    mrt_arr = reportTextMasked.split(' ')

    if len(rt_arr) == len(mrt_arr):
    # if the word in reportTextMasked is a tagged word, we dont want to split the word in reportText because the tagged word may contain special characters
    # For example:
    # ReportText      : Anush's Phone number is (123)345-4566. Hello more-text here
    # ReportTextMasked: PersonName O O O PhoneNumber O O O O O O
    # 
    # We wouldnt want to space the phone number in report text. We want to leave it as such
        for i, word in enumerate(mrt_arr): # passes each word (split on spaces) to the spacer helper function below
            rt_word = rt_arr[i]

            pattern_list = set(re.findall("[^\s\d\w']", word))
            for char in pattern_list:
                rt_word = rt_word.replace(char, "  {0}  ".format(char)) #casues issues if the special character is also found inside the masked string (small % of records)

            rt_arr[i] = rt_word

        # combine and clean the array of spaced out reportText words
        combined = ' '.join(rt_arr)
        combined = re.sub(" +", ' ', combined) # GEtting rid of plus signs, not sure yet why this is necessary

        return combined
    else:
        return "ReportText Lenght Error Ocurred"


def replaceReportText(df):
    text_replacerUDF = udf(lambda y,q : reportTextSpacer(y,q), StringType())

    df = df.withColumn("ReportTextCleaned", text_replacerUDF("ReportText", "ReportTextMasked"))         .withColumn('ReportTextCleaned', trim(col("ReportTextCleaned")))

    return df

def oTagger(reportText):
    search = re.compile(r"[^\s\d\w]|\b(?!(PhoneNumber|SSNumber|PersonName|PAddress|EAddress|URLAddress|IPAddress|AccountNumber)\b)\w+'?[a-zA-ZÀ-ÖØ-öø-ÿµ²]*")
    return re.sub(search, ' O ', reportText)

###
# Takes the masked report text column (masked with the AEXNP tags) and replaces everything else (even special characters) with an O
# 
# Input
# df DataFrame:  of reportText data
#
# Return DataFrame
# dataframe with IOB tagged column included
###
def outsideTagging(df):
    '''
    Removes square brackets and plus signs and replaces with spaces, like in whitespaceLocation, except it's doing this for ReportTextMasked 
    and not ReportTextCleaned (technically a duplicate effort on a different column). reportTextSpacer is called here the second time to space
    in ReportTextMasked. The labels on this column are generated by calling OTaggerUDF (from otagger()), which will turn everything that isn't
    a masking tag into "O". This generates a new column from ReportTextMasked, ReportTextIOB
    '''
    df = df.withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', '\r'                             ,' '))         .withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', ','                                      ,' '))         .withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', '\n'                                     ,' '))         .withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', '\t'                                     ,' '))         .withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', '\['                                     ,' '))         .withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', '\]'                                     ,' '))         .withColumn('ReportTextMasked', regexp_replace('ReportTextMasked', ' +'                                    ,' '))         .withColumn('ReportTextMasked', trim(col("ReportTextMasked")))         .na.drop(subset=['ReportTextMasked']).cache()

    text_replacer2UDF = F.udf(reportTextSpacer, StringType())
    df = df.withColumn("ReportTextMasked", text_replacer2UDF("ReportTextMasked", "ReportTextMasked"))           

    ###
    # replaces 's with nothing. Prevents words like Veteran's into Veteran ' s which would be tagged as O O O Instead of O
    # replaces everything that isnt a tag into an outside tag
    # reduces double spaces into single spaces for clarity
    ###
    df = df.withColumn('ReportTextIOB', regexp_replace("ReportTextMasked", r"'[a-zA-Z]{0,2}", ''))

    OTaggerUDF = F.udf(oTagger, StringType())
    df = df.withColumn('ReportTextIOB', OTaggerUDF("ReportTextIOB"))         .withColumn('ReportTextIOB', regexp_replace("ReportTextIOB", r" +", ' '))         .withColumn('ReportTextIOB', trim(col("ReportTextIOB"))) 

    df = df.withColumn('ReportTextIOB', regexp_replace("ReportTextIOB", r" +", ' ')) 

    return df

def replacementCharacterLocation(ReportTextCleaned):    
    a = re.finditer('[\\v\\n\\t\\r,\[\]] *', ReportTextCleaned)
    return [(match.start(), match.end(), match.group()) for match in a]

def whitespaceLocation(df):
    '''
    whitespaceLocation → Calls replacementCharacterLocation (special_character_udf) function to record the location of 
    all whitespace characters (\r, \t, \v) into a new column called Locations (tuple with actual text and location info; 
    it'll be saved and when we get the predictions they'll be inserted so it's easier to read, then it'll be replaced with empty strings). 
    Removes square brackets and plus signs in ReportTextCleaned and replaces with space as well, because NER system 
    can't take them in (presumably, Anush is unsure)
    '''
    tuple_schema = ArrayType(StructType([
        StructField("start", IntegerType(), False),
        StructField("end", IntegerType(), False),
        StructField("character", StringType(), False)
    ]))
    #print("TUPLE SCHEMA:", tuple_schema, "\n\n")
    special_character_udf = F.udf(replacementCharacterLocation, tuple_schema)
    df = df.withColumn("Locations", special_character_udf("ReportTextCleaned"))

    #clean report text
    # brackets being replaced as well
    #.withColumn('ReportText', regexp_replace('ReportText', r'([^a-zA-Z\d\s]|- )(?=[^ |\d]{0,}\1)'   ,' ')) \
    df = df.withColumn('ReportTextCleaned', regexp_replace('ReportTextCleaned', '\r'                             ,' '))         .withColumn('ReportTextCleaned', regexp_replace('ReportTextCleaned', ','                                      ,' '))         .withColumn('ReportTextCleaned', regexp_replace('ReportTextCleaned', '\n'                                     ,' '))         .withColumn('ReportTextCleaned', regexp_replace('ReportTextCleaned', '\t'                                     ,' '))         .withColumn('ReportTextCleaned', regexp_replace('ReportTextCleaned', '\['                                     ,' '))         .withColumn('ReportTextCleaned', regexp_replace('ReportTextCleaned', '\]'                                     ,' '))         .withColumn('ReportTextCleaned', regexp_replace('ReportTextCleaned', ' +'                                    ,' '))         .withColumn('ReportTextCleaned', trim(col("ReportTextCleaned")))         .na.drop(subset=['ReportTextCleaned']).cache()

    return df      

def tagCorrector(tag_list):
    '''
    Adding prefixes to certain tags where needed; called on ReportTextIOB, which then becomes 
    ReportTextIOB_old and the updated tag-corrected one is the new ReportTextIOB
    '''
    tag_list_split = [var for var in tag_list.split(" ") if var]
    previous_tag = ''
    correted_tag_list = []
    for i, stag in enumerate(tag_list_split):
        tag=''
        if stag != 'O':
            if stag != previous_tag:
                tag = "B-"+stag            
            else:
                tag = "I-"+stag   
    #         print(f"{i} {stag} {tag}")
        else:
            tag = stag
        if not tag == 'B-' and not tag =='B- ':
            correted_tag_list.append(tag)
            previous_tag = stag
    correted_tag_string = " ".join(correted_tag_list)

    return correted_tag_string     

def reinsertSpecialCharacters(ReportTextCleaned, Locations):
    '''
    This function was created but never called in main function
    '''
    for i in Locations:
        start,end,char = i
        first = ReportTextCleaned[:start]
        second = ReportTextCleaned[start:]
        ReportTextCleaned = first + char + second

    return ReportTextCleaned

schema = StructType([
    StructField("split_equals", IntegerType(), True),
    StructField("length_text", IntegerType(), True),
    StructField("length_lables", IntegerType(), True),
])

def checklinesframelikenemo(text1,lables1):
    '''
    Confirming that the text and labels are of equal length
    '''
    try:
        textarray=text1.strip().split()
        labelsarray=lables1.strip().split()
    except: 
        return 0, 1, 0

    if len(textarray) == len(labelsarray):
        return 1,len(textarray),len(labelsarray)
    else:
        return 0,len(textarray),len(labelsarray)

# Create the UDF, note that you need to declare the return schema matching the returned type
checklinesframelikenemo_udf= udf(checklinesframelikenemo, schema)

def check_labels(labels):
    '''
    Checking if all the labels are valid (should be no labels besides the set we agreed on)
    '''
    label_list = ['O', 'B-PhoneNumber', 'I-PhoneNumber', 'B-SSNumber', 'I-SSNumber', 'B-PersonName', 'I-PersonName', 'B-PAddress', 'I-PAddress', 'B-EAddress', 'I-EAddress', 'B-URLAddress', 'I-URLAddress', 'B-IPAddress', 'I-IPAddress', 'B-AccountNumber', 'I-AccountNumber']
    labelsarray = []
    if not labels:
        return 5
    try:
        labelsarray = labels.strip().split()
    except:
        return 4

    if not isinstance(labels, str):
        return 3

    for label in labelsarray:
        if label not in label_list:
            return 2

    return 1

check_labels_udf = udf(check_labels, IntegerType())


# ## Make sure these columns are understood:
# 
# - reportText: the original note itself with no changes
# - ReportTextMasked: the original note but with all PII already masked
# - ReportTextIOB: ReportTextMasked, except with special characters spaced and turned into labels (non-PII label is 'O'), and the tagCorrector step is done (before tag correcter, is ReportTextIOB_old). This newest version becomes the labels input into NER. 
# - ReportTextCleaned: reportText, except special characters etc. are spaced out with space before and after them (output of reportTextSpacer function); so nothing is masked, here. This becomes the text input into NER
# 
# 
# ## Outputs:
# - parquet file of TIU notes and their corresponding tags 
# 
# ## Validation step: 
# - Make sure that reportText and ReportTextMasked columns have the same length after running the replaceReportText() function. 
# - The checknemo functions (that print ‘split equals’ and ‘label validation’) tell us if each ReportTextCleaned vs. ReportTextIOB instance has the same length when split on space, and if each label is a valid label; the end number of rows in the dataframe output by main should equal the number of rows of the initial dataframe it took in, that would mean that no rows were dropped by those two validation functions 


def main(df):
    runStart = time.time()

    #df = retrieveColumns(df).cache()
    one = time.time()
    #print("retrieveColumns, ", one-runStart)

    ### Label Tagging
    print("sample before address:", repr(df.collect()[0]['ReportText']), repr(df.collect()[1]['ReportText']), repr(df.collect()[2]['ReportText']), repr(df.collect()[3]['ReportText']), repr(df.collect()[4]['ReportText']), "\n\n")
    df = df.withColumn("ReportTextMasked", address_replacerUDF("ReportText"))
    print("sample after address:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    df = df.withColumn("ReportTextMasked", names_replacerUDF("ReportTextMasked"))
    print("sample after name:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    df = replaceEmail(df)
    print("sample after email:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    df = replaceSSN(df)
    print("sample after ssn:",repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    df = replacePhone(df)
    print("sample after phone:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")

    df = replaceURL(df)
    print("sample after url:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    df = replaceIP(df)
    print("sample after ip:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    df = replaceAccountNumber(df)
    print("sample after acc:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    two = time.time()
    print("Label Tagging: ", two-one)

    ###### Text Cleanup
    df = replaceReportText(df)
    print("sample after replacereporttext:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    three = time.time()
    print("replaceReportText, ", three-two)
    print("\n\n")

    df = whitespaceLocation(df)
    print("sample after whitepacelocation:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    four = time.time()
    print("whitespaceLocation, ", four-three)

    df = outsideTagging(df)
    print("sample after outsidetagging:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    five = time.time()
    print("outsideTagging, ", five-four)

    tag_replacerUDF = F.udf(tagCorrector, StringType())
    df = df.withColumnRenamed("ReportTextIOB", "ReportTextIOB_old")
    print("sample after tagreplacer:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    df = df.withColumn("ReportTextIOB", tag_replacerUDF("ReportTextIOB_old"))
    print("sample after tagreplacer2:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    six = time.time()
    print("tagCorrector: ", six-five)

    #Size and Label Validations
    df = df.withColumn('label_validation', check_labels_udf('ReportTextIOB'))
    print("sample after label validation:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    df.select('label_validation').groupBy('label_validation').count().show()
    df = df.filter("label_validation == 1")
    seven = time.time() 
    print("Label Validation: ", seven-six)

    df = df.withColumn('new', checklinesframelikenemo_udf('ReportTextCleaned','ReportTextIOB'))
    print("sample after checklikenemo:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    df.select('new.*').groupBy('split_equals').count().sort(desc('split_equals')).show()
    df = df.select('TIUDocumentSID', 'new.*', 'ReportText', 'ReportTextMasked', 'ReportTextCleaned', 'ReportTextIOB', 'ReportTextIOB_old', 'locations').filter(F.col('split_equals')=='1')#.show(truncate=False)
    eight = time.time() 
    print("Size Validation: ", eight-seven)

    df = df.dropDuplicates((['ReportTextCleaned']))
    print("sample after drop dup:", repr(df.collect()[0]['ReportTextMasked']), repr(df.collect()[1]["ReportTextMasked"]), repr(df.collect()[2]["ReportTextMasked"]), repr(df.collect()[3]["ReportTextMasked"]), repr(df.collect()[4]["ReportTextMasked"]), "\n\n")
    fifteen = time.time()
    print("DropDuplicates: ", fifteen-eight)
    print("Execution time: " + str(time.time() - runStart))
    return df.select('TIUDocumentSID', 'ReportText', 'ReportTextCleaned', 'ReportTextIOB', 'ReportTextMasked')


