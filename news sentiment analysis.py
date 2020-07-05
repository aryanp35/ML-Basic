#!/usr/bin/env python
# coding: utf-8

# In[11]:


from google import google
from textblob import TextBlob
import texttable as tt
from time import sleep
# import TextBlob for text analysis
# import google for google search


# In[7]:



def search1 (site, search):
    site = site
    search = search
    num_page =5
    text='inurl:'+site+' intext:'+ search
    search_results= google.search(text)
    search_results_list = []
    subjectivity_list = []
    polarity_list = []
    num = []
    number = 1
    
    for reports in search_results:
        search_results=reports.description
        search_results_list.append(search_results)
        analysis = TextBlob(search_results)
        subjectivity=analysis.sentiment.subjectivity
        subjectivity_list.append(subjectivity)
        polarity=analysis.sentiment.polarity
        polarity_list.append(polarity)
        number=number + 1
        num.append(number)
        sleep(5)
        
    tab=tt.Texttable()
    heading=['Number','Result','Subjectivity','Polarity']
    tab.header(heading)

    for row in zip(num,search_results_list,subjectivity_list,polarity_list):
        tab.add_row(row)
        
    avg_subjectivity=sum(subjectivity_list)/len(subjectivity_list)
    avg_polarity=sum(polarity_list)/len(polarity_list)
    table=tab.draw()
    print (site)
    print (search)
    print (table)
    print (site + "avg_subjectivity "+ str(avg_subjectivity))
    print (site + "avg_polarity "+ str(avg_polarity))


# In[ ]:



search1('zee','corona')


# In[ ]:




