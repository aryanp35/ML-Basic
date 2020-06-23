#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
years=['2019A','2020F','2021F','2022F','2023F','2024F']
sales = pd.Series(index=years)
sales['2019A']=15
sales


# In[35]:


growth=0.1
for year in range(1,6):
    sales[year]=sales[year-1]*(1+growth)


# In[36]:


ebitda_margin = 0.20
depr_percent = 0.03
ebitda = sales * ebitda_margin
depreciation = sales * depr_percent
ebit = ebitda - depreciation
tax_rate = 0.30
tax_payment = -ebit * tax_rate
# for year in range(0,6):
#     tax_payment[year]=-ebit[year]*tax_rate
tax_payment = tax_payment.apply(lambda x: min(x,0))
nopat = ebit + tax_payment
nopat


# In[37]:


nwc_percent = 0.24
nwc = sales * nwc_percent
change_in_nwc = nwc.shift(1) - nwc
capex_percent = depr_percent
capex = -(sales * capex_percent)
capex


# In[53]:


free_cash_flow = nopat + depreciation + capex + change_in_nwc
free_cash_flow[1:]


# In[83]:


cost_of_capital=0.10
terminal_growth=0.02
terminal_value=free_cash_flow[-1]*(1+terminal_growth)/(cost_of_capital-terminal_growth)

presentvalue=pd.Series(index=years)
disfac=[(1/(1+cost_of_capital))**i for i in range(1,6)]
present_terminal = terminal_value*disfac[-1]

dcf_value=sum(free_cash_flow[1:]*disfac) + present_terminal
dcf_val=pd.Series(index=years)
dcf_val[0]= dcf_value
dcf_val


# In[86]:


# Exporting the Data to Excel
output = pd.DataFrame([sales, ebit, tax_payment, nopat, 
                       depreciation, capex, change_in_nwc,
                       free_cash_flow,dcf_val],
                     index=["Sales", "EBIT", "Tax Expense", 
                            "NOPAT", "D&A Expense",
                            "Capital Expenditures",
                            "Increase in NWC",
                            "Free Cash Flow",'DCF']).round(2)
# output.to_excel('Python DCF Model.xlsx')
output
# sum(free_cash_flow[1:]) + terminal_value

