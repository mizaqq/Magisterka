import pandas as pd
f=r"produkty.xls"
product=pd.read_excel('produkty.xls')
product=product.drop(columns=['Unnamed: 5','Unnamed: 6'])
product.columns=['prod','kcal','p','f','c']