import bazaproduktow as bp
import zczytywanie as z
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import re

def match_prod(p):
    best_match = process.extractOne(p, df['prod'], scorer=fuzz.token_sort_ratio) 
    index= df.loc[df['prod']==best_match[0]]
    if(best_match[1]<50):
        print("Czy to",best_match[0],"?")
        if(input()!="Tak"):          
           print("Podaj dane produktu")
           print("Nazwa, kalorie, bialko, tłuszcz, wegle")
           for i in range(0,5):
               index[i]=int(input())
    index =index.to_numpy()
    return index
    
def weight(p):
    pattern = '(\d+ (s|ł|g|k)[^\s]+)'
    result = re.findall(pattern, p)
    if(not result):
        pattern = '\d'
        result = re.findall(pattern, p)
    return result
    
def splitter(res):
    if(len(res)>0):
        res = res[0][0].split()
        if(int(res[0])>99):
            res[0]=int(res[0])/100
    return res
    
df=bp.product;
produkty = z.przepis.splitlines()
while("" in produkty):
    produkty.remove("")
    
tab_prod=[]
kalorie =0
for i in produkty:   
    index = match_prod(i)
    mnoznik = splitter(weight(i))
    tab_prod.append(index.ravel().tolist())
    kalorie +=index[0][1]*int(mnoznik[0])
index.tolist

