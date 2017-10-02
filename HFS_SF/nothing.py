#encoding=utf8
from tools import commontools as gtf
from tools import specialtools as gts
import time
import re
import csv
import os 

gt = gtf()

# filep = r'G:\MyPapers\ITS_Bio\co-author\au.out'
# anlist = gt.csv2list(filep,'\t')
# anlistw = anlist 
# h = 0
# i = 1
# try:
#     while 1:
#         j = 1
#         while anlist[i-1][0]==anlist[i][0]:        
#             i+=1
#             j += 1  
#         k = 1.0/j
#         h+=1
#         for t in range(i-j,i):
#             t = 0 if i==j else t
#             anlistw[t].append(k)
# #         print anlistw[i-1][0], '\t',anlistw[i-1][1],'\t', anlistw[i-1][2]
# #         if j<2:
#         i+=1     
#         
#         if i>2000:
#             break
# except:
#     print 'fuck'
    
filep = r'G:\MyPapers\ITS_Bio\co-author\au_weight.out'
anlist = gt.csv2list(filep,'\t') 
i = 1
result = []
try:
    while i<2002:
        j = 1
        sum = float(anlist[i-1][2])
        while anlist[i-1][1]==anlist[i][1]:        
            i+=1
            j += 1 
            sum+=float(anlist[i-1][2])
        result.append([anlist[i-1][1],sum])
        print anlist[i-1][1],'\t',sum
        i+=1
except Exception,e:
    print e
   
# for item in result:
#     print item[0], '\t',item[1]#, '\t',item[2]



# anlistw.sort(key=lambda x:x[1]) 
# print anlistw[i][0], '\t',anlistw[i][1],'\t', anlistw[i][2]
# scorelist = []
# i = 0
# j = 0
# while 1:
#     score = 0 
#     try: 
#         while anlistw[i][1]==anlistw[i+1][1]:
#             score+=anlistw[i][2]
#             i+=1
#             j =  i
#             if i>2001:
#                 break
#     except:
#         pass
#     
#     if i>2001:
#         break    
#     scorelist.append([anlistw[i][1],score])
#     print anlistw[i][0],anlistw[i][1],score
        
        
    
        
