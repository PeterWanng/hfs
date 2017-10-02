#coding=utf-8
import os
import string
import csv
import numpy
import re
import random
import matplotlib.pylab as plt

class TextArea(object):  
    def __init__(self):  
        self.buffer = []  
  
    def write(self, *args, **kwargs): 
        self.buffer.append(args) 

class specialtools():
    def map2vectors(self,listab,list2deal):
        "IN:mapping standard listab; list to deal, listab[0] must be sorted from min to max. the dimision of listab[0],listab[1] and list2deal are must the same"
        "OUT:mapping list2deal to the listab"
        "Example:"
        """listab=[[1,2,3],[3,6,9]]
        list2deal=[2,3,1]
        out=[6,9,3]"""
        result = []
        itemy = 0
        for item in list2deal:
            item = float(item)
            itemy = item
            j = -1
            for itemstand in listab[1]:
                itemstand = float(itemstand)
                j+=1
                if float(itemstand)>=item:
                    itemy = listab[0][j]
                    result.append(itemy)
                    break
                 
        return result           
        
            
class commontools():
    _path_='mycode\hfs_sf'
    kmeans3group = [['3344178035788881.coc.gml','3344605319892924.coc.gml','3356800646950624.coc.gml','3356881155164816.coc.gml','3358716283896811.coc.gml','3368776344558652.coc.gml','3369168951009868.coc.gml','3369278306978444.coc.gml','3370126999415642.coc.gml','3455798066008083.coc.gml','3464345767606270.coc.gml','3464705244725034.coc.gml','3464854910351381.coc.gml','3479304597792637.coc.gml','3489137279411789.coc.gml','3502012958680889.coc.gml','3506858382217257.coc.gml','3507543877973721.coc.gml','3507607539020444.coc.gml','3507662178094930.coc.gml','3507671015760502.coc.gml','3508278808120380.coc.gml','3510108007340190.coc.gml','3510150776647546.coc.gml','3510947052234805.coc.gml','3511312581651670.coc.gml','3512192651209611.coc.gml','3512207839497881.coc.gml','3512220862117440.coc.gml','3512224943485819.coc.gml','3512225370844164.coc.gml','3512260125547589.coc.gml','3512261920248384.coc.gml','3512288331952105.coc.gml','3512343789230980.coc.gml','3512346909867070.coc.gml','3512367365458914.coc.gml','3512387527089684.coc.gml','3512557425820703.coc.gml','3512564992138128.coc.gml','3512568150742963.coc.gml','3512586882317341.coc.gml','3512598488183675.coc.gml','3512631170591260.coc.gml','3512638635619787.coc.gml','3512649558390590.coc.gml','3512661431797880.coc.gml','3512665513423714.coc.gml','3512673221800564.coc.gml','3512681942436390.coc.gml','3512693292397369.coc.gml','3512703459106709.coc.gml','3512704620407286.coc.gml','3512722819965166.coc.gml','3512723826282658.coc.gml','3512731699677616.coc.gml','3512751521888667.coc.gml','3512755191392034.coc.gml','3512764419413627.coc.gml','3512767539668338.coc.gml','3512944329886751.coc.gml','3512956526933221.coc.gml','3512965133226559.coc.gml','3513008209201946.coc.gml','3513054618763335.coc.gml','3513299721572710.coc.gml','3513353957580369.coc.gml','3513457897170153.coc.gml','3513472585606907.coc.gml','3513477123020136.coc.gml','3513651849587189.coc.gml','3513665519425522.coc.gml','3513684632475119.coc.gml','3513732681977292.coc.gml','3513733382475662.coc.gml','3513737430369962.coc.gml','3513738009766461.coc.gml','3513747752676493.coc.gml','3513797614859184.coc.gml','3513821030864621.coc.gml','3514047335033747.coc.gml','3514054737834781.coc.gml','3514061079292261.coc.gml','3514074715123866.coc.gml','3514083762166772.coc.gml','3514112790871598.coc.gml','3514202721379789.coc.gml','3514207033581502.coc.gml','3514216143529145.coc.gml','3514229367981237.coc.gml','3514287944897392.coc.gml','3514415834044554.coc.gml','3514416295764354.coc.gml','3514448201592653.coc.gml','3514574454119360.coc.gml','3514712136139677.coc.gml','3514721241880789.coc.gml','3514725432613389.coc.gml','3516665499181717.coc.gml','3516669001647040.coc.gml','3517122988859030.coc.gml','3517807143042530.coc.gml','3518037380208073.coc.gml','3518192234385654.coc.gml','3519083113115752.coc.gml','3519104033490770.coc.gml','3519173332815242.coc.gml','3554653150827755.coc.gml','3558226899367894.coc.gml','3558246365665871.coc.gml','3560576217397500.coc.gml','3581029350321431.coc.gml','3581830525479047.coc.gml','3581866814344587.coc.gml','3581874289297524.coc.gml','3582141898089800.coc.gml','3582182788767024.coc.gml','3582187498347368.coc.gml'],['3343744527348953.coc.gml','3343901805640480.coc.gml','3344631446304834.coc.gml','3345283975088597.coc.gml','3345341063735706.coc.gml','3345672913760585.coc.gml','3345716399866144.coc.gml','3346041476969222.coc.gml','3346361808667289.coc.gml','3347020320429724.coc.gml','3347122272192199.coc.gml','3348202183182981.coc.gml','3363356413828548.coc.gml','3367269376657349.coc.gml','3367472590570390.coc.gml','3367745213249038.coc.gml','3369886157847997.coc.gml','3370187475368354.coc.gml','3370242220657016.coc.gml','3370848283881337.coc.gml','3370943210679558.coc.gml','3371095383919407.coc.gml','3371320634873316.coc.gml','3371353334212131.coc.gml','3429328731908395.coc.gml','3431812342844428.coc.gml','3443510244101746.coc.gml','3464119577576394.coc.gml','3464210090965697.coc.gml','3474593925041592.coc.gml','3477892518336048.coc.gml','3482476628770294.coc.gml','3489084565802342.coc.gml','3489558450803933.coc.gml','3489743314991378.coc.gml','3494152501426914.coc.gml','3494489962794555.coc.gml','3504252389771186.coc.gml','3505779032316582.coc.gml','3506067055046956.coc.gml','3506429741008735.coc.gml','3508035156247306.coc.gml','3511566865123223.coc.gml','3511850572507320.coc.gml','3511950958712857.coc.gml','3511953756712121.coc.gml','3511983850692431.coc.gml','3512027492461885.coc.gml','3512228487367164.coc.gml','3512228747718413.coc.gml','3512320883828568.coc.gml','3512362638453577.coc.gml','3512365087957339.coc.gml','3512371488398141.coc.gml','3512391692548104.coc.gml','3512570026130918.coc.gml','3512593978848674.coc.gml','3512597141818367.coc.gml','3512651290338120.coc.gml','3512681036745319.coc.gml','3512725802294904.coc.gml','3513009873831054.coc.gml','3513027452433705.coc.gml','3513170419502667.coc.gml','3513262479906932.coc.gml','3513361775532449.coc.gml','3513435633981893.coc.gml','3513485109002235.coc.gml','3513524855215093.coc.gml','3513671424206849.coc.gml','3513732346670068.coc.gml','3513739158587782.coc.gml','3513761854134169.coc.gml','3513762864278058.coc.gml','3513784817583701.coc.gml','3513786944578870.coc.gml','3513795827993634.coc.gml','3513822322955028.coc.gml','3513871362653946.coc.gml','3514033757910710.coc.gml','3514058537701986.coc.gml','3514069312670878.coc.gml','3514409529974363.coc.gml','3514484653860068.coc.gml','3514517281419758.coc.gml','3514793086497284.coc.gml','3516368059904172.coc.gml','3516941173798600.coc.gml','3517012276302505.coc.gml','3517263451924815.coc.gml','3517300592201464.coc.gml','3517351528480026.coc.gml','3517374844008722.coc.gml','3517378317183344.coc.gml','3517880442503213.coc.gml','3518018271478014.coc.gml','3518374388216553.coc.gml','3518554643491425.coc.gml','3518876082450868.coc.gml','3518889122334776.coc.gml','3518924249182559.coc.gml','3524708206766693.coc.gml','3553858343177336.coc.gml','3555764297312504.coc.gml','3559702166286240.coc.gml','3560740088421769.coc.gml','3560817721137817.coc.gml','3581880941207958.coc.gml','3584031612169073.coc.gml','3586578800272420.coc.gml'],['3343770315269085.coc.gml','3343828640519374.coc.gml','3344200536278976.coc.gml','3344204856189380.coc.gml','3344251610958634.coc.gml','3344947345156943.coc.gml','3345285264425627.coc.gml','3345943583930879.coc.gml','3346337229522031.coc.gml','3346646386115884.coc.gml','3348968449285798.coc.gml','3365285795421520.coc.gml','3367612479773055.coc.gml','3367696964256940.coc.gml','3369068477131339.coc.gml','3371247520439214.coc.gml','3371270547035409.coc.gml','3376346833399388.coc.gml','3383683287711280.coc.gml','3430490826135856.coc.gml','3431235789559517.coc.gml','3435254721283066.coc.gml','3440833729050905.coc.gml','3466248535071107.coc.gml','3481617253114874.coc.gml','3487580337482630.coc.gml','3488816159775764.coc.gml','3488842327677557.coc.gml','3488968551195859.coc.gml','3489462720299193.coc.gml','3497517021192038.coc.gml','3497540102476487.coc.gml','3504590328512715.coc.gml','3506452277059335.coc.gml','3506978007041225.coc.gml','3507953124535865.coc.gml','3508256699661280.coc.gml','3509097477346600.coc.gml','3509751591473526.coc.gml','3510725307943432.coc.gml','3511585550779014.coc.gml','3511661669904764.coc.gml','3512241854636226.coc.gml','3512265636249516.coc.gml','3512407622499003.coc.gml','3512467252803282.coc.gml','3512633108174568.coc.gml','3512642997758015.coc.gml','3512675969861915.coc.gml','3512846728070462.coc.gml','3513330587596297.coc.gml','3513645327614227.coc.gml','3516311047353690.coc.gml','3524530280557536.coc.gml','3524644067598719.coc.gml','3527628663048707.coc.gml','3528870550951587.coc.gml','3553179003983639.coc.gml','3553712926158978.coc.gml','3557998624957120.coc.gml','3558051988500239.coc.gml','3580448250376461.coc.gml','3581083603782299.coc.gml','3581833155041015.coc.gml','3582675862318360.coc.gml','3590965983991304.coc.gml','3385409596124201.coc.gml','3488298427647276.coc.gml','3503979856274851.coc.gml','3506843546638885.coc.gml','3512409568346880.coc.gml','3518734310070023.coc.gml','3530377388318988.coc.gml','3550786464348915.coc.gml','3552170865080083.coc.gml','3573993774557103.coc.gml','3581619170077512.coc.gml','3591379278050636.coc.gml','3571815701857951.coc.gml']]

    def findfiles(self,dirname,pattern):
        #glob 用通配符查找指定目录中的文件 print(findfiles(r'D:\python\windows-service','*.py'))
        import glob
#         cwd = os.getcwd() #保存当前工作目录
        if dirname:
            os.chdir(dirname)
    
        result = []
        for filename in glob.iglob(pattern): #此处可以用glob.glob(pattern) 返回所有结果
            result.append(filename)
        #恢复工作目录
#         os.chdir(cwd)
        return result
    
    
    def connect_allfileinlist_infolder(self, folder, percent, lista,save=False, saveFolder=None):        
        "IN:folder,filename list"
        "OUT:connected all"
        """testing code:combinefile = ['.average_path_length','.diameter']
    folder = r'G:\HFS\WeiboData\HFSWeiboStatNet\NetCore5p\\'
    # connect_all(folder,0.1,combinefile,save=True, saveFolder=folder+'combine\\')
    for percent in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        gt.connectlistinfolder(folder, percent)"""
        
        vd = []
        for filen in lista:
            if os.path.isfile(folder + '\\'+filen):
                vd2 = gt.selectPercent(gt.csv2list_new(folder + '\\'+filen),percent=percent, percentindex=1)
                vd = gt.connectlist(vd, vd2, 0, 0, 2)
        if save:
            gt.saveList(vd, saveFolder+ 'all_' + str(percent))
        print len(vd[0])
        return vd

    def drawgraph(self,gmlfile,giantornot=False):
        "IN:gmlfile path or graph ; choose to draw just giant or not"
        "OUT:the drawing of this graph"
        import igraph as ig
        from igraph.drawing import plot
        print 'Reading graph...'
        if os.path.isfile(str(gmlfile)):
            g = ig.Graph.Read_GML(gmlfile)
        else:
            g = gmlfile
        if giantornot:
            g = ig.clustering.VertexClustering.giant(g.clusters(mode='weak'))
        
        print 'Ploting graph'
        layout = g.layout('auto')#[gfr,graphopt,kk,lgl,mds,random,rt_circular,star,sugiyama,auto,circle,drl,fr,grid]
#         fig = Plot(g,target='atest.png')#Plot(g, layout = layout)
        plot(g, layout = layout)
#         plot.save(os.path.splitext(gmlfile)[0]+'.png')
        
#         "auto": "layout_auto",
#         "automatic": "layout_auto",
#         "circle": "layout_circle",
#         "circular": "layout_circle",
#         "drl": "layout_drl",
#         "fr": "layout_fruchterman_reingold",
#         "fruchterman_reingold": "layout_fruchterman_reingold",
#         "gfr": "layout_grid_fruchterman_reingold",
#         "graphopt": "layout_graphopt",
#         "grid": "layout_grid",
#         "grid_fr": "layout_grid_fruchterman_reingold",
#         "grid_fruchterman_reingold": "layout_grid_fruchterman_reingold",
#         "kk": "layout_kamada_kawai",
#         "kamada_kawai": "layout_kamada_kawai",
#         "lgl": "layout_lgl",
#         "large": "layout_lgl",
#         "large_graph": "layout_lgl",
#         "mds": "layout_mds",
#         "random": "layout_random",
#         "rt": "layout_reingold_tilford",
#         "tree": "layout_reingold_tilford",
#         "reingold_tilford": "layout_reingold_tilford",
#         "rt_circular": "layout_reingold_tilford_circular",
#         "reingold_tilford_circular": "layout_reingold_tilford_circular",
#         "sphere": "layout_sphere",
#         "spherical": "layout_sphere",
#         "star": "layout_star",
#         "sugiyama": "layout_sugiyama",


    def connectlistinfolder(self,folder,percent):
        vd = []
        for filen in os.listdir(folder):
            if os.path.isfile(folder+filen):
                vd2 = self.selectPercent(self.csv2list_new(folder+filen), percent=percent, percentindex=1)
                vd = self.connectlist(vd, vd2, 0, 0)
        self.saveList(vd,folder+'combine\\all_'+str(percent))
         
        print len(vd[0])
 
    def selectColfromList(self,lista,colindexa,colindexb):
        lista = zip(*(lista))
        return lista[colindexa:colindexb]

    def fun_dis (self,x,y,n):  
        return sum (map (lambda v1,v2:pow(abs(v1-v2),n), x,y))  
     
    def distance (self,x,y):  #adjust
        return self.fun_dis (x,y,2)  
    #     return fun_dis (x,y,1)  
          
     
    def min_dis_center(self,center, node):  
        min_index = 0 
        min_dista = self.distance (center[0][1],node)  
        for i in range (1,len(center)):  
            tmp = self.distance (center[i][1],node)  
            if (tmp < min_dista):  
                min_dista = tmp  
                min_index = i  
        return min_index  

    def k_means (self,info,k=1000):  
    #     print 'kmeans'
        center = [[1,info[i]] for i in range(k)]  
        result = [[i] for i in range(k)]  
        width  = len (info[0])  
        for i in range(k,len(info)):  
            min_center = self.min_dis_center (center,info[i])  
            for j in range(width):  
                center[min_center][1][j] = (center[min_center][1][j] * center[min_center][0] + info[i][j])/ (1.0+center[min_center][0])  
            center[min_center][0] += 1 
            result[min_center].append (i)  
        return result,center 
    
    def k_means_sklearn(self):
        import sklearn as sc

        import numpy as np
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y = np.array([1, 1, 2, 2])
        
        # X = gt.normlistlist(gt.selectPercent(gt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\Net\.vcount',passmetacol=2),1.0,1),metacolcount=0,sumormax='max')
        # print len(X)
        # # X = gt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\FansDupTest.txt',passmetacol=2)
        # y = range(len(X))
        from sklearn.svm import SVC
        clf = SVC()
        clf.fit(X, y) 
        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
            gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
            random_state=None, shrinking=True, tol=0.001, verbose=False)
        print(clf.predict([[-0.8, -1]]))
        # print(clf.predict([[0.2,9295.0,16628.0,35369.0,42537.0,51727.0,56756.0,62501.0,70341.0,79530.0,85276.0,90325.0,99678.0,116775.0,124623.0,159816.0,170709.0,180283.0,190385.0,196247.0,200733.0]]))

    def listDistribution(self,lista,disfigdatafilepath=None,xlabel='Amount',ylabel='Frequency',showfig=True,binsdivide=1):
        "IN:one list of numbers;savefigure data or not;xylabel;show the fig or not"
        "OUT:the distribution data [x,y]"
        hist = numpy.histogram(lista,bins=numpy.max(lista)/binsdivide)#bins=numpy.max(lista)
        print hist
        x=hist[1][:len(hist[1])-1]
        y=hist[0]
        if disfigdatafilepath:
            self.savefigdata(disfigdatafilepath, x, y, errorbarlist=None, title='', xlabel=xlabel, ylabel=ylabel, leglend=None)
        if showfig:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.loglog(x,y,marker='o',linestyle='')#semilogy
            plt.show()
            plt.close()
        return [x,y]
    
    def list_2_Distribution(self,listlist,xlabel='Count',ylabel='Frequence',showfig=True,binseqdiv=0):
        "IN:one list of numbers;savefigure data or not;xylabel;show the fig or not"
        "OUT:the distribution data [x,y]"
        hist = []
        x = []
        y = []
        i = 0
        for lista in listlist:
            lista = map(float,lista)
            
            maxx = numpy.max(lista)
            minx = numpy.min(lista)
            lena = len(lista)
            q = float(maxx-minx)/lena
            
            binseq = None
            if binseqdiv:
                binseq = []
                for i in range(lena/binseqdiv):
                    binseq.append(minx+binseqdiv*i*q)
            
            if binseq:
                hist_1 = numpy.histogram(lista,bins=binseq)#maxx if maxx>1 else 1)
            else:
                hist_1 = numpy.histogram(lista,maxx if maxx>1 else 1)
            hist.append(hist_1)
            print hist_1
            x_1=hist_1[1][:len(hist_1[1])-1]
            y_1=hist_1[0]
            " follow is just prepare for fig4inone"
#             if i>0:
#                 yt_1=[]
#                 for item in y_1:
#                     yt_1.append(item/10.0)
#                 y_1 = yt_1
            
            x.append(x_1)
            y.append(y_1)
            i+=1
    
#         hist_2 = numpy.histogram(listb,bins=numpy.max(listb))
#         print hist_2
#     
#         hist_3 = numpy.histogram(listc,bins=numpy.max(listc))
#         print hist_3
    
#         x_1=hist_1[1][:len(hist_1[1])-1]
#         y_1=hist_1[0]
#     
#         x_2=hist_2[1][:len(hist_2[1])-1]
#         y_2=hist_2[0]
#     
#         x_3=hist_3[1][:len(hist_3[1])-1]
#         y_3=hist_3[0]
        if showfig:
            markerlist = ['o','x','+','o','x','+','o','x','+','o','x','+','o','x','+','o','x','+','o','x','+','o','x','+','o','x','+','o','x','+']
            linestyle = ['solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted',]
            i = 0
            for fig in zip(*(x,y)):
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                
                plt.plot(fig[0],fig[1],marker=list(markerlist)[i],linestyle=linestyle[i])#semilogy#[‘solid’ | ‘dashed’ | ‘dashdot’ | ‘dotted’]
                i+=1
#                 plt.loglog(x_2,y_2,marker='x',linestyle='dashed')#semilogy
#                 plt.loglog(x_3,y_3,marker='+',linestyle='dashdot')#semilogy
            plt.show() 
            plt.close() 
        return [x,y]  
            
    def txt2list(self,infilepath,delimiter=',',metalinecnt=1,lineseprator="#^#~#^#~##\n"):
        repost=[]
    
        inpath=infilepath#dirinpath+str(filestr)
        f=open(inpath)#"G:\\HFS\\Data\\txt\\txt2\\4_3376346833399388.txt"
    
        linecount=0
        for i in range(0,metalinecnt):
            f.readline()
        line=f.readline()
        while line:
            if line.endswith(lineseprator):
                linecount+=1
                line = line.replace(lineseprator,'')
                lines=re.split(r'\',\'',line.replace('\"','').replace('\n','').replace('\',r\'','\',\''))#\",\"#\',\'
                if lines[0].startswith('id'):
                    line=f.readline()
                    continue
                else:
                    try:
                        repost.append(lines)
                    except:
                        pass 
                line=f.readline()
            else:
                line+=f.readline()
       
        f.close()    
        print 'txt2list-linecount: ',linecount,'\t',inpath
        return repost

    def normalizelist(self,lista,sumormax='sum'):
        #IN:list,OUT:guiyihua   sumormax=sum or max
     
        usersumlistNorm = []
        listsum = numpy.sum(lista) if sumormax=='sum' else numpy.max(lista)
        for item in  lista:
            usersumlistNorm.append(float(item)/listsum)  
        return usersumlistNorm

    def normlistlist(self,listlista,metacolcount,sumormax='max'):
        result = []
        for item in listlista:
            resultt = []
            meta = item[0:metacolcount]
            content = self.normalizelist(map(float,item[metacolcount:]),sumormax=sumormax)
            resultt = meta
            resultt.extend(content)
            result.append(resultt)
        return result
                                              
    def divideList(self,seq,lena,lenb):
        "divide a list into 2 random part with out overlap"
        "IN:a list ; 2 part length"
        "OUT:two list with lena,lenb"
        a = random.sample(seq,lena)
        for it in a:
            seq.remove(it)
        b = random.sample(seq,lenb)
        return [a,b]  
      
    def listSum(self,lista):
        r=0.0
        for i in lista:
            try:
                r+=float(i)
            except:
                pass
    #     result=r/len(lista)
        return r#round(result,3)

    def get_time20point(self,time20pointfilepath=r'G:\HFS\WeiboData\Statistics\data4paper\timeseries\.timeline.point_20'):
        time20point = csv.reader(file(time20pointfilepath))
        return time20point
    
#     timelist = gt.connect_list((gt.csv2list(gt.time20pointpath)))#time
    
    def search2(self,a,m): 
        #二分查找：输入排序的list a，查找目标m，返回m所在list的位置 
        low = 0  
        high = len(a) - 1  
        while low<=high:  
            mid = (low + high)/2  
            midval = a[mid]  
      
            if midval<m:  
                low = mid + 1  
            elif midval>m:  
                high = mid-1  
            else:  
                print mid  
                return mid  
        print -1  
        return -1 
    
    def saveList(self,lista,savepath,writype='w'):
        writer = csv.writer(file(savepath,writype))
        for line in lista:
            writer.writerow(line)
 
    def findall_instr(self, strline,str2find, start=0):
        """add support for findall()"""
        "IN:string strline;  string str2find; the start index which default value is 0"
        "OUT:the index list which found"
        body = strline
        result = []        
        while True:
            pos = body.find(str2find, start)
            if pos >= 0:
                result.append(pos)
                start = pos + len(str2find)
                #body = body[pos+len(arg):]
                continue
            break
        return result    
        
    def json2txt(self):
        #json与文本转换
        import json
        f = open('jsonfilepath')
        jsonf = f.read()
        # print jsonObject(jsonf)
        weibo = eval(jsonf)
    
    
    
    
    def writelog_method1(self):
        #######################日志记录#######################3
    
        import sys
        
        temp = sys.stdout
        sys.stdout = open("G:\\HFS\\WeiboData\\startweiboid\\log.txt",'a+')
        print 1,2,3 # 测试，之后可以检查下.server_all 文件
        print 1,2,3 # 测试
        print 1,2,3 # 测试
        print 1,2,3 # 测试
        print 1,2,3 # 测试
        print 1,2,3 # 测试
        print 1,2,3 # 测试
        print 1,2,3 # 测试
        sys.stdout = temp #resotre print
        sys.stdout.close()
    
    
 
    
    def writelog_method2(self): 
        import sys  
          
        stdout = sys.stdout  
        sys.stdout = TextArea()  
          
        # print to TextArea  
        print "testA"  
        print "testB"  
        print "testC"  
          
        text_area, sys.stdout = sys.stdout, stdout  
          
        # print to console  
        print text_area.buffer
        
   
    def getfilelistin_folder(self,path,filetype='.reppost'):
        result = []
        filelist=''
        firstpart='@'#r'delete from Data_Repost_Temp;\ndelete from Data_Comment_Temp;\ndelete from data_chuanbococ_temp;\nLOAD DATA INFILE "G:\\HFS\\WeiboData\\HFSWeibo\\'
        sedpart='$'#r'''.repost" INTO TABLE `Data_Repost_Temp` Fields Terminated By ',' Enclosed By '\'' Lines Terminated By '#\^#~#\^#~##\r\n' IGNORE 1 lines;\nLOAD DATA INFILE "G:\\HFS\\WeiboData\\HFSWeibo\\'''
        thirdpart='%'#r'''.comment" INTO TABLE `Data_Comment_Temp` Fields Terminated By ',' Enclosed By '\'' Lines Terminated By '#\^#~#\^#~##\r\n' IGNORE 1 lines;\nLOAD DATA INFILE "G:\\HFS\\WeiboData\\eventTXTout\\HFSWeibo\\'''
        fourthprt='&'#r'''.coc" INTO TABLE `data_chuanbococ_temp`;\ncall stat();\n'''
        filename='45768'
#         print firstpart+filename+sedpart+filename+thirdpart+filename+fourthprt
        for filename in os.listdir(path):
            filepre=os.path.splitext(filename)
            if filepre[1] == filetype:
#                 print filename
                result.append(filename)
                filelist+=firstpart+filename+sedpart+filename+thirdpart+filename+fourthprt+'\n'
                
        return result#filelist
    
        f=open(r'G:\HFS\WeiboData\NewCrawper\20130622.filename','w')
        f.write(self.getfilelistin_folder(r'G:\HFS\WeiboData\NewCrawper\20130622'))
        f.close()
     
    def listdivide(self,lista,periodcnt):
        "divide a list into periods averages"
        "IN:list,how many periods to be divide"
        "OUT:the index of list divided"
        lenyt = len(lista)/float(periodcnt)
    #     leny = lenyt if lenyt>1 else 1
        y = []
        for j in range(1,periodcnt+1):
            i = int(round(j*lenyt))
            i = i if i<len(lista) else len(lista)-1
            i = i if i>1 else 1
            y.append(i)
        return y 
     
    def select308(self):
        sucmeta_file = r'G:\HFS\WeiboData\Statistics\meta_successed.txt'
        sucmetaw_file = r'G:\HFS\WeiboData\Statistics\meta_successed308.txt'
        f=open(sucmeta_file)
        fw=open(sucmetaw_file,'a')
        line=f.readline()
        cnt = 0
        while line:
            print line
            try:
                linelist = line.split('\t')
                if linelist[3] == '-1':
                    fw.write(str(line))
                    cnt += 1
            except:
                pass
            line=f.readline()
        print cnt
        fw.close()  
        
    def drawmeta_successed308png(self):
        import matplotlib.pyplot as plt
        sucmeta_file = r'G:\HFS\WeiboData\Statistics\data4paper\kmeans_3_list'
        f=open(sucmeta_file)
        line=f.readline()
        type = 0
        x=range(1)
        colorlist = list('bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk')#['r','g','b',]
        while line:
            if str(line).startswith('kmeans'):
                type+=1
            try:
                if type/3==0:
                    print type
                    linelist = line.split('\t')[2:22]
            #         fig = plt.semilogy(x,linelist,color=colorlist[type],marker='o')
                    fig = plt.plot(x,linelist,color=colorlist[type],marker='o')        
            except:
                pass
            line=f.readline()
             
        plt.show()
        f.close() 
        
    def savefigdata(self,datafilepath,x,y,errorbarlist=None,title='title',xlabel='',ylabel='',leglend=''):
        f = open(datafilepath,'w')
        writer = csv.writer(f)
        writer.writerow([datafilepath,str(x),str(y),str(errorbarlist),title,xlabel,ylabel,leglend])
#         f.writelines([datafilepath,str(x),str(y),str(errorbarlist),title,xlabel,ylabel,leglend])  
        f.close() 


    def selectPercent(self,lista,percent,percentindex=1):
        result = []
        for item in lista:
            if str(item[percentindex])==str(percent):
                result.append(item)
        return result
         
    def repairline(self,csvfile,seprator):
    #     return csvfile.readline().replace('\n','').replace(' ','')
        return csvfile.readline().replace(seprator+'\n','').replace('\n','')#.replace(' ','').replace(seprator+'\n'+seprator,'')

    def csv2list_new(self,csvfilepath,delimitertype='excel',passmetacol=0,convertype=str,nan2num=False,passmetarowcnt=0):
        #IN:csv format file path; the csvdialect fromat(such as 'excel'=>',',''excel_tab'=>'\t')
        #OUT:the list of the csv file
        csvlist = []
        csvfile = open(csvfilepath)
        lista = csv.reader(csvfile,dialect=delimitertype)
        linecnt = 0
        for line in lista:
            linecnt+=1
            if linecnt<passmetarowcnt+1:
                pass
            else:
                linew = map(convertype,line[passmetacol:])
                if nan2num:
                    linew = list(numpy.nan_to_num(linew))
                if passmetacol>0:
                    linewtemp = line[:passmetacol]
                    linewtemp.extend(linew)
                    linew = linewtemp
    
                csvlist.append(linew)
        csvfile.close()
        return csvlist  

    def csv2list(self,csvfilepath,seprator='\t',start_index=0):
#         print 'convert the file:',csvfilepath
        csvfile = open(csvfilepath)
        resultlistemp = []
        line=self.repairline(csvfile,seprator)
        if line=='':
            line=self.repairline(csvfile,seprator) 
        i = 0    
        while line:
            linelist = line.split(seprator)
            resultlistemp.append(linelist[start_index:])
    #         datalistemp.append(map(float,datalist[i][1:]))
        #     datalist.insert(0,linelist[0])    
            line=self.repairline(csvfile,seprator)
            i+=1
        csvfile.close()
        return resultlistemp  
    
    def list2dict(self,l):
        ret={}
        for it in l:
#             it = str(it).replace('.coc','').replace('.gml','')
            ret[str(it[0])] = it
        return ret 
     
    def connectlist(self,lista,belista,sameposition_a=0,sameposition_be=0,passcol=1,listakey_suffix=None,belistakey_suffix=None):
        # 'input:欲连接的两个list，连接条件所在的索引位置'
        # 'out：连接好的list，belista在后 '
        listout=[]
        belistadic = self.list2dict(lista)
        suclistadic = self.list2dict(belista)
        for it in suclistadic.iterkeys():
            belistakey = it
            if belistakey_suffix:
               belistakey = str(it).replace(str(belistakey_suffix),'') 
#             print it
            i = suclistadic.get(it)
            j = belistadic.get(belistakey)
            
            if i and j:#i[sameposition_a].split('.')[0]==j[sameposition_be].split('.')[0]:
#                 if len(j)<len(belista[2]):#避免空行，但以第三行为标准了，必须保证第三行正确
#                     print 'error line when connectlist:',j
#                     break
                i.extend(j[passcol:])
                listout.append(i)
#                 break   
#         for i in lista:
#             for j in belista:
#                 
#                 if i[sameposition_a].split('.')[0]==j[sameposition_be].split('.')[0]:
#                     if len(j)<len(belista[2]):#避免空行，但以第三行为标准了，必须保证第三行正确
#                         print 'error line when connectlist:',j
#                         break
#                     i.extend(j[passcol:])
#                     listout.append(i)
#                     break
#         import operator
#         listout.sort(key=operator.itemgetter(1))
        return listout if listout else belista 
      
    def connectlist_sf(self,belista,sameposition_be=0,suclista=[],rangemin=0,rangemax=10000000,sorted=True,keysuffix=None):
        # 'input:欲连接的两个list，连接条件所在的索引位置'
        # 'out：连接好的list，belista在后 '
        listout=[] 
        if suclista==[]:
            suclista=self.csv2list(r'G:\HFS\WeiboData\Statistics\meta_successed308.txt')  
#         for i in suclista:
#             for j in belista:
#                 if str(i[0])==str(j[0]) and int(i[4]) in range(rangemin,rangemax):#[sameposition_be].split('.')[0]:#.repost
#     #                     j.insert(1,str(i[3]).replace(' ','')+'_'+str(j[0]).replace(' ',''))
#                     j.insert(1,str(i[3]).replace(' ',''))
# #                     j.insert(2,str(j[0]).replace(' ',''))
# #                     j=j[1:]
#                     #print j
#                     listout.append(j)
#                     break
                

        belistadic = self.list2dict(belista)
        suclistadic = self.list2dict(suclista)
        for it in suclistadic.iterkeys():
    #             print it
            i = suclistadic.get(it)
            if keysuffix:
                it = str(it)+keysuffix
#             it = str(it).split('.')[0]

            j = belistadic.get(it)
            if j and int(i[4]) in range(rangemin,rangemax):
                j.insert(1,str(i[3]).replace(' ',''))
                listout.append(j)
    #                 break
        if sorted:
            import operator
            listout.sort(key=operator.itemgetter(1))
    #         listout.sort(key=lambda x:x[1])
    #         listout.sort(cmp=lambda x,y:cmp(x[1],y[1]))
    #         listout = [(x[1],i,x) for i,x in enumerate(listout)] #i can confirm the stable sort
    #         listout.sort()
    #         listout = [s[0] for s in listout]
#             print listout
            listout.reverse()
        return listout
    
    def repairY(self,listy,n):
        m = len(listy)
        for i in range(m,n,1):
            listy.append(listy[len(listy)-1])
        return listy   
     
    def get_timepoint_byaverageusercnt(self,periodcnt=20,timelistfile = r"G:\HFS\WeiboData\Statistics\data4paper\timeseries\.timeline"):   
          
        reader = csv.reader(file(timelistfile,'rb'))
        writer = csv.writer(file(timelistfile+'.point_'+str(periodcnt),'w'))
        for line in reader:
            line_metapart = line[0:3]
            line_timepart = line[3:]
            
            lenyt = int(len(line_timepart)/periodcnt)
            leny = lenyt if lenyt>1 else 1
            timey = []
            timex = range(leny,len(line_timepart),leny)
            for i in timex[0:periodcnt-2]:
                timey.append(line_timepart[i])
            s = len(line)
            timey.append(str(line[s-1]))
            timey.reverse() 
            if len(timey)<periodcnt:
                timey = self.repairY(timey,periodcnt)
            line_metapart.extend(timey)
            writer.writerow(line_metapart)

    def get_distinct_inlist(self,S):
        return {}.fromkeys(S).keys()
    
    def findtimelist(self,indexstr,metatimelist):
        timel = []
        for line in metatimelist:
            if str(line[0].split('.')[0])==str(indexstr):
                timel = line
                break
        return timel


    def del_min_oneline(self,line,startindex=0,minindex=0):
        minlinew=line
        try:
            line_part = map(float,line[startindex:])
            min = float(line_part[0])
            if minindex!=0:
                min = numpy.min(line_part)
            minlinew = []                
            for it in line_part:
                minlinew.append(it-min)
            minlinew.insert(0,line[0])
        except:
            print 'error in del_min_oneline(self,line,startindex=0,minindex=0)',line
        return minlinew

                
    def del_min_list(self,lista,startindex=0,minindex=0):
        min = 0
        minlist = []
        for line in lista:
            minlinew = self.del_min_oneline(line)
            minlist.append(minlinew)
        return minlist

    def departlist(self,listb,s,f,sfposition,passn=0):
        #IN:list;OUT:将list按sfposition对应的值以条件s，f分割开分别作为list的元素加入  ,忽略前passn项 
        lists = []
        listf = []
        listsf = []
        for item in listb:
            if item[sfposition] == s:
                lists.append(item[passn:])
            if item[sfposition] == f:
                listf.append(item[passn:])
        listsf.append(lists)
        listsf.append(listf)
        return listsf
    
    def averageLongtitude(self,lista):
        #IN:list matrix;OUT:longtitude average
        averageresults = []
        stdresults = []
        for item in zip(*lista):
            item = list(item)
            try:
                itemn = []
                for it in item:
                    if it=='nan' or it=='inf':
                        pass
                    else:
                        itemn.append(it)
            except:
                pass

            try:           
                averageresults.append(numpy.average(map(float,itemn)))
                stdresults.append(numpy.std(map(float,itemn)))#/len(item)
            except:
                print  'error:',item 
        return [averageresults,stdresults]
    
    def fansdraw(self,yf,sucmeta_list,figposition,diroutpath,sfpositonx=0,passline=2):    
        sflist = self.connectlist(sucmeta_list,yf,0,0)            
        sflistall = self.departlist(sflist,'1','-1',sfpositonx,passline)#若为未clear的为4
        slist = sflistall[0]#sflist_norm(sflistall[0])#视情况决定是否需要归一化
        flist = sflistall[1]#sflist_norm(sflistall[1])#
    #             print sflistall
        averageresults = self.averageLongtitude(slist)[0]
        averageresultf = self.averageLongtitude(flist)[0]
        print slist
        print 'averageresults:',averageresults,'\naverageresultf',averageresultf
        stdresults = None
        stdresultf = None
    #     if has_yerr:
    #         stdresults = averageLongtitude(slist)[1]
    #         stdresultf = averageLongtitude(flist)[1]
    
        xlabel = 'S-'+str(len(slist))+':F-'+str(len(flist))
        ylabel = 'fans'
        self.savefigdata(diroutpath+'fans.figdata',range(len(averageresults)),averageresults,errorbarlist=stdresults,title=None,xlabel=xlabel,ylabel=ylabel,leglend='S')
        self.savefigdata(diroutpath+'fans.figdata',range(len(averageresultf)),averageresultf,errorbarlist=stdresultf,title=None,xlabel=xlabel,ylabel=ylabel,leglend='F')
        import matplotlib.pyplot as plt
        plt.subplot(1,3,figposition)
        plt.plot(range(len(averageresults)),averageresults,marker='o',color='r')
        plt.plot(range(len(averageresultf)),averageresultf,marker='x',color='b')
        plt.legend()
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)    
        plt.legend(('S','F'),loc='upper left')
        
    def txt2libsvmformat(self,filepathin,filepathout,columlist,signposition):
        #inout:文本文件输入出位置，列名list, 标签位置列名
        #output：svm格式的文本
        f=open(filepathin)
        fw=open(filepathout,'a+')
        metaline=f.readline().replace('\n','')
        metalist=metaline.split('\t')
        signp=metalist.index(str(signposition))
        metacolist=[]
        for item in columlist:
            metacolist.append(metalist.index(str(item.strip())))
        print metacolist
        
        line=f.readline()
        while line:
            linelist=line.replace('\n','').split('\t')
            newline=''
            i=0
            for item in metacolist:
                newline+=str(metacolist[i])+':'+linelist[item]+'\t'
                i+=1
            svmline = linelist[signp].replace('N','-1').replace('Y','1')+'\t'+newline
            print svmline
            fw.write(svmline+'\n')
            line=f.readline()

    def csv2libsvmformat(self,filepathin,csvdialect='excel-tab',passlinecnt=1,metacolindex=0,flagindex=1):
        #IN:the file path formated by csv, the sepator is the csvdialect
        #out:write the file suffixed with 'svm' formated by libsvm, also it return the list of this
        fw = open(filepathin+'.svm','w')
        writer = csv.writer(fw,dialect='excel-tab')
        reader = csv.reader(file(filepathin),dialect=csvdialect)
        svmlist = []
            
        for line in reader:
            i = 0
            svmlistline =  []
            for item in line[2:]:
                i+=1
                itemnew = str(i)+':'+str(item)
                svmlistline.append(itemnew)
            svmlistline.insert(0,line[metacolindex])
            svmlistline.insert(1,line[flagindex])
            writer.writerow(svmlistline)
        
            svmlist.append(svmlistline)
        return svmlist  
     
    def get_startweibo(self):#yici
        fp = r'G:\HFS\WeiboData\Statistics\MySqlExport\startweibo_all.txt'
        a = self.csv2list(fp)
        meta_suclist = self.connectlist_sf(a)
        fpw = open(fp+'.metasuc','w')
        writer= csv.writer(fpw)
        for line in meta_suclist:
            writer.writerow(line) 
        fpw.close()    

# a=commontools()
# a.get_timepoint_byaverageusercnt(20)

# r = r'G:\HFS\WeiboData\HFSWeiboGML\SVG\3373235549224874.gml_strongGaint.gml'
# print os.path.splitext(r)[0]
# a.drawgraph(r)