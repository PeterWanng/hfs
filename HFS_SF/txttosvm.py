#coding=utf8

 
#inout:文本文件输入出位置，列名list, 标签位置列名
#output：svm格式的文本
from tools import commontools as gtf
import random
import os  
import numpy
gt = gtf()

class txt2svm():
    def list2libsvmformat(self,lista,percindex=2):
        listResult = ''
        for item in lista:
            i = 0
            litem = str(item[1])+'\t'#item[0]+'\t'+
            for it in item[percindex+1:]:
                i+=1
                litem+=(str(i)+':'+str(it))+'\t'
                listResult+=litem
                litem = ''
            listResult+='\n' 
        return listResult.replace('\t\n','\n')


    
    def txt2libsvmformat(self,filepathin,filepathout,columlist,signposition):
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
#             fw.write(svmline+'\n')
            line=f.readline()
        
    def selectPercent(self,lista,percent,percentindex=1):
        result = []
        for item in lista:
            if str(item[percentindex])==str(percent):
                result.append(item)
        return result


    def random_sample(self,listslen,listflen,sucnt,failcnt):
        success_list=range(1,sucnt+1)
        failture_list=range(sucnt+1,sucnt+failcnt)
        
        success_sample = random.sample(success_list,listslen)
        failture_sample = random.sample(failture_list,listflen)
        
        sample_list=success_sample+failture_sample
        sample_list.sort()
        return sample_list
    
    #########
    def sampline(self,svmfile,svmoutfile,lista,listb,listc,listd):
                
        f = open(svmfile)
        fw1 = open(svmoutfile+'.tr','w')
        fw2 = open(svmoutfile+'.ts','w')
        lincnt = 0
        for line in f:
            lincnt+=1
            if lincnt in lista:
                fw1.write(line)
                continue
            if lincnt in listb:
                fw2.write(line)
                continue
            if lincnt in listc:
                fw1.write(line)
                continue
            if lincnt in listd:
                fw2.write(line)
                continue
            
        
        
    def sample2(self,svmfilepath,listsindex,listfindex,fratio,sratio,tssratio,tsfratio,svmoutfile):
        sucnt = len(listsindex)
        failcnt = len(listfindex)
        
        scnt=int(round(sucnt*sratio))
        fcnt=scnt#int(round(failcnt*fratio))
        stscnt=int(round(sucnt*tssratio-0.01))
        ftscnt=stscnt#int(round(failcnt*tsfratio-0.01))
#         print '===============',scnt, stscnt
        s = gt.divideList(listsindex, scnt, stscnt)
        f = gt.divideList(listfindex, fcnt, ftscnt)
     
        self.sampline(svmfilepath,svmoutfile,s[0],s[1],f[0],f[1])
            
        return [s,f]
        
    def sample(self,svmfilepath,sucnt,failcnt,fratio,sratio,tssratio,tsfratio,svmoutfile):
        scnt=int(sucnt*sratio)
        fcnt=int(failcnt*fratio)
        stscnt=int(sucnt*tssratio)
        ftscnt=int(failcnt*tsfratio)
        
        
        filetr='.tr'#'.'+str(scnt)+str(fcnt)+'.tr'
        filets='.ts'#'.'+str(scnt)+str(fcnt)+'.ts'
        print scnt,fcnt
        if os.path.isfile(svmfilepath):        
            f=open(svmfilepath)
            outfile = svmoutfile
            fw=open(outfile+filetr,'w+')
            fw2=open(outfile+filets,'w+')
            samplelist = self.random_sample(scnt,fcnt,sucnt,failcnt)
            linecnt=0
            sn=int(tssratio*sucnt)+1
            fn=int(tsfratio*failcnt)+1
    #         print samplelist
            line=f.readline()
            ll=''
            while line:
                linecnt+=1
                if linecnt in samplelist:
                    ll += str(linecnt)+','
                    fw.write(line)
                else:
                    if random.random()<tssratio and line.startswith('1'):
                        sn+=1
                        fw2.write(line)
                    if random.random()<tsfratio and line.startswith('-1'):
                        fn+=1
                        fw2.write(line)
                line=f.readline()
            print sn,fn  
            print outfile+filetr,'  finished'
            f.close()
            fw.close()    
            fw2.close() 

a=txt2svm()
def getsindex(sflist,flag):
    result=[]
#     f = gt.csv2list(filep)
    linecnt = 0
    flagindex = 0
    for line in sflist:      
        linecnt+=1
        if line[1]==flag and ~flagindex:
            flagindex = linecnt
            break
        else:
            pass  
    
    result.append(flagindex)        
    result.append(len(sflist))        
            
    return result

def svmfeatures(vd,filename,rangemin,rangemax):
    vdsf = gt.connectlist_sf(vd, sameposition_be=0,suclista=gt.csv2list(r'G:\HFS\WeiboData\Statistics\meta_successed308.txt'),rangemin=rangemin,rangemax=rangemax)
    

    vdsvm = a.list2libsvmformat(vdsf)
    svmfilepath = r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\vd'+filename+'_km.svm'
    
    fw = open(svmfilepath,'w')
    fw.write(vdsvm)
    fw.close()
    for i in range(1,11):
        svmoutfilepath = r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\vd'+filename+'_km_'+str(i)+'.svm' 
        sfindex = getsindex(vdsf,'-1')
        scnt = sfindex[0]
        fcnt = sfindex[1]
        a.sample2(svmfilepath,listsindex=range(1,scnt),listfindex=range(scnt,fcnt),fratio=(0.9*scnt)/(fcnt-scnt),sratio=(0.9*scnt)/scnt,tssratio=(0.1*scnt)/scnt,tsfratio=(0.1*scnt)/(fcnt-scnt),svmoutfile=svmoutfilepath)
#         a.sample2(svmfilepath,listsindex=range(1,scnt),listfindex=range(scnt,fcnt),fratio=44/262.0,sratio=44/48.0,tssratio=4/48.0,tsfratio=4/262.0,svmoutfile=svmoutfilepath)
    print '=============================',scnt,fcnt

def generateSH(foldername=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm',percent='all'):
    tr = gt.getfilelistin_folder(foldername,filetype='.tr')   
    ts = gt.getfilelistin_folder(foldername,filetype='.ts')
    
    pa = "python easy.py" 
    pb = "/home/p/workspace/MyData/svm/svm/"
    pc = "/home/p/workspace/MyData/svm/svm/"
    
    fw = open(foldername+'\\shell4all'+str(percent)+'.sh','w')
    flag = True
    if percent=='all':
        flag = False
    
    fw.write('echo \'Starting=================================================================\'\r')
    for r,s in zip(*(tr,ts)):
        if flag and str(r).startswith('vd'+str(percent)):
            line = str(pa+' '+pb+r+' '+pc+s)
#             print line
            fw.write(line+'\r')
        else:     
            line = str(pa+' '+pb+r+' '+pc+s)
            fw.write(line+'\r')
    fw.write('echo \'Finished=================================================================\'\r')

def generateBAT(foldername=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm',percent='all',predictfilename='predict_none.txt'):
    batpath = foldername+'\\shell4all'+str(percent)+'.bat'
    tr = gt.getfilelistin_folder(foldername,filetype='.tr')   
    ts = gt.getfilelistin_folder(foldername,filetype='.ts')
    
    pa = "python C:\\Python27\\LibSvm\\libsvm-3.17\\tools\\easy.py" 
    pb = "G:\\HFS\\WeiboData\\HFSWeiboStatNet\\Svm\\"
    pc = "G:\\HFS\\WeiboData\\HFSWeiboStatNet\\Svm\\"
    pd = r" >>G:\HFS\WeiboData\HFSWeiboStatNet\Svm\results\\"+predictfilename
    fw = open(batpath,'w')
    flag = True
    if percent=='all':
        flag = False
    
    fw.write('echo \'Starting=================================================================\'\n')
    for r,s in zip(*(tr,ts)):
        if flag and str(r).startswith('vd'+str(percent)):
            line = str(pa+' '+pb+r+' '+pc+s+pd)
#             print line
            fw.write(line+'\n')
        else:     
            line = str(pa+' '+pb+r+' '+pc+s+pd)
            fw.write(line+'\n')
    fw.write('echo \'Finished=================================================================\'\n')
    return batpath

def deal_file_libsvmpredict(predfilepath):
    result = []
    f = open(predfilepath)
#     lines = f.readlines()
    while 1:
        try:
            line = [f.readline(),f.readline(),f.readline()]
            if line[2]=='':
                break
            model = line[2].split(':')[1].replace(' ','').replace('.svm.tr.model\n','')
            precision = line[0].split('=')[1].replace(' ','').split('%')[0]
            precision = float(precision)/100
            print [model,precision]
    
            result.append([model,precision])
        except:
            pass
    return result

def delcoc(lista,delstr,colindex):
    result=[]
    for item in lista:
        item0 = item[colindex].replace(delstr,'')
        it = item[1:]
        it.insert(0,item0)
        result.append(it)
    return result

#     listb = zip(*(lista))
#     col = []
#     for it in listb[colindex]:
#         re = it.replace(delstr,'')
#         col.append(re)
#     listb = listb[1:]
#     listb.insert(0,col)
#     return (zip(*(listb)))
                
"==================================================================================================================="
# predictR = deal_file_libsvmpredict(r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\results\predict.txt')
# predictRZ = zip(*(predictR))
# x = numpy.average(predictRZ[1])
# print x,predictRZ
# avgapl = []
# avglmw = []
# avgad = []
# avglms = []
# avgaid = []
# avgaod = []
# avgden = []
# avgdia = []
# avge2v = []
# avgtra = []
# avgv = []
# avgall = []
# for item in predictR:
#     if item[0].find('average_path_length')!=-1:
#         avgapl.append(item[1])
#     if item[0].find('lenclustersmodeisweak')!=-1:
#         avglmw.append(item[1])
#     if item[0].find('.lenclustersmodeisstrong')!=-1:
#         avglms.append(item[1])
#     if item[0].find('.assortativitydegreedirectedistrue')!=-1:
#         avgad.append(item[1])
#     if item[0].find('.assortativityindegreedirectedistrue')!=-1:
#         avgaid.append(item[1])
#     if item[0].find('.assortativityoutdegreedirectedistrue')!=-1:
#         avgaod.append(item[1])
#     if item[0].find('.density')!=-1:
#         avgden.append(item[1])
#     if item[0].find('.diamter')!=-1:
#         avgdia.append(item[1])
#     if item[0].find('.ecount2vcount')!=-1:
#         avge2v.append(item[1])
#     if item[0].find('.transitivity_undirectedmodeis0')!=-1:
#         avgtra.append(item[1])
#     if item[0].find('.vcount')!=-1:
#         avgv.append(item[1])
#     if item[0].find('all')!=-1:
#         avgall.append(item[1])
# #     if item[0].find('dia_apl')!=-1:
# #         avgall.append(item[1])
#     if item[0].find('TimeSeries_orgin')!=-1:
#         avgall.append(item[1])
#         
# 
# print numpy.average(avgapl)
# print numpy.average(avglmw)
# print numpy.average(avgad)
# print numpy.average(avglms)
# print numpy.average(avgaid)
# print numpy.average(avgaod)
# print numpy.average(avgden)
# print numpy.average(avgdia)
# print numpy.average(avge2v)
# print numpy.average(avgtra)
# print numpy.average(avgv)
# print 'fcuk',avgall
# print 'fcuk',numpy.average(avgall)

# gt = gtf()
# cocfolder = 'G:\\HFS\\WeiboData\\eventTXTout\\HFSWeibo\\'
# w = csv.writer(file('G:\\timelist.txt','w'))
# 
# for cocfilename in os.listdir(cocfolder):
#     cocfilepath=cocfolder+cocfilename
#     
#     vfg = gt.csv2list(cocfolder+cocfilename)
#     vfg.reverse()
#     timelist = gt.selectColfromList(vfg, 4, 5)
#     x = list(timelist[0])
#     x.insert(0,cocfilename)
#     w.writerow((x))
    
def allmain(percent):
    perc = percent
    rangemin=0
    rangemax=100000
    vd = gt.normlistlist(a.selectPercent(gt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\Net\.vcount'),perc,1),metacolcount=2,sumormax='max')
    # vd = gt.connectlist(vd, aspllist, 0, 0, passcol=2)
    featurep = ''
    for filep in os.listdir(r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\IN'):
        filepath = 'G:\\HFS\\WeiboData\\HFSWeiboStatNet\\Svm\\IN'+'\\'+filep
        if os.path.isfile(filepath):
            feature = gt.normlistlist(a.selectPercent(gt.csv2list_new(filepath),perc,1),metacolcount=2,sumormax='max')#gt.csv2list_new(filepath)#
            svmfeatures(vd=feature,filename=str(perc)+'_'+filep,rangemin=rangemin,rangemax=rangemax)
            vd = gt.connectlist(vd, feature, 0, 0, passcol=2)
            print 'finished one:',filep
            featurep+='_'+filep
     
    svmfeatures(vd,filename='all'+str(perc),rangemin=rangemin,rangemax=rangemax)#featurep)
#     generateSH(foldername=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm',percent=str(perc))
    generateBAT(foldername=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm',percent=str(perc))
    
    print gt.getfilelistin_folder(r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm')

def allmain2(percent):
    perc = percent
    rangemin=0
    rangemax=100000
    
    # vd = gt.connectlist(vd, aspllist, 0, 0, passcol=2)
    featurep = ''
    k = 0
    gt.kmeans3group = [['3344178035788881.coc.gml','3344605319892924.coc.gml','3356800646950624.coc.gml','3356881155164816.coc.gml','3358716283896811.coc.gml','3368776344558652.coc.gml','3369168951009868.coc.gml','3369278306978444.coc.gml','3370126999415642.coc.gml','3455798066008083.coc.gml','3464345767606270.coc.gml','3464705244725034.coc.gml','3464854910351381.coc.gml','3479304597792637.coc.gml','3489137279411789.coc.gml','3502012958680889.coc.gml','3506858382217257.coc.gml','3507543877973721.coc.gml','3507607539020444.coc.gml','3507662178094930.coc.gml','3507671015760502.coc.gml','3508278808120380.coc.gml','3510108007340190.coc.gml','3510150776647546.coc.gml','3510947052234805.coc.gml','3511312581651670.coc.gml','3512192651209611.coc.gml','3512207839497881.coc.gml','3512220862117440.coc.gml','3512224943485819.coc.gml','3512225370844164.coc.gml','3512260125547589.coc.gml','3512261920248384.coc.gml','3512288331952105.coc.gml','3512343789230980.coc.gml','3512346909867070.coc.gml','3512367365458914.coc.gml','3512387527089684.coc.gml','3512557425820703.coc.gml','3512564992138128.coc.gml','3512568150742963.coc.gml','3512586882317341.coc.gml','3512598488183675.coc.gml','3512631170591260.coc.gml','3512638635619787.coc.gml','3512649558390590.coc.gml','3512661431797880.coc.gml','3512665513423714.coc.gml','3512673221800564.coc.gml','3512681942436390.coc.gml','3512693292397369.coc.gml','3512703459106709.coc.gml','3512704620407286.coc.gml','3512722819965166.coc.gml','3512723826282658.coc.gml','3512731699677616.coc.gml','3512751521888667.coc.gml','3512755191392034.coc.gml','3512764419413627.coc.gml','3512767539668338.coc.gml','3512944329886751.coc.gml','3512956526933221.coc.gml','3512965133226559.coc.gml','3513008209201946.coc.gml','3513054618763335.coc.gml','3513299721572710.coc.gml','3513353957580369.coc.gml','3513457897170153.coc.gml','3513472585606907.coc.gml','3513477123020136.coc.gml','3513651849587189.coc.gml','3513665519425522.coc.gml','3513684632475119.coc.gml','3513732681977292.coc.gml','3513733382475662.coc.gml','3513737430369962.coc.gml','3513738009766461.coc.gml','3513747752676493.coc.gml','3513797614859184.coc.gml','3513821030864621.coc.gml','3514047335033747.coc.gml','3514054737834781.coc.gml','3514061079292261.coc.gml','3514074715123866.coc.gml','3514083762166772.coc.gml','3514112790871598.coc.gml','3514202721379789.coc.gml','3514207033581502.coc.gml','3514216143529145.coc.gml','3514229367981237.coc.gml','3514287944897392.coc.gml','3514415834044554.coc.gml','3514416295764354.coc.gml','3514448201592653.coc.gml','3514574454119360.coc.gml','3514712136139677.coc.gml','3514721241880789.coc.gml','3514725432613389.coc.gml','3516665499181717.coc.gml','3516669001647040.coc.gml','3517122988859030.coc.gml','3517807143042530.coc.gml','3518037380208073.coc.gml','3518192234385654.coc.gml','3519083113115752.coc.gml','3519104033490770.coc.gml','3519173332815242.coc.gml','3554653150827755.coc.gml','3558226899367894.coc.gml','3558246365665871.coc.gml','3560576217397500.coc.gml','3581029350321431.coc.gml','3581830525479047.coc.gml','3581866814344587.coc.gml','3581874289297524.coc.gml','3582141898089800.coc.gml','3582182788767024.coc.gml','3582187498347368.coc.gml','3343744527348953.coc.gml','3343901805640480.coc.gml','3344631446304834.coc.gml','3345283975088597.coc.gml','3345341063735706.coc.gml','3345672913760585.coc.gml','3345716399866144.coc.gml','3346041476969222.coc.gml','3346361808667289.coc.gml','3347020320429724.coc.gml','3347122272192199.coc.gml','3348202183182981.coc.gml','3363356413828548.coc.gml','3367269376657349.coc.gml','3367472590570390.coc.gml','3367745213249038.coc.gml','3369886157847997.coc.gml','3370187475368354.coc.gml','3370242220657016.coc.gml','3370848283881337.coc.gml','3370943210679558.coc.gml','3371095383919407.coc.gml','3371320634873316.coc.gml','3371353334212131.coc.gml','3429328731908395.coc.gml','3431812342844428.coc.gml','3443510244101746.coc.gml','3464119577576394.coc.gml','3464210090965697.coc.gml','3474593925041592.coc.gml','3477892518336048.coc.gml','3482476628770294.coc.gml','3489084565802342.coc.gml','3489558450803933.coc.gml','3489743314991378.coc.gml','3494152501426914.coc.gml','3494489962794555.coc.gml','3504252389771186.coc.gml','3505779032316582.coc.gml','3506067055046956.coc.gml','3506429741008735.coc.gml','3508035156247306.coc.gml','3511566865123223.coc.gml','3511850572507320.coc.gml','3511950958712857.coc.gml','3511953756712121.coc.gml','3511983850692431.coc.gml','3512027492461885.coc.gml','3512228487367164.coc.gml','3512228747718413.coc.gml','3512320883828568.coc.gml','3512362638453577.coc.gml','3512365087957339.coc.gml','3512371488398141.coc.gml','3512391692548104.coc.gml','3512570026130918.coc.gml','3512593978848674.coc.gml','3512597141818367.coc.gml','3512651290338120.coc.gml','3512681036745319.coc.gml','3512725802294904.coc.gml','3513009873831054.coc.gml','3513027452433705.coc.gml','3513170419502667.coc.gml','3513262479906932.coc.gml','3513361775532449.coc.gml','3513435633981893.coc.gml','3513485109002235.coc.gml','3513524855215093.coc.gml','3513671424206849.coc.gml','3513732346670068.coc.gml','3513739158587782.coc.gml','3513761854134169.coc.gml','3513762864278058.coc.gml','3513784817583701.coc.gml','3513786944578870.coc.gml','3513795827993634.coc.gml','3513822322955028.coc.gml','3513871362653946.coc.gml','3514033757910710.coc.gml','3514058537701986.coc.gml','3514069312670878.coc.gml','3514409529974363.coc.gml','3514484653860068.coc.gml','3514517281419758.coc.gml','3514793086497284.coc.gml','3516368059904172.coc.gml','3516941173798600.coc.gml','3517012276302505.coc.gml','3517263451924815.coc.gml','3517300592201464.coc.gml','3517351528480026.coc.gml','3517374844008722.coc.gml','3517378317183344.coc.gml','3517880442503213.coc.gml','3518018271478014.coc.gml','3518374388216553.coc.gml','3518554643491425.coc.gml','3518876082450868.coc.gml','3518889122334776.coc.gml','3518924249182559.coc.gml','3524708206766693.coc.gml','3553858343177336.coc.gml','3555764297312504.coc.gml','3559702166286240.coc.gml','3560740088421769.coc.gml','3560817721137817.coc.gml','3581880941207958.coc.gml','3584031612169073.coc.gml','3586578800272420.coc.gml','3343770315269085.coc.gml','3343828640519374.coc.gml','3344200536278976.coc.gml','3344204856189380.coc.gml','3344251610958634.coc.gml','3344947345156943.coc.gml','3345285264425627.coc.gml','3345943583930879.coc.gml','3346337229522031.coc.gml','3346646386115884.coc.gml','3348968449285798.coc.gml','3365285795421520.coc.gml','3367612479773055.coc.gml','3367696964256940.coc.gml','3369068477131339.coc.gml','3371247520439214.coc.gml','3371270547035409.coc.gml','3376346833399388.coc.gml','3383683287711280.coc.gml','3430490826135856.coc.gml','3431235789559517.coc.gml','3435254721283066.coc.gml','3440833729050905.coc.gml','3466248535071107.coc.gml','3481617253114874.coc.gml','3487580337482630.coc.gml','3488816159775764.coc.gml','3488842327677557.coc.gml','3488968551195859.coc.gml','3489462720299193.coc.gml','3497517021192038.coc.gml','3497540102476487.coc.gml','3504590328512715.coc.gml','3506452277059335.coc.gml','3506978007041225.coc.gml','3507953124535865.coc.gml','3508256699661280.coc.gml','3509097477346600.coc.gml','3509751591473526.coc.gml','3510725307943432.coc.gml','3511585550779014.coc.gml','3511661669904764.coc.gml','3512241854636226.coc.gml','3512265636249516.coc.gml','3512407622499003.coc.gml','3512467252803282.coc.gml','3512633108174568.coc.gml','3512642997758015.coc.gml','3512675969861915.coc.gml','3512846728070462.coc.gml','3513330587596297.coc.gml','3513645327614227.coc.gml','3516311047353690.coc.gml','3524530280557536.coc.gml','3524644067598719.coc.gml','3527628663048707.coc.gml','3528870550951587.coc.gml','3553179003983639.coc.gml','3553712926158978.coc.gml','3557998624957120.coc.gml','3558051988500239.coc.gml','3580448250376461.coc.gml','3581083603782299.coc.gml','3581833155041015.coc.gml','3582675862318360.coc.gml','3590965983991304.coc.gml','3385409596124201.coc.gml','3488298427647276.coc.gml','3503979856274851.coc.gml','3506843546638885.coc.gml','3512409568346880.coc.gml','3518734310070023.coc.gml','3530377388318988.coc.gml','3550786464348915.coc.gml','3552170865080083.coc.gml','3573993774557103.coc.gml','3581619170077512.coc.gml','3591379278050636.coc.gml','3571815701857951.coc.gml']]
    for filep in os.listdir(r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\IN'):
        filepath = 'G:\\HFS\\WeiboData\\HFSWeiboStatNet\\Svm\\IN'+'\\'+filep
        if os.path.isfile(filepath):
            print filep
            featurep+='_'+filep
#             for it in gt.kmeans3group:
            featlist = gt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\Net\.indegreePowerLawFitalpha')
            featlist = delcoc(featlist,delstr='.coc',colindex=0)
            vd = gt.normlistlist(a.selectPercent(featlist,perc,1),metacolcount=2,sumormax='max')
#                 print len(featlist)
#                 itresult = []
#                 k+=1
#                 for iti in it:
#                     itresult.append([iti.replace('.coc.gml','')])
# #                 print 'vd=====',vd,'\nitr===',itresult
#                 vd = gt.connectlist(vd,itresult,0,0)
#                 print len(itresult),len(vd)   


            svmfeatures(vd=vd,filename=str(k)+'_'+str(perc)+'_'+filep,rangemin=rangemin,rangemax=rangemax)
            feature = gt.normlistlist(a.selectPercent(gt.csv2list_new(filepath),perc,1),metacolcount=2,sumormax='max')#gt.csv2list_new(filepath)#
            vd = gt.connectlist(vd, feature, 0, 0, passcol=2)
            print 'finished one:',filep
#                 featurep+='_'+filep
 
#                 svmfeatures(vd,filename='all'+str(k)+'_'+featurep+'_'+str(perc),rangemin=rangemin,rangemax=rangemax)#featurep)
                
    generateSH(foldername=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm',percent=str(perc))
    
    print gt.getfilelistin_folder(r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm')



# def allmain22(percent):
#     perc = percent
#     rangemin=0
#     rangemax=100000
#     
#     # vd = gt.connectlist(vd, aspllist, 0, 0, passcol=2)
#     featurep = ''
#     k = 0
#     for it in gt.kmeans3group:
#         vd = gt.normlistlist(a.selectPercent(gt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\Net\.vcount'),perc,1),metacolcount=2,sumormax='max')
#         itresult = []
#         k+=1
#         for iti in it:
#             itresult.append([iti.replace('.coc.gml','')])
#         print 'vd=====',vd,'\nitr===',itresult
#         vd = gt.connectlist(vd,itresult,0,0)
#         print len(itresult)    
# #         filepath = 'G:\\HFS\\WeiboData\\HFSWeiboStatNet\\Svm\\IN'+'\\'+filep
# #         if os.path.isfile(filepath):
# #             feature = gt.normlistlist(a.selectPercent(gt.csv2list_new(filepath),perc,1),metacolcount=2,sumormax='max')#gt.csv2list_new(filepath)#
# #             svmfeatures(vd=feature,filename=str(perc)+'_'+filep,rangemin=rangemin,rangemax=rangemax)
# #             vd = gt.connectlist(vd, feature, 0, 0, passcol=2)
# #             print 'finished one:',filep
# #             featurep+='_'+filep
#      
#         svmfeatures(vd,filename='all'+str(k)+'_'+str(perc),rangemin=rangemin,rangemax=rangemax)#featurep)
#     generateSH(foldername=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm',percent=str(perc))
#     
#     print gt.getfilelistin_folder(r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm')



        
def drawsf(vdsf):
    import matplotlib.pyplot as plt

    vdsfdep = gt.departlist(vdsf,'1','-1',1,1)
    for it in vdsfdep[0]:
        y1 = it
        plt.plot(range(len(y1)-2),y1[2:],color='r',linestyle='',marker='o')
    for it in vdsfdep[1]:
        y2 = it
        plt.plot(range(len(y2)-2),y2[2:],color='b',linestyle='',marker='o')
     
    x = gt.averageLongtitude(vdsfdep[0])
    y = gt.averageLongtitude(vdsfdep[1])
    plt.plot(range(len(x[0])-2),x[0][2:],color='r')
    plt.plot(range(len(y[0])-2),y[0][2:],color='b')
    plt.show()          

# inpath=r'G:\\HFS\\WeiboData\\Statistics\\20130622new\\stat_net_combine.txt'
# outpath='G:\\HFS\\WeiboData\\Statistics\\20130622new\\svm\\stat_net_combine.svm.netall'
# # colselect=['vcount','ecount','density','lenclustersmodeisweak','assortativityindegree-directedistrue']
# # colselect=['r_gr_recnt', 'r_gr_repostcnt', 'r_gr_comcnt', 'r_gr_attitudecnt', 'r_gr_fanscnt', 'r_gr_befanscnt', 'r_gr_weibocnt', 'r_ue_distusercnt', 'r_ue_allusercnt', 'r_ue_echouseratio', 'r_uv_distVusercnt', 'r_uv_Vusercnt', 'r_uv_echoVuseratio', 'c_gr_recnt', 'c_gr_fanscnt', 'c_gr_befanscnt', 'c_gr_weibocnt', 'c_ue_distusercnt', 'c_ue_allusercnt', 'c_ue_echouseratio', 'c_uv_distVusercnt', 'c_uv_Vusercnt', 'c_uv_echoVuseratio', 'coc_s_allsender', 'coc_s_distsender', 'coc_s_senderecho', 'coc_r_allreceiver', 'coc_r_distreceiver', 'coc_r_receiverecho', 'coc_rt1_allreceiver1', 'coc_rt1_distreceiver1', 'coc_rt1_receiverecho1', 'coc_rt1_allreceiver2', 'coc_rt1_distreceiver2', 'coc_rt1_receiverecho2', 'coc_rt2_allreceiver0', 'coc_rt2_distreceiver0', 'coc_rt2_receiverecho0', 'coc_rt2_allreceiver1', 'coc_rt2_distreceiver1', 'coc_rt2_receiverecho1']
# colselect=['vcount', 'ecount', 'density', 'lenclustersmodeisweak', 'lenclustersmodeisstrong', 'clusVertexClusteringiantclustersmodeisweakvcount', 'clusVertexClusteringiantclustersmodeisweakecount', 'ecount2vcount', 'transitivity_undirectedmodeis0', 'average_path_length', 'diameter', 'assortativitydegree-directedistrue', 'assortativityindegree-directedistrue', 'assortativityoutdegree-directedistrue', 'assortativitydegree-directedistrueF', 'assortativityindegree-directedistrueF', 'assortativityoutdegree-directedistrueF', 'degreePowerLawFitalpha', 'degreePowerLawFitxmin', 'degreePowerLawFitp', 'degreePowerLawFitL', 'degreePowerLawFitD', 'indegreePowerLawFitalpha', 'indegreePowerLawFitxmin', 'indegreePowerLawFitp', 'indegreePowerLawFitL', 'indegreePowerLawFitD', 'outdegreePowerLawFitalpha', 'outdegreePowerLawFitxmin', 'outdegreePowerLawFitp', 'outdegreePowerLawFitL', 'outdegreePowerLawFitD']

def txt2svm_tr_pred(tr=False,predict=False):
    "IN:tr- weather svm train or not; if yes,predict or not"
    "OUT:svm format of txt; predict results"
    
    # a.txt2libsvmformat(inpath,outpath,colselect,'successed')
    perclist=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]#0.2,0.4,0.6,0.8,0.9,
    for it in perclist:
        allmain2(it)
    
    import time
    predictfilename = 'predict_'+str(time.time()) 
    # generateSH(foldername=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm')
    batfile = generateBAT(foldername=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm', percent='all',predictfilename=predictfilename)
    if tr:
        os.system(batfile)
    if tr and predict:
        predictR = deal_file_libsvmpredict(r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\results\\'+predictfilename)
        predictRZ = zip(*(predictR))
        print numpy.average(predictRZ[1])

txt2svm_tr_pred(tr=False,predict=False)