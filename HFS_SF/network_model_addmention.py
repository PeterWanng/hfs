#!/usr/bin/python
#coding=utf-8
# -*- coding=UTF-8 -*-

import networkx as nx
import random
import matplotlib.pyplot as plt
import codecs
import sys
import igraph as ig
from igraph import clustering as clus
from igraph import statistics as stcs
from tools import commontools as wtf
wt = wtf()


def random_pick(some_list, weights):
	weight_sum=sum(weights)
	x=random.uniform(0,1) * weight_sum
	cumulative_weight=0.0
	for item, item_weight in zip(some_list, weights):
		cumulative_weight += item_weight
		if x < cumulative_weight:
			break
	return [item]


def random_pick_index(weights):
	weight_sum=sum(weights)
	x=random.uniform(0,1) * weight_sum
	cumulative_weight=0.0
	i=0
	for item_weight in weights:
		cumulative_weight += item_weight
		if x < cumulative_weight:
			break
		i += 1

	return i

def random_test(y):
	x=random.uniform(0,1)
	#print x, y
	if x <= y:
		return True
	else:
		return False


class list_node:
	screen_name=''
	uid=''
	fan_number=0
	follow_number=0
	def __init__(self,name,uid,fan,follow):
		self.screen_name=name
		self.uid=uid
		self.fan_number=fan
		self.follow_number=follow

def read_list_nodes_from_file(input_file_path, start_line=1):

	input=open(input_file_path)
	#input=codecs.open(input_file_path, "r", "utf-8")
	i=1
	
	while i < start_line:
		input.readline()
		i+=1

	result=[]
	for line in input.readlines():
		line=line.strip()
		tokens=line.split(",")
		uid=tokens[0]
		screen_name=tokens[1]
		fans_number=int(tokens[2])
		follow_number=int(tokens[3])
		n=list_node(screen_name, uid, fans_number, follow_number)
		result.append(n)
	
	input.close()

	return result

def find_mention_list_from_file(mentionlist, findstr):
	result = []
	for item in mentionlist:
		if str(item[0].replace('.repost',''))==str(findstr):
			result = item[1:]
			break
	return result	

def _random_subset(seq,m):
	targets=set()
	while len(targets)<m:
		x=random.choice(seq)
		targets.add(x)

	return targets
    

# linear preferential attachment (fans_number) networkx
def model_1():
	G=nx.DiGraph()
	nodes = read_list_nodes_from_file("3519173332815242.repost", 2)
	
	source=1
	repeated_nodes=[]
	for node in nodes:
		print node.screen_name.decode('utf8').encode('gbk')

		G.add_node(source)

		if source == 1:
			repeated_nodes.extend([source]*node.fan_number)

		else:
			target = _random_subset(repeated_nodes,1)

			G.add_edges_from(zip([source],target))
		
			repeated_nodes.extend([source]*node.fan_number)

		source +=1

	nx.draw(G)
	plt.savefig("test.png")
	plt.show()


# linear preferential attachment (in_degree+1) networkx
def model_2():
	G=nx.DiGraph()
	nodes = read_list_nodes_from_file("3519173332815242.repost", 2)
	
	source=1
	repeated_nodes=[]
	for node in nodes:
		print node.screen_name.decode('utf8').encode('gbk')

		G.add_node(source)

		if source == 1:
			repeated_nodes.extend([source])

		else:
			#target is a list
			target = _random_subset(repeated_nodes,1)

			G.add_edges_from(zip([source],target))

			repeated_nodes.extend(target)
		
			repeated_nodes.extend([source])

		source +=1

	nx.draw(G)
	plt.savefig("model2.png")
	plt.show()

def model_3_weight(fans_number, distance):
	return fans_number * (0.1**distance)

# linear preferential attachment (fans_number and shortest path to original node) networkx
def model_3():
	G=nx.DiGraph()
	nodes = read_list_nodes_from_file("3519173332815242.repost", 2)
	
	source=1
	#repeated_nodes=[]
	weight_list=[]
	for node in nodes:
		#print node.screen_name.decode('utf8').encode('gbk')

		G.add_node(source)

		if source == 1:
			G.node[source]['distance']=0

			weight_list.append(model_3_weight(node.fan_number, 0))
	

		else:
			#target is a list
			target_list = random_pick(G.node, weight_list)

			G.add_edges_from(zip([source],target_list))

			G.node[source]['distance']= nx.shortest_path_length(G, source=source, target=1)

			print source

			print G.node[source]['distance']

			weight_list.append(model_3_weight(node.fan_number, G.node[source]['distance']))


		source +=1

	nx.draw(G)
	plt.savefig("model3.png")
	plt.show()


# linear preferential attachment (fans_number and shortest path to original node) igraph
def model_4(node_list):
	IG=ig.Graph(directed=True)

	source=0
	weight_list=[]
	i = 0
	for node in node_list:
		i+=1
		IG.add_vertices(1)
		IG.vs[source]['label']=i
		if source == 0:
			IG.vs[source]['distance']=0

			weight_list.append(model_3_weight(node.fan_number, 0))
	

		else:
			
			target = random_pick_index(weight_list)

			IG.add_edges(zip([source],[target]))

			IG.vs[source]['distance']= IG.shortest_paths_dijkstra([source], [0])[0][0]

			#print source

			#print IG.vs[source]['distance']

			weight_list.append(model_3_weight(node.fan_number, IG.vs[source]['distance']))


		source +=1
	return IG

	
# linear preferential attachment (in_degree+1) igraph
def model_5(node_list):
	IG=ig.Graph(directed=True)

	source=0
	weight_list=[]
	for node in node_list:

		IG.add_vertices(1)

		if source == 0:
			IG.vs[source]['distance']=0

		else:
			#target is a list
			target = random_pick_index(weight_list)

			IG.add_edges(zip([source],[target]))

			#print zip([source],[target])

			weight_list[target] += 1
		
		weight_list.append(1)
		source +=1

	return IG

def mention_model_1(IG, mention_list, total_number):
	source=0
	weight_list = [1] * len(IG.vs)
	add_number=0
	target_set = set()

	current_number=len(IG.vs)
	for mention_number in mention_list:
		new_node_number = 0
		exist_node_number = 0
		for i in range(1, int(mention_number)+1):
			# connect new node
			#print total_number-current_number+1
			if not random_test(1.0/float(total_number-current_number+1)):
				new_node_number += 1
				current_number += 1

			# connect exist node
			else:
				exist_node_number += 1
		
		#print "mention_number", mention_number
		target_set.clear()
		while exist_node_number > len(target_set):
			target = random_pick_index(weight_list)
			target_set.add(target)

		for target in target_set:
			weight_list[target] += 1
			IG.add_edge(source,target, type="mention")
			add_number += 1


		while new_node_number > 0:
			target = len(IG.vs)
			IG.add_vertices(1)
			IG.add_edge(source,target, type="mention")
			add_number += 1
			weight_list.append(1)

			new_node_number -= 1

			
			

		source += 1
	print "add number ", add_number
	return IG

def model_6():
	input=open("test.txt")
	repeated_nodes=[]

	source=1
	line = input.readline()
	line=line.strip()
	tokens=line.split(",")
	uid=tokens[0]
	fans_number=int(tokens[1])
	follow_number=int(tokens[2])
	G.add_node(source)

	repeated_nodes.extend([source]*fans_number)

	source += 1

	for line in input.readlines():
		line=line.strip()
		tokens=line.split(",")
		uid=tokens[0]
		fans_number=int(tokens[1])
		follow_number=int(tokens[2])
		
		G.add_node(source)

		target = _random_subset(repeated_nodes,1)

		G.add_edges_from(zip([source],target))
		
		repeated_nodes.extend([source]*fans_number)

		source += 1


	nx.draw(G)
	plt.savefig("test.png")
	plt.show()

def multi_run(node_list, run_time, model_type):
	i=0
	graph_list=[]
	while i < run_time:
		if model_type==4:
			IG=model_4(nodes)
			
		elif model_type==5:
			IG=model_5(nodes)

		graph_list.append(IG)	
		i +=1
	return graph_list

def real_data(file_path):
	IG=ig.Graph.Read_GML(file_path)
	return IG

def real_data_repost(file_path):
	IG=ig.Graph.Read_GML(file_path)
	IG = IG.subgraph_edges(IG.es.select(retwitype_in=['0','8']))
	return IG

def do_experiment(node_file_path, gml_file_path, runtime):
	nodes = read_list_nodes_from_file(node_file_path, 2)
	BA_list=multi_run(nodes, 30, 5)
	FAN_list=multi_run(nodes, 30, 4)
	RG=real_data(gml_file_path)

def analysisNet(graph):    
    g=graph
    vcount=g.vcount()   
    ecount=g.ecount() 
#     degree=g.degree() 
#     indegree=g.indegree() 
#     outdegree=g.outdegree() 
#     degreePowerLawFit=stcs.power_law_fit(degree,method='auto',return_alpha_only=False) 
#     indegreePowerLawFit=stcs.power_law_fit(indegree, method='auto',return_alpha_only=False) 
#     outdegreePowerLawFit=stcs.power_law_fit(outdegree,method='auto',return_alpha_only=False)
#     assorDeg=g.assortativity(degree,directed= False) 
#     assorDegD=g.assortativity(degree,directed= True) 
#     assorInDeg=g.assortativity(indegree,directed= True)
#     assorOutDeg=g.assortativity(outdegree,directed= True)
    
#     assorDegF='1' if assorDeg>0 else '-1'  
#     assorInDegF='1' if assorInDeg>0 else '-1'   
#     assorOutDegF= '1' if assorOutDeg>0 else '-1'          
#     print g.average_path_length()
    return [\
# 		str(vcount),\
#     str(ecount),\
    str(g.density()),\
#     str(len(g.clusters(mode='weak'))),\
#     str(len(g.clusters(mode='strong'))),\
#     str(clus.VertexClustering.giant(g.clusters(mode='weak')).vcount()),\
#     str(clus.VertexClustering.giant(g.clusters(mode='weak')).ecount()),\
#     str((ecount*2.0)/(vcount*(vcount-1))),\
    str(g.transitivity_undirected(mode='0')) ,\
    str(g.average_path_length()),\
    str(g.diameter()),\
#     str(assorDeg),\
#     str(assorDegD),\
#     str(assorInDeg),\
#     str(assorOutDeg),\
#     str(assorDegF),\
#     str(assorInDegF),\
#     str(assorOutDegF),\
#     str(degreePowerLawFit.alpha),\
#     str(degreePowerLawFit.xmin),\
#     str(degreePowerLawFit.p),\
#     str(degreePowerLawFit.L),\
#     str(degreePowerLawFit.D),\
#     str(indegreePowerLawFit.alpha),\
#     str(indegreePowerLawFit.xmin),\
#     str(indegreePowerLawFit.p),\
#     str(indegreePowerLawFit.L),\
#     str(indegreePowerLawFit.D),\
#     str(outdegreePowerLawFit.alpha),\
#     str(outdegreePowerLawFit.xmin),\
#     str(outdegreePowerLawFit.p),\
#     str(outdegreePowerLawFit.L),\
#     str(outdegreePowerLawFit.D)
	]

import os
import csv
import time
def SR():
	nodeAttrFolder = r'G:\HFS\WeiboData\HFSWeiboNodeData'
	gmlFolder = r'G:\HFS\WeiboData\HFSWeiboGML'
	mentionlist = wt.csv2list_new(csvfilepath=r"G:\HFS\WeiboData\HFSWeiboStatNet\Stat\Mentioncntlist.txt")
	meta = wt.csv2list(r'G:\HFS\WeiboData\Statistics\meta_successed308.txt')
	metadic = wt.list2dict(meta)
	print metadic
	netAttr_R = []
	netAttr_S = []
	indegree_R = []
	indegree_S = []
	outdegree_R = []
	outdegree_S = []
	netAttr_R_writer = csv.writer(file(r'G:\HFS\WeiboData\HFSWeiboStatNet\RS\netAttr_R.csv','w'))
	netAttr_S_writer = csv.writer(file(r'G:\HFS\WeiboData\HFSWeiboStatNet\RS\netAttr_S.csv','w'))
	indegree_R_writer = csv.writer(file(r'G:\HFS\WeiboData\HFSWeiboStatNet\RS\indegree_R.csv','w'))
	indegree_S_writer = csv.writer(file(r'G:\HFS\WeiboData\HFSWeiboStatNet\RS\indegree_S.csv','w'))
	outdegree_R_writer = csv.writer(file(r'G:\HFS\WeiboData\HFSWeiboStatNet\RS\outdegree_R.csv','w'))
	outdegree_S_writer = csv.writer(file(r'G:\HFS\WeiboData\HFSWeiboStatNet\RS\outdegree_S.csv','w'))
	
	errorlist = []
	i = 0
	for filef in os.listdir(gmlFolder):
# 	  	try:
			i+=1
			
			if i<3000:
				filename = os.path.splitext(filef)[0]
				forwordcnt = 10000000
				try:
					forwordcnt = metadic.get(str(filename))[4]
				except:
					pass
				if float(forwordcnt)<2001:
	
					print i, filename,time.clock()
			
					IG_r=real_data(gmlFolder+'\\'+filename+'.gml')#real_data(r"G:\MyPapers\CMO\testData4Cui\3343740313561521.gml")

					"two steps nwtwork growth, one by one "
					nodes = read_list_nodes_from_file(nodeAttrFolder+'\\'+filename+'.repost', start_line=2)
					IG_s=model_4(nodes)		
					mention_list = find_mention_list_from_file(mentionlist, filename)
					IG_s=mention_model_1(IG_s, mention_list, len(IG_r.vs))
					
					gmlf_s = gmlFolder+'\\SimulationGml\\sim'+filename+'.gml'
					ig.write(IG_s,gmlf_s)
					
# 					IG_r=clus.VertexClustering.giant(IG_r.clusters(mode='weak'))
			
					ra = analysisNet(IG_r)#(ig.VertexClustering.giant(IG_r.clusters(mode='weak')))
					sa = analysisNet(IG_s)
					ra.insert(0,filename)
					sa.insert(0,filename)
					netAttr_R.append(ra)
					netAttr_S.append(sa)
					
					rIndegree = IG_r.indegree()
					sIndegree = IG_s.indegree()
					rOutdegree = IG_r.outdegree()
					sOutdegree = IG_s.outdegree()
		 			rIndegree.insert(0,filename)
		 			sIndegree.insert(0,filename)
		 			rOutdegree.insert(0,filename)
		 			sOutdegree.insert(0,filename)
					
					indegree_R.append(rIndegree)
					indegree_S.append(sIndegree)
					outdegree_R.append(rOutdegree)
					outdegree_S.append(sOutdegree)
					
					netAttr_R_writer.writerow(ra)
					netAttr_S_writer.writerow(sa)
					indegree_R_writer.writerow(rIndegree)
					indegree_S_writer.writerow(sIndegree)
					outdegree_R_writer.writerow(rOutdegree)
					outdegree_S_writer.writerow(sOutdegree)
				else:
					pass
# 		except:
# 			errorlist.append(filef)
# 			print 'error==========================:',filef
	
	print 'all the errors==========================:',len(errorlist),errorlist
	##	all the errors==========================: ['3346646386115884.gml', '3488818235948066.gml', '3488825538082032.gml', '3488842327677557.gml', '3488968551195859.gml', '3489084565802342.gml', '3489089556938378.gml', '3489092799395971.gml', '3489137279411789.gml', '3489148290157520.gml', '3489462720299193.gml', '3489558450803933.gml', '3489586389438248.gml', '3489743314991378.gml', '3489804664943100.gml', '3491356209051281.gml', '3491608849042650.gml', '3492678328811636.gml', '3492682624290826.gml', '3492684352482154.gml', '3492805542177005.gml', '3493173752439900.gml', '3494152501426914.gml', '3494236496496836.gml', '3494489962794555.gml', '3495157817228723.gml', '3495425338709660.gml', '3495557874364772.gml', '3495908950319195.gml', '3497476986739290.gml', '3497517021192038.gml', '3497540102476487.gml', '3498227716765728.gml', '3498422722386146.gml', '3501198583561829.gml', '3502012958680889.gml', '3503979856274851.gml', '3504252389771186.gml', '3504355070314763.gml', '3504590328512715.gml', '3505165031934094.gml', '3505779032316582.gml', '3506067055046956.gml', '3506429741008735.gml', '3506452277059335.gml', '3506843546638885.gml', '3506858382217257.gml', '3506978007041225.gml', '3507543877973721.gml', '3507607539020444.gml', '3507662178094930.gml', '3507671015760502.gml', '3507953124535865.gml', '3508035156247306.gml', '3508256699661280.gml', '3508278808120380.gml', '3509097477346600.gml', '3509347126682264.gml', '3509438231211989.gml', '3509751591473526.gml', '3509885691781722.gml', '3510108007340190.gml', '3510150776647546.gml', '3510725307943432.gml', '3510947052234805.gml', '3511312581651670.gml', '3511566865123223.gml', '3511585550779014.gml', '3511661669904764.gml', '3511850572507320.gml', '3511918478199744.gml', '3511950958712857.gml', '3511953756712121.gml', '3511983850692431.gml', '3512027492461885.gml', '3512192651209611.gml', '3512207839497881.gml', '3512220862117440.gml', '3512224943485819.gml', '3512225370844164.gml', '3512228487367164.gml', '3512228747718413.gml', '3512241854636226.gml', '3512260125547589.gml', '3512261920248384.gml', '3512265636249516.gml', '3512288331952105.gml', '3512320883828568.gml', '3512343789230980.gml', '3512346909867070.gml', '3512362638453577.gml', '3512365087957339.gml', '3512367365458914.gml', '3512371488398141.gml', '3512387527089684.gml', '3512391692548104.gml', '3512407622499003.gml', '3512409568346880.gml', '3512467252803282.gml', '3512557425820703.gml', '3512564992138128.gml', '3512568150742963.gml', '3512570026130918.gml', '3512586882317341.gml', '3512593978848674.gml', '3512597141818367.gml', '3512598488183675.gml', '3512631170591260.gml', '3512633108174568.gml', '3512638635619787.gml', '3512642997758015.gml', '3512649558390590.gml', '3512651290338120.gml', '3512661431797880.gml', '3512665513423714.gml', '3512673221800564.gml', '3512675969861915.gml', '3512681036745319.gml', '3512681942436390.gml', '3512693292397369.gml', '3512703459106709.gml', '3512704620407286.gml', '3512722819965166.gml', '3512723826282658.gml', '3512725802294904.gml', '3512731699677616.gml', '3512751521888667.gml', '3512753392115009.gml', '3512755191392034.gml', '3512764419413627.gml', '3512767539668338.gml', '3512846728070462.gml', '3512944329886751.gml', '3512956526933221.gml', '3512965133226559.gml', '3513008209201946.gml', '3513009873831054.gml', '3513027452433705.gml', '3513054618763335.gml', '3513170419502667.gml', '3513262479906932.gml', '3513299721572710.gml', '3513330587596297.gml', '3513353957580369.gml', '3513361775532449.gml', '3513435633981893.gml', '3513457897170153.gml', '3513472585606907.gml', '3513477123020136.gml', '3513485109002235.gml', '3513524855215093.gml', '3513645327614227.gml', '3513651849587189.gml', '3513665519425522.gml', '3513671424206849.gml', '3513684632475119.gml', '3513732346670068.gml', '3513732681977292.gml', '3513733382475662.gml', '3513737430369962.gml', '3513738009766461.gml', '3513739158587782.gml', '3513747752676493.gml', '3513761854134169.gml', '3513762864278058.gml', '3513784817583701.gml', '3513786944578870.gml', '3513795827993634.gml', '3513797614859184.gml', '3513821030864621.gml', '3513822322955028.gml', '3513871362653946.gml', '3514033757910710.gml', '3514047335033747.gml', '3514054737834781.gml', '3514058537701986.gml', '3514061079292261.gml', '3514069312670878.gml', '3514074715123866.gml', '3514083762166772.gml', '3514112790871598.gml', '3514202721379789.gml', '3514207033581502.gml', '3514216143529145.gml', '3514229367981237.gml', '3514287944897392.gml', '3514409529974363.gml', '3514415834044554.gml', '3514416295764354.gml', '3514448201592653.gml', '3514484653860068.gml', '3514517281419758.gml', '3514574454119360.gml', '3514712136139677.gml', '3514721241880789.gml', '3514725432613389.gml', '3514793086497284.gml', '3516311047353690.gml', '3516368059904172.gml', '3516665499181717.gml', '3516669001647040.gml', '3516712157036112.gml', '3516941173798600.gml', '3517012276302505.gml', '3517122988859030.gml', '3517263451924815.gml', '3517300592201464.gml', '3517351528480026.gml', '3517374844008722.gml', '3517378317183344.gml', '3517807143042530.gml', '3517880442503213.gml', '3518018271478014.gml', '3518037380208073.gml', '3518192234385654.gml', '3518374388216553.gml', '3518554643491425.gml', '3518734310070023.gml', '3518797568597345.gml', '3518864421482109.gml', '3518876082450868.gml', '3518889122334776.gml', '3518889973774430.gml', '3518896877283474.gml', '3518897993492899.gml', '3518902011319515.gml', '3518909502486323.gml', '3518924249182559.gml', '3519083113115752.gml', '3519104033490770.gml', '3519173332815242.gml', '3519233508278922.gml', '3521836014420909.gml', '3522462203254847.gml', '3522707796773995.gml', '3523242033543909.gml', '3524530280557536.gml', '3524644067598719.gml', '3524708206766693.gml', '3526708160065364.gml', '3527143084709163.gml', '3527449691832220.gml', '3527628663048707.gml', '3528870550951587.gml', '3530377388318988.gml', '3550786464348915.gml', '3552170865080083.gml', '3553179003983639.gml', '3553712926158978.gml', '3553858343177336.gml', '3554653150827755.gml', '3555764297312504.gml', '3557998624957120.gml', '3558051988500239.gml', '3558226899367894.gml', '3558246365665871.gml', '3559702166286240.gml', '3560576217397500.gml', '3560740088421769.gml', '3560817721137817.gml', '3571815701857951.gml', '3573993774557103.gml', '3580448250376461.gml', '3581029350321431.gml', '3581083603782299.gml', '3581619170077512.gml', '3581830525479047.gml', '3581833155041015.gml', '3581866814344587.gml', '3581874289297524.gml', '3581880941207958.gml', '3582141898089800.gml', '3582182788767024.gml', '3582187498347368.gml', '3582675862318360.gml', '3584031612169073.gml', '3586578800272420.gml', '3590965983991304.gml', '3591379278050636.gml', '3591398001249115.gml', 'SimulationGml', 'test']
		
	# 			print IG_r.vs.attributes()
	# 			ss = IG_s.vs.attributes()
	# 			print ss
	# print netAttr_R,'\n',netAttr_S,'\n',indegree_R,'\n',indegree_S,'\n',outdegree_R,'\n',outdegree_S
# SR()
netAttr_R = zip(*wt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\RS\netAttr_R.csv'))			
netAttr_S = zip(*wt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\RS\netAttr_S.csv'))	
i=0	
j=0
xlabel = ['Clustering Coefficient','Average Path Length','Diameter']	
ylabel = ['Frequency','','']
plt.legend(('Real','Simulated'),loc='upper right')
for itemr,items in zip(*(netAttr_R,netAttr_S))[1:]:
	i+=1
	x=range(1,len(itemr)+1)
	y1=map(float,itemr)
	y2=map(float,items)
	
	binseqdiv=0
	if i == 1:
		binseqdiv=10
	
	a = wt.list_2_Distribution([y1,y2],showfig=False,binseqdiv=binseqdiv)

	markerlist = ['o','x','o','x','o','x',]
	linestyle = ['solid','dashed','solid','dashed','solid','dashed',]
 	poslist = [1,2,3]
	plt.legend(('Real','Simulated','Real','Simulated'),loc='upper right')
	for fig in zip(*a):
		if i!=2:
			plt.subplot(1,3,poslist[j/2])
			plt.xlabel(xlabel[j/2])
			plt.ylabel(ylabel[j/2])
			x = fig[0]
			y = fig[1]
# 			if j==2:
# 				for it in x:
# 					xnew = []
# 					it=it/10.0
# 					xnew.append(it)
# 				x=xnew
			plt.plot(x,y,marker=list(markerlist)[j],linestyle=linestyle[j])#semilogy#[‘solid’ | ‘dashed’ | ‘dashdot’ | ‘dotted’]
			j+=1
	plt.legend(('Real','Simulated','Real','Simulated'),loc='upper right')
# 	y1.sort(reverse=True)
# 	y2.sort(reverse=True)
# #  	print y1,'\n',y2
#  	plt.plot(x,y1)
#  	plt.plot(x,y2)
#  	plt.show()
	
# 	plt.subplot(2,2,i)
# 	xlabel=i
# 	ylabel='Frenquency'
# 	plt.xlabel(xlabel)
# 	plt.ylabel(ylabel)
# 	fig1 = plt.semilogy(x,y1,marker='o',linestyle='')
# 	fig2 = plt.semilogy(x,y2,marker='o',linestyle='')
plt.show()

"just deal giant"
# gmlFolder = r'G:\HFS\WeiboData\HFSWeiboGML'
# i=0
# for file in os.listdir(gmlFolder):
# 	i+=1
# 	filename = os.path.splitext(file)[0]
# 	if i <3:
# 		IG_r=real_data(gmlFolder+'\\'+filename+'.gml')
# 		IG_r=clus.VertexClustering.giant(IG_r.clusters(mode='weak'))
# 		ra = analysisNet(IG_r)
# 		print ra

# degSin = []
# degRin = []
# degSout = []
# deginRt = wt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\RS\indegree_R.csv')
# deginSt = wt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\RS\indegree_S.csv')
# for it in deginRt:
# 	degRin.extend(it[1:])
# for it in deginSt:
# 	degSin.extend(it[1:]) 	
# degin = [degRin,degSin]	
# wt.list_2_Distribution(degin)
	
	# for v in IG_s.vs:
# 	print v['label']

#wt.list_2_Distribution(IG_s.indegree(), IG_r.indegree())



#degree_list = multi_run(nodes, 1, 5)

#all_degree=[]
#for degree in degree_list:
#	all_degree.extend(degree)
	
#degree_sequence=sorted(all_degree, reverse=True)
#print degree_sequence
#wt.listDistribution(degree_sequence)


#real_degree=real_data("3519173332815242_1.gml")

#wt.list_2_Distribution(degree_sequence, real_degree)


#plt.loglog(degree_sequence, 'b-', marker='o')
#plt.savefig("degree_histogram.png")
#plt.show()


# read file in to list

# model, run once, get the output, degree, distribution and some statistics

# run muliple times

