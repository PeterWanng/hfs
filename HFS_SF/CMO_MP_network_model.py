#!/usr/bin/python
#coding=utf-8
# -*- coding=UTF-8 -*-

import networkx as nx
import random
import matplotlib.pyplot as plt
import codecs
import sys
import igraph as ig
from tools import commontools as wtt

wt = wtt()


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
	for node in node_list:

		IG.add_vertices(1)

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
	
	
	ig.write(graph_list[0],r'G:\HFS\WeiboData\HFSWeiboGML\test\simulation_'+str(model_type)+'.gml')
	return graph_list

def real_data(file_path):
	name = os.path.split(os.path.basename(file_path))[0]
	IG=ig.Graph.Read_GML(file_path)
	IG = IG.subgraph_edges(IG.es.select(plzftype_in=['2','8']))
	ig.write(IG,r'G:\HFS\WeiboData\HFSWeiboGML\test\real_'+name+'.gml')
	return IG


def do_experiment(node_file_path, gml_file_path, runtime):
	name = os.path.split(os.path.basename(gml_file_path))[0]
	
	nodes = read_list_nodes_from_file(node_file_path, 2)
	BA_list=multi_run(nodes, 30, 5 ,name)
	FAN_list=multi_run(nodes, 30, 4 ,name)
	RG=real_data(gml_file_path)
	
	ig.write(BA_list[0],r'G:\HFS\WeiboData\HFSWeiboGML\test\simulation_BA_'+name+'.gml')
	ig.write(FAN_list[0],r'G:\HFS\WeiboData\HFSWeiboGML\test\simulation_Fans_'+name+'.gml')
	ig.write(RG,r'G:\HFS\WeiboData\HFSWeiboGML\test\real_'+name+'.gml')


def getdegreelist(nodes,modelindex=4):
	graph_list_BA = multi_run(nodes, 10, modelindex)
	
	degree_list_BA = []
	for igra in graph_list_BA:
		degree= igra.degree()
# 		print igra.vs.attributes()
# 		sourceNode = igra.vs.find(distance=0)
# 		degree= igra.shortest_paths(source=None,target=sourceNode)
# 		print degree
# 		degree= igra.diameter()
		degree_list_BA.extend(degree)
	
	all_degree=[]
	for degree in degree_list_BA:
		all_degree.extend([degree])
	degree_sequence=sorted(all_degree, reverse=True)
	return degree_sequence


import os
nodeAttrFolder = r'G:\HFS\WeiboData\HFSWeiboNodeData'
gmlFolder = r'G:\HFS\WeiboData\HFSWeiboGML'
for file in os.listdir(gmlFolder):
#  	try:
		filename = os.path.splitext(file)[0]
		print filename
		nodes = read_list_nodes_from_file(nodeAttrFolder+'\\'+filename+'.repost', start_line=2)
		
		degree_sequence_ba = getdegreelist(nodes,5)
		degree_sequence_me = getdegreelist(nodes,4)	
	# 	print degree_sequence
		# wt.listDistribution(degree_sequence)
		
		
		real_degree=real_data(gmlFolder+'\\'+filename+'.gml')
		real_degree = real_degree.degree()
		 
		wt.list_2_Distribution([real_degree, degree_sequence_me])#degree_sequence_ba=
#  	except Exception, e:
#  		print filename,e		
		
	# 	plt.semilogy(degree_sequence, 'b-', marker='o')
	# 	plt.savefig("degree_histogram.png")
	# 	plt.show()
		
		
		# read file in to list
		
		# model, run once, get the output, degree, distribution and some statistics
		
		# run muliple times





