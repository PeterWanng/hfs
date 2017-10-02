#!/usr/bin/python
#coding=utf-8
# -*- coding=UTF-8 -*-

import networkx as nx
import random
import matplotlib.pyplot as plt
import codecs
import sys
import igraph as ig
import tools as wt


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

def read_mention_list_from_file(input_file_path, start_line=3):
	input=open(input_file_path)
	i = 1

	while i < start_line:
		input.readline()
		i+=1
	w_str = input.readline()
	mention_strs = w_str.split(",")
	mention_list=[]
	for str in mention_strs[1:]:
		mention_list.append(int(str))
	
	return mention_list

	

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
		for i in range(1, mention_number+1):
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


IG_r=real_data(r"G:\MyPapers\CMO\testData4Cui\3343740313561521.gml")#real_data(r"G:\MyPapers\CMO\testData4Cui\3343740313561521.gml")
print "gml file================"
print len(IG_r.vs)
print len (IG_r.es)
# print "8 edge number: ", len(IG_r.es.select(plzftype_in=['8']))
# print "2 edge number: ", len(IG_r.es.select(plzftype_in=['2']))

mention_list = read_mention_list_from_file(r"G:\HFS\WeiboData\HFSWeiboStatNet\Stat\test\Mentioncntlist.txt", 3)
print "mention list================"
print len(mention_list)
print sum(mention_list)
nodes = read_list_nodes_from_file(r"G:\HFS\WeiboData\HFSWeiboNodeData\3343740313561521.repost", 2)
IG_s=model_4(nodes)
print "withou mention ================"
print len(IG_s.vs)
print len(IG_s.es)
IG_s=mention_model_1(IG_s, mention_list, len(IG_r.vs))
print "with mention ================"
print len(IG_s.vs)
print len(IG_s.es)

gmlf_s = 'temp.gml'
ig.write(IG_s,gmlf_s)

print IG_r.vs.attributes()
ss = IG_s.vs.attributes()
print ss
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

