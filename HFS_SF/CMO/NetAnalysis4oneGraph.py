#encoding=utf8
import sys
sys.path.append("..\..")
from tools import commontools
from matplotlib import pyplot as plt
import numpy as np

               
import igraph as ig
from igraph import clustering as clus


import re
import time
import os
from tools import commontools as gtf
import csv
import scipy as sp 
from scipy import stats
gt=gtf()

    
class analyzeNet_AllNodes():
    '''example usage:
    gmlf = file(gml)
    g = ig.Graph.Read_GML(gmlf);print 'ig.Graph.Read_GML(gmlf) done'
    gmlf.close()
    netan = analyzeNet_AllNodes()
    netan.analyzeNetNodes(g,savefolder = workfolder+"giantatt\\")'''
    def drawandsave(self,lista,filep,xlabel):
        try:
            gt.saveList([lista],filep,writype='a+')
            #gt.listDistribution(lista,disfigdatafilepath=None,xlabel=xlabel,ylabel='Frequency',showfig=True,binsdivide=1)
        except Exception,e:
            print 'error:',filep,e
            
    def analyzeNetNodes(self,graphme,savefolder):
            '''IN: graph instance and the attribute save folder
            OUT:20 attribute files of the graph nodes
            Degree,Indegree,Outdegree,Betweeness,Shell,Coreness_ALL,Coreness_IN,Coreness_OUT,Clossess_ALL,Clossess_IN,Clossess_OUT,Eccentricity_ALL,Eccentricity_IN,Eccentricity_OUT,Pagerank
            '''
            result = []
#         try:
#             g=graphme
#             gg=clus.VertexClustering.giant(g.clusters(mode='weak'))
            gg=graphme
            gg.as_directed()
            degree=gg.degree()
            indegree=gg.indegree() 
            outdegree=gg.outdegree()
            self.drawandsave(degree,savefolder+'Degree.txt','Degree') 
            self.drawandsave(indegree,savefolder+'Indegree.txt','Indegree') 
            self.drawandsave(outdegree,savefolder+'Outdegree.txt','Outdegree')
            result.append(degree)
            result.append(indegree)
            result.append(outdegree)
            
            bet = gg.betweenness(vertices=None, directed=True, cutoff =None, weights=None,nobigint=True)
            self.drawandsave(bet,savefolder+'Betweeness.txt','Betweeness')
            result.append(bet)
            # bet = gg.betweenness(vertices=None, directed=True, cutoff =None, weights='w',nobigint=True)
            # self.drawandsave(bet,savefolder+'Betweeness_Weighted.txt','Betweeness_Weighted')
            #            kcore = gg.kcore()
            #            print assorDegD
            shell = gg.shell_index(mode='ALL')
            self.drawandsave(shell,savefolder+'Shell.txt','Shell')
            result.append(shell)
            coreness = gg.coreness(mode='ALL')#Reference: Vladimir Batagelj, Matjaz Zaversnik: An O(m) Algorithm for Core Decomposition of Networks.
            self.drawandsave(coreness,savefolder+'Coreness_ALL.txt','Coreness_ALL') 
            result.append(coreness)
            coreness = gg.coreness(mode='IN')#Reference: Vladimir Batagelj, Matjaz Zaversnik: An O(m) Algorithm for Core Decomposition of Networks.
            self.drawandsave(coreness,savefolder+'Coreness_IN.txt','Coreness_IN') 
            result.append(coreness)
            coreness = gg.coreness(mode='OUT')#Reference: Vladimir Batagelj, Matjaz Zaversnik: An O(m) Algorithm for Core Decomposition of Networks.
            self.drawandsave(coreness,savefolder+'Coreness_OUT.txt','Coreness_OUT') 
            result.append(coreness)

            clossess = gg.closeness(vertices=None, mode='ALL', cutoff =None, weights=None)#cutoff
            self.drawandsave(clossess,savefolder+'Clossess_ALL.txt','Clossess_ALL') 
            result.append(clossess)
            clossess = gg.closeness(vertices=None, mode='IN', cutoff =None, weights=None)#cutoff
            self.drawandsave(clossess,savefolder+'Clossess_IN.txt','Clossess_IN') 
            result.append(clossess)
            clossess = gg.closeness(vertices=None, mode='OUT', cutoff =None, weights=None)#cutoff
            self.drawandsave(clossess,savefolder+'Clossess_OUT.txt','Clossess_OUT')            
            result.append(clossess)
            # clossess = gg.closeness(vertices=None, mode='ALL', cutoff =None, weights='w')#cutoff
            # self.drawandsave(clossess,savefolder+'Clossess_ALL_Weighted.txt','Clossess_ALL_Weighted')
            # clossess = gg.closeness(vertices=None, mode='IN', cutoff =None, weights='w')#cutoff
            # self.drawandsave(clossess,savefolder+'Clossess_IN_Weighted.txt','Clossess_IN_Weighted')
            # clossess = gg.closeness(vertices=None, mode='OUT', cutoff =None, weights='w')#cutoff
            # self.drawandsave(clossess,savefolder+'Clossess_OUT_Weighted.txt','Clossess_OUT_Weighted')
            #            eigenvector_centrality = gg.eigenvector_centrality(directed=True, scale=True, weights=None,return_eigenvalue=False)
            #            print eigenvector_centrality
            
            eccentricity = gg.eccentricity(vertices=None, mode='ALL')
            self.drawandsave(eccentricity,savefolder+'Eccentricity_ALL.txt','Eccentricity_ALL') 
            result.append(eccentricity)
            eccentricity = gg.eccentricity(vertices=None, mode='IN')
            self.drawandsave(eccentricity,savefolder+'Eccentricity_IN.txt','Eccentricity_IN') 
            result.append(eccentricity)
            eccentricity = gg.eccentricity(vertices=None, mode='OUT')
            self.drawandsave(eccentricity,savefolder+'Eccentricity_OUT.txt','Eccentricity_OUT') 
            result.append(eccentricity)

            pagerank = gg.pagerank(vertices=None, directed=True, damping=0.85, weights=None,arpack_options=None)
            self.drawandsave(pagerank,savefolder+'Pagerank.txt','Pagerank') 
            result.append(pagerank)
            # pagerankw = gg.pagerank(vertices=None, directed=True, damping=0.85, weights='w',arpack_options=None)
            # self.drawandsave(pagerankw,savefolder+'Pagerank_Weighted.txt','Pagerank_Weighted')

            if False:
                "network profile"
                vcount=g.vcount()
                ecount=g.ecount()
                vcountg=gg.vcount()
                ecountg=gg.ecount()
                print vcount,ecount,vcountg,ecountg,
                density = g.density();print density
                transitivity_undirected = gg.transitivity_undirected(mode='0');print transitivity_undirected
                assorDegD=gg.assortativity(degree,directed= True) ;print assorDegD
                diameter = gg.diameter();print diameter
                radius = gg.radius(mode='ALL');print radius
                aslength = gg.average_path_length();print aslength
                #            print assorDegD
                #            assorInDeg=gg.assortativity(indegree,directed= True)
                #            assorOutDeg=gg.assortativity(outdegree,directed= True)
                profile = [['vcount','ecount','vcountg','ecountg','density','transitivity_undirected','diameter','radius','aslength','assorDegD'],[vcount,ecount,vcountg,ecountg,density,transitivity_undirected,diameter,radius,aslength,assorDegD]]
                print profile
                self.drawandsave(profile,savefolder+'profile.txt','Profile')
            
            #         except Exception, e:
            #             print e
            return result

    def analyzeNetNodes_more(self,graphme,savefolder):
            '''IN: graph instance and the attribute save folder
            OUT:20 attribute files of the graph nodes
            Degree,Indegree,Outdegree,Betweeness,Shell,Coreness_ALL,Coreness_IN,Coreness_OUT,Clossess_ALL,Clossess_IN,Clossess_OUT,Eccentricity_ALL,Eccentricity_IN,Eccentricity_OUT,Pagerank
            '''
            result = []
#         try:
#             g=graphme
#             gg=clus.VertexClustering.giant(g.clusters(mode='weak'))
            gg=graphme
            gg.as_directed()
            degree=gg.degree()
            indegree=gg.indegree() 
            outdegree=gg.outdegree()
            self.drawandsave(degree,savefolder+'Degree.txt','Degree') 
            self.drawandsave(indegree,savefolder+'Indegree.txt','Indegree') 
            self.drawandsave(outdegree,savefolder+'Outdegree.txt','Outdegree')
            result.append(degree)
            result.append(indegree)
            result.append(outdegree)
            
            bet = gg.betweenness(vertices=None, directed=True, cutoff =None, weights=None,nobigint=True)
            self.drawandsave(bet,savefolder+'Betweeness.txt','Betweeness')
            result.append(bet)
            # bet = gg.betweenness(vertices=None, directed=True, cutoff =None, weights='w',nobigint=True)
            # self.drawandsave(bet,savefolder+'Betweeness_Weighted.txt','Betweeness_Weighted')
            #            kcore = gg.kcore()
            #            print assorDegD
            shell = gg.shell_index(mode='ALL')
            self.drawandsave(shell,savefolder+'Shell.txt','Shell')
            result.append(shell)
            coreness = gg.coreness(mode='ALL')#Reference: Vladimir Batagelj, Matjaz Zaversnik: An O(m) Algorithm for Core Decomposition of Networks.
            self.drawandsave(coreness,savefolder+'Coreness_ALL.txt','Coreness_ALL') 
            result.append(coreness)
            coreness = gg.coreness(mode='IN')#Reference: Vladimir Batagelj, Matjaz Zaversnik: An O(m) Algorithm for Core Decomposition of Networks.
            self.drawandsave(coreness,savefolder+'Coreness_IN.txt','Coreness_IN') 
            result.append(coreness)
            coreness = gg.coreness(mode='OUT')#Reference: Vladimir Batagelj, Matjaz Zaversnik: An O(m) Algorithm for Core Decomposition of Networks.
            self.drawandsave(coreness,savefolder+'Coreness_OUT.txt','Coreness_OUT') 
            result.append(coreness)

            clossess = gg.closeness(vertices=None, mode='ALL', cutoff =None, weights=None)#cutoff
            self.drawandsave(clossess,savefolder+'Clossess_ALL.txt','Clossess_ALL') 
            result.append(clossess)
            clossess = gg.closeness(vertices=None, mode='IN', cutoff =None, weights=None)#cutoff
            self.drawandsave(clossess,savefolder+'Clossess_IN.txt','Clossess_IN') 
            result.append(clossess)
            clossess = gg.closeness(vertices=None, mode='OUT', cutoff =None, weights=None)#cutoff
            self.drawandsave(clossess,savefolder+'Clossess_OUT.txt','Clossess_OUT')            
            result.append(clossess)
            # clossess = gg.closeness(vertices=None, mode='ALL', cutoff =None, weights='w')#cutoff
            # self.drawandsave(clossess,savefolder+'Clossess_ALL_Weighted.txt','Clossess_ALL_Weighted')
            # clossess = gg.closeness(vertices=None, mode='IN', cutoff =None, weights='w')#cutoff
            # self.drawandsave(clossess,savefolder+'Clossess_IN_Weighted.txt','Clossess_IN_Weighted')
            # clossess = gg.closeness(vertices=None, mode='OUT', cutoff =None, weights='w')#cutoff
            # self.drawandsave(clossess,savefolder+'Clossess_OUT_Weighted.txt','Clossess_OUT_Weighted')
            #            eigenvector_centrality = gg.eigenvector_centrality(directed=True, scale=True, weights=None,return_eigenvalue=False)
            #            print eigenvector_centrality
            
            eccentricity = gg.eccentricity(vertices=None, mode='ALL')
            self.drawandsave(eccentricity,savefolder+'Eccentricity_ALL.txt','Eccentricity_ALL') 
            result.append(eccentricity)
            eccentricity = gg.eccentricity(vertices=None, mode='IN')
            self.drawandsave(eccentricity,savefolder+'Eccentricity_IN.txt','Eccentricity_IN') 
            result.append(eccentricity)
            eccentricity = gg.eccentricity(vertices=None, mode='OUT')
            self.drawandsave(eccentricity,savefolder+'Eccentricity_OUT.txt','Eccentricity_OUT') 
            result.append(eccentricity)

            pagerank = gg.pagerank(vertices=None, directed=True, damping=0.85, weights=None,arpack_options=None)
            self.drawandsave(pagerank,savefolder+'Pagerank.txt','Pagerank') 
            result.append(pagerank)
            # pagerankw = gg.pagerank(vertices=None, directed=True, damping=0.85, weights='w',arpack_options=None)
            # self.drawandsave(pagerankw,savefolder+'Pagerank_Weighted.txt','Pagerank_Weighted')

#             clustercc = gg(vertices=None, directed=True, damping=0.85, weights=None,arpack_options=None)
#             self.drawandsave(pagerank,savefolder+'Pagerank.txt','Pagerank') 
#             result.append(pagerank)
            "++++++++++++++++++diversity,eccentricity"            
            diversity = gg.diversity()
            self.drawandsave(diversity,savefolder+'Diversity.txt','Diversity') 
            result.append(diversity)
            eccentricity = gg.eccentricity()
            self.drawandsave(eccentricity,savefolder+'Eccentricity.txt','Eccentricity') 
            result.append(eccentricity)

            if True:
                "network profile"
                vcount=gg.vcount()
                ecount=gg.ecount()
                vcountg=gg.vcount()
                ecountg=gg.ecount()
                print vcount,ecount,vcountg,ecountg,
                density = gg.density();print density
                transitivity_undirected = gg.transitivity_undirected(mode='0');print transitivity_undirected
                assorDegD=gg.assortativity(degree,directed= True) ;print assorDegD
                diameter = gg.diameter();print diameter
                radius = gg.radius(mode='ALL');print radius
                aslength = gg.average_path_length();print aslength
                print assorDegD
                "++++++++++++++++++++++++assorInDeg,assorOutDeg,biconnected_components,triad_census"
                assorInDeg=gg.assortativity(indegree,directed= True)
                assorOutDeg=gg.assortativity(outdegree,directed= True)
                biconnected_components = 'ignore'#gg.biconnected_components()
                triad_census = 'ignore'#gg.triad_census()

#                 omega()
#                 alpha()
#                 evcent(directed =True, scale =True, weights =None, return_eigenvalue =False,arpack_options =None)
#                 vertex_disjoint_paths(source =-1, target =-1, checks =True,neighbors ="ignore")
#                 edge_disjoint_paths(source =-1, target =-1, checks =True)
                
                
#                 g.alpha()
#                 g.biconnected_components()
#                 g.Bipartite(method)
#                 g.clusters()
#                 g.cohesion()
#                 g.community_leading_eigenvector_naive()
#                 g.count_automorphisms_vf2()
#                 g.cut_vertices()
#                 g.degree_distribution()
#                 g.edge_disjoint_paths()
#                 g.evcent()
#                 g.get_adjedgelist()
#                 g.get_adjlist()
#                 g.get_automorphisms_vf2()
#                 g.get_inclist()
#                 g.indegree()
#                 g.k_core()
#                 g.maximum_bipartite_matching()
#                 g.omega()
#                 g.outdegree()
#                 g.pagerank()
#                 g.shell_index()
#                 g.shortest_paths_dijkstra()
#                 g.spanning_tree()
#                 g.subgraph()
#                 
#                 
#                 "====="
#                 
#                 g.assortativity()
#                 g.assortativity_degree()
#                 g.assortativity_nominal()
#                 g.Asymmetric_Preference()
#                 g.Atlas()
#                 g.attributes()
#                 g.authority_score()
#                 g.average_path_length()
#                 g.Barabasi()
#                 g.betweenness()
#                 g.bfs()
#                 g.bfsiter()
#                 g.bibcoupling()
#                 g.biconnected_components()
#                 g.bipartite_projection()
#                 g.bipartite_projection_size()
#                 g.cliques()
#                 g.closeness()
#                 g.clusters()
#                 g.cocitation()
#                 g.cohesive_blocks()
#                 g.community_edge_betweenness()
#                 g.community_fastgreedy()
#                 g.community_infomap()
#                 g.community_label_propagation()
#                 g.community_leading_eigenvector()
#                 g.community_multilevel()
#                 g.community_optimal_modularity()
#                 g.community_spinglass()
#                 g.community_walktrap()
#                 g.complementer()
#                 g.compose()
#                 g.constraint()
#                 g.contract_vertices()
#                 g.convergence_degree()
#                 g.convergence_field_size()
#                 g.copy()
#                 g.count_isomorphisms_vf2()
#                 g.count_multiple()
#                 g.count_subisomorphisms_vf2()
#                 g.De_Bruijn()
#                 g.decompose()
#                 g.degree()
#                 g.Degree_Sequence()
#                 g.delete_edges()
#                 g.delete_vertices()
#                 g.density()
#                 g.diameter()
#                 g.difference()
#                 g.disjoint_union()
#                 g.diversity()
#                 g.dyad_census()
#                 g.eccentricity()
#                 g.ecount()
#                 g.edge_attributes()
#                 g.edge_betweenness()
#                 g.Erdos_Renyi()
#                 g.Establishment()
#                 g.Famous()
#                 g.farthest_points()
#                 g.feedback_arc_set()
#                 g.Forest_Fire()
#                 g.Full()
#                 g.Full_Citation()
#                 g.get_adjacency()
#                 g.get_all_shortest_paths()
#                 g.get_diameter()
#                 g.get_edgelist()
#                 g.get_eid()
#                 g.get_eids()
#                 g.get_incidence()
#                 g.get_isomorphisms_vf2()
#                 g.get_shortest_paths()
#                 g.get_subisomorphisms_vf2
#                 g.girth()
#                 g.Growing_Random()
#                 g.has_multiple()
#                 g.hub_score()
#                 g.incident()
#                 g.independent_vertex_sets()
#                 g.intersection()
#                 g.is_bipartite()
#                 g.is_connected()
#                 g.is_dag()
#                 g.is_directed()
#                 g.is_loop()
#                 g.is_minimal_separator()
#                 g.is_multiple()
#                 g.is_mutual()
#                 g.is_separator()
#                 g.is_simple()
#                 g.Isoclass()
#                 g.isoclass()
#                 g.isomorphic()
#                 g.isomorphic_bliss()
#                 g.isomorphic_vf2()
#                 g.Kautz()
#                 g.knn()
#                 g.laplacian()
#                 g.largest_cliques()
#                 g.largest_independent_vertex_sets()
#                 g.Lattice()
#                 g.layout_circle()
#                 g.layout_drl()
#                 g.layout_fruchterman_reingold()
#                 g.layout_graphopt()
#                 g.layout_grid()
#                 g.layout_grid_fruchterman_reingold()
#                 g.layout_kamada_kawai()
#                 g.layout_lgl()
#                 g.layout_mds()
#                 g.layout_random()
#                 g.layout_reingold_tilford()
#                 g.layout_reingold_tilford_circular()
#                 g.layout_star()
#                 g.LCF()
#                 g.linegraph()
#                 g.maxdegree()
#                 g.maxflow()
#                 g.maxflow_value()
#                 g.maximal_cliques()
#                 g.maximal_independent_vertex_sets()
#                 g.mincut()
#                 g.mincut_value()
#                 g.minimum_size_separators()
#                 g.modularity()
#                 g.motifs_randesu()
#                 g.motifs_randesu_estimate()
#                 g.motifs_randesu_no()
#                 g.neighborhood()
#                 g.neighborhood_size()
#                 g.neighbors()
#                 g.path_length_hist()
#                 g.permute_vertices()
#                 g.personalized_pagerank()
#                 g.predecessors()
#                 g.Preference()
#                 g.radius()
#                 g.Read_DIMACS()
#                 g.Read_DL()
#                 g.Read_Edgelist()
#                 g.Read_GML()
#                 g.Read_GraphDB()
#                 g.Read_GraphML()
#                 g.Read_Lgl()
#                 g.Read_Ncol()
#                 g.Read_Pajek()
#                 g.Recent_Degree()
#                 g.reciprocity()
#                 g.rewire()
#                 g.rewire_edges()
#                 g.Ring()
#                 g.similarity_dice()
#                 g.similarity_inverse_log_weighted()
#                 g.similarity_jaccard()
#                 g.simplify()
#                 g.Star()
#                 g.Static_Fitness()
#                 g.Static_Power_Law()
#                 g.strength()
#                 g.subcomponent()
#                 g.subgraph_edges()
#                 g.subisomorphic_vf2()
#                 g.successors()
#                 g.to_directed()
#                 g.to_undirected()
#                 g.topological_sorting()
#                 g.transitivity_avglocal_undirected()
#                 g.transitivity_local_undirected()
#                 g.transitivity_undirected()
#                 g.Tree()
#                 g.triad_census()
#                 g.unfold_tree()
#                 g.union()
#                 g.vcount()
#                 g.vertex_attributes()
                
                
                
                profile = [['vcount','ecount','vcountg','ecountg','density','transitivity_undirected','diameter','radius','aslength','assorDegD','assorInDeg','assorOutDeg','triad_census','biconnected_components'],[vcount,ecount,vcountg,ecountg,density,transitivity_undirected,diameter,radius,aslength,assorDegD,assorInDeg,assorOutDeg,triad_census,list(biconnected_components)]]
                print profile
                self.drawandsave(profile,savefolder+'profile.txt','Profile')
            
            #         except Exception, e:
            #             print e
            return result


def coc2gml(cocfilepath,coclineseprator='\t',gmlfilepath='IHaveNoName.gml'):
    ##将图以节点对方式读入,含有其它边属性，输出gml格式
#     print time.clock()
    import networkx as nx
    #单边向DiGraph
    #             G=nx.read_edgelist(inpath, delimiter='\t', create_using=nx.Graph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    #单边有向DiGraph
    #             G=nx.read_edgelist(inpath, delimiter='\t', create_using=nx.DiGraph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    #多边无向MultiGraph
#     G=nx.read_edgelist(cocfilepath, delimiter=coclineseprator, create_using=nx.MultiGraph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    #多边有向MultiDiGraph
    G=nx.read_edgelist(cocfilepath, delimiter=coclineseprator, create_using=nx.DiGraph(),data=(('w',float),('r',int)))
    gmlfile = open(gmlfilepath,'w')
    nx.write_gml(G,gmlfile)
    gmlfile.close()
    return gmlfilepath

def read_weighted_graph(file_path):
        g = igraph.Graph(directed = True)
        g.add_vertices(100000)

        with open(file_path) as f:
                for line in f:
                        tokens = line.split("\t")
                        s_id = int(tokens[0])
                        t_id = int(tokens[1])
                        w = float(tokens[2])
                        g.add_edge(s_id,t_id, weight=w)
        return g

#     IG = ig.load(gfile, format='edgelist')
#     
#     IG = read_weighted_graph(graph_file_path)

                  
if __name__=='__main4cui__':
    workfolder = "G:\\HFS\\WeiboData\\Cui\\"
    
    gfile = open(workfolder+'100k_sample_unique_TE_igraph_weight.txt')#test.coc')#'G:\HFS\WeiboData\HFSWeiboNoCOC\3342670838100183.coc'#  
    gml = coc2gml(gfile,coclineseprator='\t',gmlfilepath=workfolder+'100k_sample_unique_TE_igraph_weight.gml');print 'coc2gml done'
#     g = ig.Graph.Read_Edgelist(gfile)
    gmlf = file(gml)
    g = ig.Graph.Read_GML(gmlf);print 'ig.Graph.Read_GML(gmlf) done'
    gmlf.close()
    netan = analyzeNet_AllNodes()
    netan.analyzeNetNodes(g,savefolder = workfolder+"giantatt\\")
#     resultone.extend(featuresofgraph(g))
#     result.append(resultone)
    
if __name__=='__main4cui__2':
    workfolder = "G:\\HFS\\WeiboData\\Cui\\giantatt2\\"
    for filen in os.listdir(workfolder):
        path = workfolder+filen
        filenlist = os.path.splitext(filen) 
        if os.path.isfile(path) and filenlist[-1]=='.txt': 
            print path 
            lista = csv.reader(file(path)) 
            xlabel = filenlist[0]
            for line in lista:
                print line
                gt.list_2_Distribution([line],xlabels=[xlabel],ylabels=['Frequency'],showfig=True,binseqdiv=20)



if __name__=='__main__2':
    workfolder = "I:\\dataset\\HFS_XunRen\\User\\"
    workfolder_gml = "I:\\dataset\\HFS_XunRen\\User\\GML\\"
    filecnt = 0
    for gfile in os.listdir(workfolder_gml):
        
#         try:
            gfilep = workfolder_gml+gfile
            print filecnt,gfilep,' start...',time.asctime()
            g = ig.Graph.Read_GML(gfilep)
            #grt.analyzeNetNodes_New(g,savefolder = workfolder+"ATTUser\\All\\",graphid='User_Nodes_All',keepold=True,saveMode='a+')
            netan = analyzeNet_AllNodes()

            User_Nodes_All = netan.analyzeNetNodes(g,savefolder = workfolder+"ATTUser\\All\\")
            User_Nodes_All.insert(0,g.vs['label'])
            graphid = []
            gname = os.path.splitext(gfile)[0]
            for i in xrange(len(User_Nodes_All[0])):
                graphid.append(gname)
            User_Nodes_All.insert(0,graphid)
            User_Nodes_Allz = zip(*User_Nodes_All)
            if filecnt<2:
                User_Nodes_Allz.insert(0,['Wbid','Userid','Degree','Indegree','Outdegree','Betweeness','Shell','Coreness_ALL','Coreness_IN','Coreness_OUT','Clossess_ALL','Clossess_IN','Clossess_OUT','Eccentricity_ALL','Eccentricity_IN','Eccentricity_OUT','Pagerank'])
            gt.saveList(User_Nodes_Allz,workfolder+"ATTUser\\All\\User_Nodes_All.att",writype='a+')
            filecnt+=1



            gg=clus.VertexClustering.giant(g.clusters(mode='weak'))
            User_Nodes_Giant = netan.analyzeNetNodes(gg,savefolder = workfolder+"ATTUser\\Giant\\")

            User_Nodes_Giant.insert(0,g.vs['label'])
            graphid = []
            gname = os.path.splitext(gfile)[0]
            for i in xrange(len(User_Nodes_Giant[0])):
                graphid.append(gname)
            User_Nodes_Giant.insert(0,graphid)
            User_Nodes_Giantz = zip(*User_Nodes_Giant)
            if filecnt<2:
                User_Nodes_Giantz.insert(0,['Wbid','Userid','Degree','Indegree','Outdegree','Betweeness','Shell','Coreness_ALL','Coreness_IN','Coreness_OUT','Clossess_ALL','Clossess_IN','Clossess_OUT','Eccentricity_ALL','Eccentricity_IN','Eccentricity_OUT','Pagerank'])
            gt.saveList(User_Nodes_Giantz,workfolder+"ATTUser\\Giant\\User_Nodes_Giant.att",writype='a+')
            #grt.analyzeNetNodes_New(gg,savefolder = workfolder+"ATTUser\\Giant\\",graphid='User_Nodes_Giant',keepold=True,saveMode='a+')
            #     resultone.extend(featuresofgraph(g))
            #     result.append(resultone)
#         except Exception,e:
#             print gfilep,'========error:',e


if __name__=='__main2__':
    workfolder = "I:\\dataset\\HFS_XunRen\\User\\ATTUser\\All\\"
    for filen in os.listdir(workfolder):
        path = workfolder+filen
        filenlist = os.path.splitext(filen)
        if os.path.isfile(path) and filenlist[-1]=='.txt':
            print path
            lista = csv.reader(file(path))
            xlabel = filenlist[0]
            res = []
            for line in lista:
                res.extend(line)
            try:
                gt.list_2_Distribution([res],xlabels=[xlabel],ylabels=['Frequency'],showfig=True,binseqdiv=20)
            except:
                pass



if __name__=='__main__':
    workfolder = "I:\\dataset\\HFS_XunRen\\User\\"
    workfolder_gml = "I:\\dataset\\HFS_XunRen\\User\\GML\\MT9\\"
    filecnt = 0
#     for gfile in os.listdir(workfolder_gml):
    if True:
        gfile = 'user_network_mt9_2014.gml'
        try:
            gfilep = workfolder_gml+gfile
            print filecnt,gfilep,' start...',time.asctime()
            g = ig.Graph.Read_GML(gfilep)
            #grt.analyzeNetNodes_New(g,savefolder = workfolder+"ATTUser_community\\All\\",graphid='User_Nodes_All',keepold=True,saveMode='a+')
            netan = analyzeNet_AllNodes()

            User_Nodes_All = netan.analyzeNetNodes_more(g,savefolder = workfolder+"ATTUser_community\\All\\")
            User_Nodes_All.insert(0,g.vs['label'])
            graphid = []
            gname = os.path.splitext(gfile)[0]
            for i in xrange(len(User_Nodes_All[0])):
                graphid.append(gname)
            User_Nodes_All.insert(0,graphid)
            User_Nodes_Allz = zip(*User_Nodes_All)
            if filecnt<2:
                User_Nodes_Allz.insert(0,['Wbid,Userid,Degree,Indegree,Outdegree,Betweeness,Shell,Coreness_ALL,Coreness_IN,Coreness_OUT,Clossess_ALL,Clossess_IN,Clossess_OUT,Eccentricity_ALL,Eccentricity_IN,Eccentricity_OUT,Pagerank,Diversity,Eccentricity'])
            gt.saveList(User_Nodes_Allz,workfolder+"ATTUser_community\\All\\User_Nodes_All.att",writype='a+')
            filecnt+=1



            gg=clus.VertexClustering.giant(g.clusters(mode='weak'))
            User_Nodes_Giant = netan.analyzeNetNodes_more(gg,savefolder = workfolder+"ATTUser_community\\Giant\\")

            User_Nodes_Giant.insert(0,g.vs['label'])
            graphid = []
            gname = os.path.splitext(gfile)[0]
            for i in xrange(len(User_Nodes_Giant[0])):
                graphid.append(gname)
            User_Nodes_Giant.insert(0,graphid)
            User_Nodes_Giantz = zip(*User_Nodes_Giant)
            if filecnt<2:
                User_Nodes_Giantz.insert(0,['Wbid,Userid,Degree,Indegree,Outdegree,Betweeness,Shell,Coreness_ALL,Coreness_IN,Coreness_OUT,Clossess_ALL,Clossess_IN,Clossess_OUT,Eccentricity_ALL,Eccentricity_IN,Eccentricity_OUT,Pagerank,Diversity,Eccentricity'])
            gt.saveList(User_Nodes_Giantz,workfolder+"ATTUser_community\\Giant\\User_Nodes_Giant.att",writype='a+')
            #grt.analyzeNetNodes_New(gg,savefolder = workfolder+"ATTUser_community\\Giant\\",graphid='User_Nodes_Giant',keepold=True,saveMode='a+')
            #     resultone.extend(featuresofgraph(g))
            #     result.append(resultone)
        except Exception,e:
            print gfilep,'========error:',e

