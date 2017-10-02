import os
import igraph as ig
from igraph.drawing import plot
from igraph import clustering as clus

g = ig.Graph.Full(3)
vs = ig.VertexSeq(g)
print vs
def gml2core(gmlfolder,svgfolder):
    fw=open(svgoutpath+'vecount.txt','w')
    svgtype=['_weakGaint','_strongGaint','_weakGaintSPT','_strongGaintSPT']
    if os.path.exists(svgoutpath)==0:
        os.mkdir(svgoutpath)
    for file in os.listdir(gmlinpath):
        if os.path.splitext(file)[1]=='.gml':
            print 'Reading graph from ...',file
            try:
                gmlfile=open(gmlinpath+file)
                g = ig.Graph.Read_GML(gmlfile)
                gg=clus.VertexClustering.giant(g.clusters(mode='weak'))
                print g.vcount(),gg.vcount()
#                 es=ig.EdgeSeq(gg)
                vs=ig.VertexSeq(gg)
                for v in vs:
                    indeg = v.indegree()
                    outdeg = v.outdegree()
                    if indeg<1 or outdeg<1:
                        gg.delete_vertices(v)
#                     if outdeg<1:
#                         gg.delete_vertices(v)
                print gg.vcount()   
                ig.write(gg,svgoutpath+file+svgtype[1]+'.gml')    
    #             ig.Graph.write_svg(gg, svgoutpath+file+svgtype[1]+'.svg', layout='large') 
                    
            except Exception,e:
                print gmlinpath+file,' failed',e
                pass
                gmlfile.close()
    fw.close()
    print 'all done'
    
    
def gml2svg(gmlfolder,svgfolder):
    fw=open(svgoutpath+'vecount.txt','w')
    svgtype=['_weakGaint','_strongGaint','_weakGaintSPT','_strongGaintSPT']
    if os.path.exists(svgoutpath)==0:
        os.mkdir(svgoutpath)
    for file in os.listdir(gmlinpath):
        if os.path.splitext(file)[1]=='.gml':
            print 'Reading graph from ...',file
            if os.path.exists(svgoutpath+file+svgtype[3]+'.2svg'):
                print file,'has existesed'
                pass
            else:
                try:
                    gmlfile=open(gmlinpath+file)
                    g = ig.Graph.Read_GML(gmlfile)
                    gg=clus.VertexClustering.giant(g.clusters(mode='strong'))
                    es=ig.EdgeSeq(gg)
                    subg = gg.subgraph_edges(es.select(retwitype_eq = '0'))
                    es=ig.EdgeSeq(subg)
                    timelist=map(float,es.get_attribute_values('time'))
                    gsp=ig.Graph.spanning_tree(subg,timelist)
                    
                    vecountstr = str(g.vcount())+'\t'+str(g.ecount())+'\t'+str(gg.vcount())+'\t'+str(gg.ecount())+'\t'+str(subg.vcount())+'\t'+str(subg.ecount())+'\t'+str(gsp.vcount())+'\t'+str(gsp.ecount())
                    fw.write(file+'\t'+vecountstr+'\n')
                    if os.path.exists(svgoutpath+file+svgtype[3]+'.svg'):
                        print file,'has existesed'
                    else:
                        print 'Ploting graph'
                        ig.Graph.write_svg(subg, svgoutpath+file+svgtype[1]+'.svg', layout='large')
                        ig.Graph.write_svg(gsp, svgoutpath+file+svgtype[3]+'.svg', layout='large')
                    layout = gsp.layout("large")
                    fig = plot(gsp, layout = layout)
                    plot.show()
            #         ig.Graph.write_svg(gsp, svgoutpath+file+'_w.svg', layout='large')
                    #.save(gmlinpath+file+'.fig')
                except Exception,e:
                    print gmlinpath+file,' failed',e
                    pass
                gmlfile.close()
    fw.close()
    print 'all done'



gmlinpath='G:\\HFS\\WeiboData\\HFSWeiboGML\\'
svgoutpath='G:\\HFS\\WeiboData\\HFSWeiboGML\\SVG\\'#
gml2core(gmlinpath,svgoutpath)