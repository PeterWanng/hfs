#encoding=utf8

import sys
sys.path.append("../..")
from tools import commontools as gtf

gt = gtf()

sfansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_small_s.txt'
ffansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_small_f.txt'
sfansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_middle_s.txt'
ffansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_middle_f.txt'
sfansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_large_s.txt'
ffansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_large_f.txt'
sfansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_all_s.txt'
ffansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_all_f.txt'

# sfansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_bifans_all_s.txt'
# ffansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_bifans_all_f.txt';


# sfansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_all_ss.txt'
# ffansfp = r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_all_ff.txt'


csvfan_s = gt.csv2list_new(sfansfp)
csvfan_f = gt.csv2list_new(ffansfp)
x = gt.list_2_Distribution_normlized([csvfan_s,csvfan_f],xlabels=['S','F'],ylabels=['Cumulative Frequency',],binseqdiv=1)#
gt.saveList(x,r'G:\MyPapers\HFS_SF\Fig\data\hfs_fans_all_sf_fig.txt')

