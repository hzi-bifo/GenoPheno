__author__ = "Ehsaneddin Asgari"
__license__ = "To be added -- for now all rights are reserved for the author"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "ehsan.asgari@gmail.com"
__source__ = "from personal repo"

import os
GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
import sys
sys.path.append(GENO2PHENO_DIR)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from ete3 import Tree, TreeStyle, NodeStyle, faces, AttrFace, CircleFace, TextFace, RectFace, ProfileFace
from src.utility.file_utility import FileUtility
class VisualizeCircularTree(object):
    '''
        ETE-based CircularTree Visualization
    '''

    def __init__(self, nwk_format, logger=None, overwrite=True):
        self.nwk = nwk_format
        self.logger = logger
        self.overwrite = overwrite

    def create_circle(self, filename, title, name2color=None, name2class_dic=None, class2color_dic=None, vector=None,
                      ignore_branch_length=True):

        plt.clf()
        axis_font = {'size': '10'}
        plt.rc('xtick', labelsize=0.1)
        plt.rc('ytick', labelsize=0.1)
        plt.rc({'font.size': 0.1})



        # legend creation
        if name2class_dic and class2color_dic:
            leg = []
            for cls, color in class2color_dic.items():
                leg.append(mpatches.Patch(color=color, label=str(cls)))

        t = Tree(self.nwk)

        font_size=int(len(list(t.iter_leaves()))/2)

        # iterate over tree leaves only
        for l in t.iter_leaves():
            ns = NodeStyle()
            if name2color:
                ns["bgcolor"] = name2color[l.name] if l.name in name2color else 'white'
            elif name2class_dic and class2color_dic:
                ns["bgcolor"] = class2color_dic[name2class_dic[l.name]] if l.name in name2class_dic and name2class_dic[
                                                                                                            l.name] in class2color_dic else 'white'

            l.img_style = ns
            F = TextFace(l.name)
            F.ftype = 'Times'
            if vector:
                if l.name in vector:
                    l.add_features(profile=vector[l.name])
                    l.add_features(deviation=[0 for x in range(len(vector[l.name]))])
                    l.add_face(ProfileFace(max_v=1, min_v=-1, center_v=0, width=200, height=40, style='heatmap',
                                           colorscheme=1), column=0, position='aligned')
        # Create an empty TreeStyle
        ts = TreeStyle()

        # Set our custom layout function
        ts.layout_fn = VisualizeCircularTree.layout

        # Draw a tree
        ts.mode = "c"

        # We will add node names manually
        ts.show_leaf_name = False
        # Show branch data
        ts.show_branch_length = True
        ts.show_branch_support = True
        ts.force_topology = ignore_branch_length
        ts.title.add_face(TextFace(title, fsize=font_size, ftype='Times'), column=15)

        # legend creation
        if name2class_dic and class2color_dic:
            keys = list(class2color_dic.keys())
            keys.sort()
            for k, cls in enumerate(keys):
                col=class2color_dic[cls]
                x = RectFace(100, 100, 'black', col)
                # x.opacity=0.5
                ts.legend.add_face(x, column=8)
                ts.legend.add_face(TextFace(' ' + str(cls) + '   ', fsize=font_size, ftype='Times'), column=9)

        if not FileUtility.exists(F"{filename}.pdf") or self.overwrite:
            t.render(F"{filename}.pdf", tree_style=ts, dpi=5000)
            if self.logger:
                self.logger.info(F"The phylogenetic visualization is created: {filename}.pdf")
        elif self.logger:
            self.logger.info(F"The phylogenetic visualization already existed: {filename}.pdf")

    @staticmethod
    def layout(node):
        if node.is_leaf():
            # Add node name to laef nodes
            N = AttrFace("name", fsize=14, fgcolor="black")
            faces.add_face_to_node(N, node, 0)
        if "weight" in node.features:
            # Creates a sphere face whose size is proportional to node's
            # feature "weight"
            C = CircleFace(radius=node.weight, color="RoyalBlue", style="sphere")
            # Let's make the sphere transparent
            C.opacity = 0.3
            # And place as a float face over the tree
            faces.add_face_to_node(C, node, 0, position="float")