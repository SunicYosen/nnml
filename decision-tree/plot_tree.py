#! /usr/bin/python

import matplotlib.pyplot as plt

decision_node = dict(boxstyle="round4",  fc="0.6")
leaf_node     = dict(boxstyle="round4",  fc="0.8")
arrow_args    = dict(arrowstyle="<-")

# Get leaf number
def get_leaf_num(tree):
    leaf_num = 0
    first_key = list(tree.keys())[0]
    next_dict = tree[first_key]
    for key in next_dict.keys():
        if type(next_dict[key]).__name__=="dict":
            leaf_num +=get_leaf_num(next_dict[key])
        else:
            leaf_num +=1
    return leaf_num

# Get tree depth
def get_tree_depth(tree):
    depth = 0
    first_key = list(tree.keys())[0]
    next_dict = tree[first_key]
    for key in next_dict.keys():
        if type(next_dict[key]).__name__ == "dict":
            thisdepth = 1+ get_tree_depth(next_dict[key])
        else:
            thisdepth = 1
        if thisdepth>depth: depth = thisdepth
    return depth

# plot node
def plot_node(nodeTxt, centerPt, parentPt, nodeType):
    create_plot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

#
def plot_middle_text(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    create_plot.ax1.text(xMid, yMid, txtString, va="center", ha="right", rotation=0)

def plot_tree(tree_dict, parentPt, nodeTxt):
    numLeafs = get_leaf_num(tree_dict)
    depth = get_tree_depth(tree_dict)
    firstStr = list(tree_dict.keys())[0]
    cntrPt = (plot_tree.xOff + (1.0 + float(numLeafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_middle_text(cntrPt, parentPt, nodeTxt)
    plot_node(firstStr, cntrPt, parentPt, decision_node)
    secondDict = tree_dict[firstStr]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':
            plot_tree(secondDict[key], cntrPt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(secondDict[key], (plot_tree.xOff, plot_tree.yOff), cntrPt, leaf_node)
            plot_middle_text((plot_tree.xOff, plot_tree.yOff), cntrPt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD

def create_plot(tree_dict):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_leaf_num(tree_dict))
    plot_tree.totalD = float(get_tree_depth(tree_dict))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(tree_dict, (0.5, 1.0), '')

    plt.rcParams['font.sans-serif']=['SimHei']  # For Chinese label
    plt.rcParams['axes.unicode_minus']=False    # For -
    plt.show()