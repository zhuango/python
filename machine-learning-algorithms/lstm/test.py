
def process_dict1(file_name, enTrain, enTest, cnTrain, cnTest):
    bTrain1 = 1
    eTrain1 = enTrain+1
    bTest1 = enTrain+1
    eTest1 = enTrain+1+enTest
    bTrain2 = enTrain+1+enTest
    eTrain2 = enTrain+1+enTest+cnTrain
    bTest2 = enTrain+1+enTest+cnTrain
    eTest2 = enTrain+1+enTest+cnTrain+cnTest
    fp = open(file_name, 'r')
    dic1 = {}
    dic2 = {}
    dic3 = {}
    dic4 = {}
    nn = 1
    for line in fp:
        if eTrain1 > nn >= bTrain1:
            if line.strip() in dic1:
                dic1[line.strip()].append(nn)
            else:
                dic1[line.strip()] = [nn]
        elif eTest1 > nn >= bTest1:
            if line.strip() in dic1:
                dic2[nn] = dic1[line.strip()]
            # else:
            # 	  dic2[nn] = None
        elif eTrain2 > nn >= bTrain2:
            if line.strip() in dic3:
                dic3[line.strip()].append(nn)
            else:
                dic3[line.strip()] = [nn]
        elif eTest2 > nn >= bTest2:
            if line.strip() in dic3:
                dic4[nn] = dic3[line.strip()]
        nn += 1
    fp.close()
    return dic2, dic4



def similar_(i, lis, tp, length):
    dic = {}
    minn = numpy.inf
    minIndex = -1
    for j in lis:
        # 1: m = numpy.exp(numpy.dot(tp[i, length:], tp[j, length:]))
        m = numpy.sqrt(numpy.sum((tp[i-1, length:]-tp[j-1, length:])**2))
        dic[j-1] = m
        if m <= minn:
            minn = m
            minIndex = j-1
    return dic, minIndex

def process_Wemb(dic2, dic4, tparams, length, oriorpolarityList):
    #tp = tparams['Wemb'].get_value()
    tp = tparams['Wemb']
    # aaa = len(dic2)+len(dic4)
    # nn = 0
    for i in dic2:
        # nn += 1
        # if nn % 1000 == 0:
        #     edd = time.time()
        #     print nn, " of ", aaa, edd
        if(oriorpolarityList[i]):
            index_dic, min_index = similar_(i, dic2[i], tp, length)
        
        # 1: tp[i, length:] = tp[min_index, length:]
        tp[i-1, :length] = tp[min_index, :length]
        """
        tp[i, :length] = 0.0
        for ii in index_dic:
            tp[i, :length] += tp[ii, :length]*index_dic[ii]
        """
    for j in dic4:
        # nn += 1
        # if nn % 1000 == 0:
        #     edd = time.time()
        #     print nn, " of ", aaa, edd
        if(oriorpolarityList[j]):
            index_dic1, min_index1 = similar_(j, dic4[j], tp, length)
        # 1: tp[j, length:] = tp[min_index1, length:]
        tp[j-1, :length] = tp[min_index1, :length]
        """
        tp[j, :length] = 0.0
        for jj in index_dic1:
            tp[j, :length] += tp[jj, :length]*index_dic1[jj]
        """
    #tparams['Wemb'].set_value(tp)

from loadDict import loadPriorpolarityPosList

tparams = {}
tparams['Wemb'] = [1, 2,3 , 4, 5]
afd, acd = process_dict1("G:/liuzhuang/data/music_wordList.txt", 335388, 146881, 369464, 159125)
priorpolarityPosList = loadPriorpolarityPosList("G:/liuzhuang/data/music_wordList.txt")
process_Wemb(afd, acd, tparams, 50, priorpolarityPosList)