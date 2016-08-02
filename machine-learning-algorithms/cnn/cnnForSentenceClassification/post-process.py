#! /usr/bin/python3
import re

chemTag = "Chemical_D"
diseTag = "Disease_D"
string = "Chemical_D015738 Disease_D003693, sdf sdfsdf sdfsfd sdf Chemical_D015738-associated Disease_D003693. A series of six cases. Chemical_D015738 is a histamine H2-receptor antagonist used in inpatient settings for prevention of stress Disease_D014456 and is showing increasing popularity because of its low cost. Although all of the currently available H2-receptor antagonists have shown the propensity to cause Disease_D003693, only two previously reported cases have been associated with Chemical_D015738. The authors report on six cases of Chemical_D015738-associated Disease_D003693 in hospitalized patients who cleared completely upon removal of Chemical_D015738. The pharmacokinetics of Chemical_D015738 are reviewed, with no change in its metabolism in the elderly population seen. The implications of using Chemical_D015738 in elderly persons are discussed."
#print(string.index('is'))
def findRelations(string):
    relations = []
    relationWords= ['related', 'during', 'caused', 'associated', 'induced']
    m = re.findall('Chemical_D.*?Disease_D[0-9]*', string)

    for substr in m:
        if len(substr.split(" ")) == 2:
            #print("########%s" % substr)
            relations.append(substr[0:len(chemTag) + 6].split("_")[1] + "\t" +substr[-len(diseTag) - 6:].split("_")[1] + "\t")
            continue
        for rword in relationWords:
            if substr.find(rword):
                relations.append(substr[0:len(chemTag) + 6].split("_")[1] + "\t" +substr[-len(diseTag) - 6:].split("_")[1] + "\t")
                break

    return relations

#relations = findRelations(string)


gold = open("my.gold", "w")
with open("out.txt") as f:
    for line in f:
        id = line.strip().split("|")[0]
        relations = findRelations(line.strip())
        for relation in relations:
            gold.write(id + " " + relation + "\n")