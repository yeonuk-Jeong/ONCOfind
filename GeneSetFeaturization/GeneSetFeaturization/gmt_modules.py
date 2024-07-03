
def LoadGMT2List(File):
    with open(File, 'r', errors='replace') as f:
        l1 = f.readlines()
    return l1

def Str2Dic(D):
    l2 = D.split('\n')[0].split('\t')
    l2 = [s for s in l2 if len(s) > 0]
    return {l2[0]:l2[2:]}

def List2Dict(List):
    for i in range(len(List)):
        if i == 0:
            Dic1 = Str2Dic(List[i])
        else:
            Dic1.update(Str2Dic(List[i]))
    return Dic1

def List2Str(List):
    Str = ''
    for l in List:
        Str = Str + "\t" + str(l)
    return Str

def Dic2GMTfile(Dic, Output):
    ResTXT = ''
    for ID in Dic.keys():
        try: ResTXT = ResTXT + ID + '\t' + 'NULL' + List2Str(Dic[ID]) + '\n'
        except: ""

    with open(Output, 'w') as f:
        f.writelines(ResTXT)
        
def LoadGMT2Dict(File):
    return List2Dict(LoadGMT2List(File))
