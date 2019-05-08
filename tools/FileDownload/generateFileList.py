from pathlib import Path           

def generateFileList(dirname, fs, names, level=3):
    if level == 0:
        return
    dir = Path(dirname)
    dirs = dir.iterdir()
    dirs = sorted(dirs, key=lambda d : d.name, reverse=True)
    for item in dirs:
        if item.is_dir():
            if level == 3:
                fs.write("  <div><b>{}</b></div>\n".format(item.name))
            # <div><b>20180710</b></div>
            # <li> WMNN </li>  
            generateFileList(str(item), fs, names, level - 1)
        if item.is_file():
            filePath = str(item)
            if "Thumbs.db" in filePath:
                continue
            if item.name.startswith("."):
                continue
            if item.name.startswith("~"):
                continue
            # print(str(item)) 
            fs.write("    <li>{}</li> \n".format(item.name))

# <div><a>20180710</a></div>
# <li> WMNN  <span class="paper">Paper</span></li>      

recordsPath = "/WangPan/共享文件夹/开会记录/"
names = {"刘壮", "郎成堃", "宁时贤", "雷弼尊", "刘喆", "lck", "lbz", "lxf", "李雪菲", "liuzhe", "LBZ", "nsx", ""}

filelistPath = "./filelist.txt"
fS = open(filelistPath, 'w')
generateFileList(recordsPath, fS, names)
fS.close()