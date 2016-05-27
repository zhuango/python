import subprocess
import re
def findProcess( processId ):
    #linux
    ps= subprocess.Popen("ps -ef | grep "+processId, shell=True, stdout=subprocess.PIPE)
    #windows
    #ps= subprocess.Popen(r'tasklist.exe /NH /FI "PID eq %d"' % processId, shell=True, stdout=subprocess.PIPE)
    output = ps.stdout.read()
    ps.stdout.close()
    ps.wait()
    return output
def isProcessRunning( processId):
    output = findProcess( processId )
    if re.search(str(processId), output) is None:
        return False
    else:
        return True

if __name__ == '__main__': 
    #check_exsit('chrome.exe')
    print(isProcessRunning(16664))