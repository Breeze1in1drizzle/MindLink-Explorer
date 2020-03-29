# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:46:36 2018
Modified on Tue Nov 12 20:56:40 2019
@author1: Yongrui Huang
@author2: Ruixin Lee
"""

import os


def find_pid_from_name(process_name):
    '''
        This method is used for finding the process id by their name.
        Some process' name corresponds multiple process, we only return the 
        first one because when the first one is killed, the other will die too.
        
        Arguments:
            
            process_name: the name of the process
            
        Returns:
            
            process_exist: whether the process_exist, return True if exist
            
            pid: the process' id, string
    '''
    cmd = 'wmic process where name="%s" get processid' % process_name
    output = os.popen(cmd).read()
    
    pid = output.split('\n')[2]
    process_exist = len(pid) > 0
    
    return process_exist, pid


def close_browser():
    '''
        This method is used for closing the browser
        TODO: make it work even in Linux, it only works in windows because the command belongs to windows.
    '''
    # When the operate system is windows
    bowers_process_names = [
        'Chrome.exe', 'Firefox.exe', 'iexplore.exe',
        '360SE.exe', 'sogouexplorer.exe', 'opera.exe',
        'Safari.exe', 'Maxthon.exe', 'Netscape.exe'
    ]
    for name in bowers_process_names:
        process_exist, pid = find_pid_from_name(name)
        if process_exist:
            command = 'taskkill /PID ' + pid + ' /F'
            os.popen(command).read()


def find_pid_from_output(output):
    '''
        This method is used for finding the pid from command's one-line output.
        We only kill the process whose status is listening.
        Arguments:
            output: a string from command
        Returns:
            the pid
    '''
    pid_tail = len(output) - 1
    # if it is not a line
    if pid_tail <= 0:
        return 0
    # find the pid
    while ord(output[pid_tail]) >= ord('0') and ord(output[pid_tail]) <= ord('9'):
        pid_tail -= 1
    # find the status, we only kill the one which is listening
    status_tail = pid_tail
    while output[status_tail] == ' ':
        status_tail -= 1
        continue
#    G is the last alphabet of the word LISTENING
#    We return the process's id if its status is LISTENING
#    Four status for a process in TCP:
#    LISTENING, ESTABLISHED, TIME_WAIT and CLOSE_WAIT
    if output[status_tail] == 'G':
        return output[pid_tail+1:]
    else:
        return 0


def kill_process_by_port(port):
    '''
        This method is used for killing the process by its port
        Specially, we use it to kill the falsk when the experiment is over.
        TODO: apply it in Linux. Note that this method works only for Windows
        so far, since its API belongs to Windows. Specially, 'the kill_command'
        Arguments:
            port: a string describes the number of the port the process is taking.
    '''
    find_port_id_command = 'netstat -ano | findstr :5000'
    outputs = os.popen(find_port_id_command).read().split('\n')
#    It should be noted that the outputs contains multiply line.
#    Here is an example of the outputs
#    TCP    127.0.0.1:5000         0.0.0.0:0              LISTENING       1012
#    TCP    127.0.0.1:5000         127.0.0.1:65261        TIME_WAIT       0
#    TCP    127.0.0.1:5000         127.0.0.1:65262        ESTABLISHED     1012
#    TCP    127.0.0.1:5000         127.0.0.1:65266        TIME_WAIT       0
#    TCP    127.0.0.1:65262        127.0.0.1:5000         ESTABLISHED     8032
    for output in outputs:
        pid = find_pid_from_output(output)
        if pid != 0:
            kill_command = 'taskkill /PID ' + str(pid) +' /F'
            os.popen(kill_command)

# errors
def dat2csv():
    import os
    # open('1.txt', encoding='gbk')
    # path1 = 'C:\\Users\\awake_ljw\\Documents\\python for data analysis\\test1'
    path1_load = 'D:\\workSpace\\python_workspace\\Eumpy-master\\data_collection_framework\\util\\dat2csv'
    # path2 = 'C:\\Users\\awake_ljw\\Documents\\python for data analysis\\test2'
    path2_load = 'D:\\workSpace\\python_workspace\\Eumpy-master\\data_collection_framework\\util\\csv'
    filelist = os.listdir(path1_load)

    for files in filelist:
        Olddir = os.path.join(path1_load, files)
        filename = os.path.splitext(files)[0]
        filetype = os.path.splitext(files)[1]
        print
        Olddir
        file_test = open(Olddir, 'r', encoding=' gb18030', errors='ignore')
        Newdir = os.path.join(path2_load, str(filename) + '.csv')
        print
        Newdir
        file_test2 = open(Newdir, 'w')
        for lines in file_test.readlines():
            strdata = ",".join(lines.split('\t'))
            file_test2.write(strdata)
        file_test.close()
        file_test2.close()
# errors


if __name__ == "__main__":
    # dat2csv()
    pass
