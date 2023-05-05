
from domain import Record

import os
import time
import threading
import signal
from errno import ENOENT
from subprocess import Popen
from affinity import set_process_affinity_mask

maxtime = 1
delay = 1


class Sample(threading.Thread):
    def __init__(self, program):
        threading.Thread.__init__(self)
        self.setDaemon(1)
        self.timedout = False
        self.p = program
        self.maxMem = 0
        self.childpids = None
        self.start()

    def run(self):
        try:
            remaining = maxtime
            while remaining > 0:
                mem = gtop.proc_mem(self.p).resident
                time.sleep(delay)
                remaining -= delay
                # race condition - will child processes have been created yet?
                self.maxMem = max(
                    (mem + self.childmem())/1024, self.maxMem)
            else:
                self.timedout = True
                os.kill(self.p, signal.SIGKILL)
        except OSError:
            False

    def childmem(self):
        if self.childpids == None:
            self.childpids = set()
            for each in gtop.proclist():
                if gtop.proc_uid(each).ppid == self.p:
                    self.childpids.add(each)
        mem = 0
        for each in self.childpids:
            mem += gtop.proc_mem(each).resident
        return mem

