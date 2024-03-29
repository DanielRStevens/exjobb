# The Computer Language Benchmarks Game
# $Id: planA.py,v 1.1 2012/12/29 19:19:30 igouy-guest Exp $

"""
measure with libgtop2 and CPU affinity
"""
__author__ = 'Isaac Gouy'


from domain import Record

import os
import syspy
import cPickle
import time
import threading
import signal
import gtop
from errno import ENOENT
from subprocess import Popen
from affinity import set_process_affinity_mask


def measure(arg, commandline, delay, maxtime,
            outFile=None, errFile=None, inFile=None, logger=None, affinitymask=None):

    r, w = os.pipe()
    forkedPid = os.fork()

    if forkedPid:  # read pickled measurements from the pipe
        os.close(w)
        rPipe = os.fdopen(r)
        r = cPickle.Unpickler(rPipe)
        measurements = r.load()
        rPipe.close()
        os.waitpid(forkedPid, 0)
        return measurements

    else:
        # Sample thread will be destroyed when the forked process _exits
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
                    if logger:
                        logger.error('%s %s', e, err)

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

        try:
            def setAffinity():
                if affinitymask:
                    set_process_affinity_mask(os.getpid(), affinitymask)

            m = Record(arg)

            # only write pickles to the pipe
            os.close(r)
            wPipe = os.fdopen(w, 'w')
            w = cPickle.Pickler(wPipe)

            # gtop cpu is since machine boot, so we need a before measurement
            cpus0 = gtop.cpu().cpus
            start = time.time()

            # spawn the program in a separate process
            p = Popen(commandline, stdout=outFile, stderr=errFile,
                      stdin=inFile, preexec_fn=setAffinity)

            # start a thread to sample the program's resident memory use
            t = Sample(program=p.pid)

            # wait for program exit status and resource usage
            rusage = os.wait3(0)

            # gtop cpu is since machine boot, so we need an after measurement
            elapsed = time.time() - start
            cpus1 = gtop.cpu().cpus

            # summarize measurements
            if t.timedout:
                m.setTimedout()
            elif rusage[1] == os.EX_OK:
                m.setOkay()
            else:
                m.setError()

            m.userSysTime = rusage[2][0] + rusage[2][1]
            m.maxMem = t.maxMem

            load = map(
                lambda t0, t1:
                int(round(
                    100.0 * (1.0 - float(t1.idle-t0.idle) /
                             (t1.total-t0.total))
                )), cpus0, cpus1)

            # load.sort(reverse=1) # maybe more obvious unsorted
            m.cpuLoad = ("% ".join([str(i) for i in load]))+"%"

            m.elapsed = elapsed

        except KeyboardInterrupt:
            os.kill(p.pid, signal.SIGKILL)

        except ZeroDivisionError:
            if logger:
                logger.warn('%s %s', err, 'too fast to measure?')

        except (OSError, ValueError) as (e, err):
            if e == ENOENT:  # No such file or directory
                if logger:
                    logger.warn('%s %s', err, commandline)
                m.setMissing()
            else:
                if logger:
                    logger.error('%s %s', e, err)
                m.setError()

        finally:
            w.dump(m)
            wPipe.close()

            # Sample thread will be destroyed when the forked process _exits
            os._exit(0)
