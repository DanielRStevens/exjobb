
from domain import Record

import os
import time
import threading
import signal
from errno import ENOENT
from subprocess import Popen
from affinity import set_process_affinity_mask
import psutil  # We're running this on Windows rather than Linux, so gtop isn't an option
import pickle


def measure(arg, commandline, delay, maxtime,
            outFile=None, errFile=None, inFile=None, logger=None, affinitymask=None):

    r, w = os.pipe()
    forkedPid = os.fork()

    if forkedPid:  # read pickled measurements from the pipe
        os.close(w)
        rPipe = os.fdopen(r)
        r = pickle.Unpickler(rPipe)
        measurements = r.load()
        rPipe.close()
        os.waitpid(forkedPid, 0)
        return measurements

    else:
        class Sample(threading.Thread):
            def __init__(self, program):
                threading.Thread.__init__(self)
                self.setDaemon(1)
                self.timedout = False
                self.p = program
                self.process = psutil.Process(program)
                self.maxMem = 0
                self.children = None
                self.start()

            def run(self):
                try:
                    remaining = maxtime
                    while remaining > 0:
                        mem = self.process.memory_info().rss
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
                mem = self.process.memory_info().rss
                for each in self.process.children(recursive=True):
                    mem += each.memory_info().rss
                return mem

        try:
            def setAffinity():
                if affinitymask:
                    set_process_affinity_mask(os.getpid(), affinitymask)

            m = Record(arg)

            # only write pickles to the pipe
            os.close(r)
            wPipe = os.fdopen(w, 'w')
            w = pickle.Pickler(wPipe)

            # gtop cpu is since machine boot, so we need a before measurement
            #cpus0 = gtop.cpu().cpus
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
            #cpus1 = gtop.cpu().cpus

            # summarize measurements
            if t.timedout:
                m.setTimedout()
            elif rusage[1] == os.EX_OK:
                m.setOkay()
            else:
                m.setError()

            m.userSysTime = rusage[2][0] + rusage[2][1]
            m.maxMem = t.maxMem

            m.elapsed = elapsed

        except KeyboardInterrupt:
            os.kill(p.pid, signal.SIGKILL)

        except ZeroDivisionError as err:
            if logger:
                logger.warn('%s %s', err, 'too fast to measure?')

        except (OSError, ValueError) as e:
            if e == ENOENT:  # No such file or directory
                if logger:
                    logger.warn('%s %s', e[1], commandline)
                m.setMissing()
            else:
                if logger:
                    logger.error('%s %s', e[0], err)
                m.setError()

        finally:
            w.dump(m)
            wPipe.close()

            # Sample thread will be destroyed when the forked process _exits
            os._exit(0)
