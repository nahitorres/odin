from queue import Queue, Empty
from threading import Thread


class SafeWriter:
    '''Class that allows to save a json launching a thread '''
    def __init__(self, *args):
        self.filewriter = open(*args)
        self.queue = Queue()
        self.finished = False
        Thread(name="SafeWriter", target=self.internal_writer).start()

    def write(self, data):
        self.queue.put(data)

    def internal_writer(self):
        while not self.finished:
            try:
                data = self.queue.get(True, 1)
            except Empty:
                continue
            self.filewriter.write(data)
            self.queue.task_done()

    def close(self):
        self.queue.join()
        self.finished = True
        self.filewriter.close()
