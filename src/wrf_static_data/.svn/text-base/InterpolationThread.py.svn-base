# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="huziy"
__date__ ="$19 juil. 2010 16:29:05$"


import threading
class InterpolationThread(threading.Thread):
    def __init__(self, the_function , arguments):
        self.arguments = arguments
        self.function = the_function
        threading.Thread.__init__(self)
#        print 'Starting %s' % self.name
    def run(self):
        self.function(*self.arguments)

#        print 'finished %s' % self.name

if __name__ == "__main__":
    print "Hello World"
