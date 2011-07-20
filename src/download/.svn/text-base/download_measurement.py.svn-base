import os.path

__author__="huziy"
__date__ ="$26 mai 2010 09:14:33$"

from ftplib import FTP
from ftplib import *

import os

import application_properties
application_properties.set_current_directory() 

server_link = 'daac.ornl.gov'
server_folder_path = '/data/rivdis/STATIONS/TEXT/CANADA/'
destination = 'data/measurements'



def main():
    ftp = FTP(host = server_link)
    ftp.login()
    ftp.cwd(server_folder_path)

    for the_folder in ftp.nlst():
        try:
            the_destination = destination + os.sep + the_folder

            if not os.path.isdir(the_destination):
                os.makedirs(the_destination)

            for the_file in ftp.nlst(the_folder):
                print the_file
                local_file = open( destination + os.sep + the_file ,'wb')
                ftp.retrbinary('RETR %s' % the_file, local_file.write )
        except Exception, e:
            print e
            ftp.quit()

    ftp.quit()
    pass


def test_number_of_downloaded_folders():
    ftp = FTP(host = server_link)
    ftp.login()
    ftp.cwd(server_folder_path)

    n1 = len(ftp.nlst())
    print n1

    n2 = len( os.listdir( destination ) )

    print os.listdir( destination )
    print n2

    assert n1 == n2 - 1


if __name__ == "__main__":
    #main()
    test_number_of_downloaded_folders()
