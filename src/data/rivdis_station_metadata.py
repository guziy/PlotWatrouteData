
__author__="huziy"
__date__ ="$Aug 10, 2011 1:01:21 PM$"


import application_properties
from shapely.geometry import Point

class RivDisStationMeta():
    def __init__(self, name = None, country = None, point = None):
        self.country = country
        self.name = name
        self.basin = None

        self.model_i = -1
        self.model_j = -1
        self.point = point

        pass


class Country():
    def __init__(self, name = ''):
        self.name = name
        self.stations = []


class RivDisStationManager():
    def __init__(self):
        self.countries = []
        pass

    def getCountries(self):
        return self.countries

    def parseStationMetaData(self, path = 'data/rivdis/lat_long_station_africa.csv'):
        for line in open(path):
            line = line.strip()
            if line == '': continue
            if not ';' in line:
                country = Country(name = line) #current country
                self.countries.append(country)
                continue

            if 'Station' in line or 'Longitute' in line: continue

            fields = line.split(';')
            lon = float(fields[1].strip())
            lat = float(fields[2].strip())
            p = Point(lon, lat)
            s = RivDisStationMeta(name = fields[0].strip(), country = country, point = p)
            country.stations.append(s)


    def saveStationsMetaData(self, path = 'station_info.txt'):
        f = open(path, 'w')
        
        f.write('Station;Longitude;Latitude;Country; Basin;model_i;model_j \n')
        for country in self.countries:
            for theStation in country.stations:
                if theStation.basin == None:
                    continue
                line = '{0};{1};{2};{3};{4};{5};{6}'.format(
                            theStation.name, theStation.point.x, theStation.point.y,
                            country.name,
                            theStation.basin, theStation.model_i, theStation.model_j
                            )
                print line
                f.write(line + '\n')
                
        
        f.close()
        pass



if __name__ == "__main__":
    application_properties.set_current_directory()
    rm = RivDisStationManager()
    rm.parseStationMetaData()
    for c in rm.getCountries():
        print c.name
    print "Hello World"
