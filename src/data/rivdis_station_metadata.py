
__author__="huziy"
__date__ ="$Aug 10, 2011 1:01:21 PM$"


import application_properties
from shapely.geometry import Point

class RivDisStationMeta():

    def __init__(self, name = None, country = None, point = None):
        self.country = country
        self.name = name
        self.basin = None

        self.drainage_area = -1

        self.model_i = -1
        self.model_j = -1
        self.point = point
        self.gridcell_polygon = None
        self.corresponding_cell = None

        pass


class Country():
    def __init__(self, name = ''):
        """
        represent a country
        properties:
        name = country name
        stations = list of stations in the country
        """
        self.name = name
        self.stations = []


class RivDisStationManager():
    def __init__(self):
        self.countries = []
        pass

    def getCountries(self):
        return self.countries

    def countStations(self):
        result = 0
        for c in self.countries:
            # @type c Country
            result += len(c.stations)
        return result


    def getStations(self):
        # @type c Country
        res = []
        for c in self.countries:
            res.extend(c.stations)        
        return res


    def compare_accumulation_areas(self, areas):
        for s in self.getStations():
            theCell = s.corresponding_cell
            if theCell is None: continue
            print s.drainage_area, theCell.drainage_area, theCell.coords(), areas[theCell.coords()], s.name

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
            lon = float(fields[2].strip())
            lat = float(fields[3].strip())
            p = Point(lon, lat)
            s = RivDisStationMeta(name = fields[0].strip(), country = country, point = p)
            s.drainage_area = float(fields[4].strip())
            country.stations.append(s)


    def saveStationsMetaData(self, path = 'station_info.txt'):
        f = open(path, 'w')
        
        f.write('Station;Longitude;Latitude;Country; Basin;model_i;model_j;stationCA;cellCA\n')
        for country in self.countries:
            for theStation in country.stations:
                if theStation.basin is None:
                    print 'no basin for the station {0}'.format(theStation.name)
                    print theStation.point.x, theStation.point.y
                    print theStation.point.wkt
                    continue
                line = '{0};{1};{2};{3};{4};{5};{6};{7};{8}'.format(
                            theStation.name, theStation.point.x, theStation.point.y,
                            country.name,
                            theStation.basin, theStation.model_i, theStation.model_j,
                            # @type theStation RivDisStationMeta
                            theStation.drainage_area,
                            theStation.corresponding_cell.drainage_area
                            )
                print line
                f.write(line + '\n')
                
        
        f.close()
        pass



if __name__ == "__main__":
    application_properties.set_current_directory()
    rm = RivDisStationManager()
    rm.parseStationMetaData()

    print len(rm.getStations())

    print rm.countStations()
    for c in rm.getCountries():
        print '-------' + c.name + '-------'
        for s in c.stations:
            print s.name, s.point.x, s.point.y, s.drainage_area
        
    print "Hello World"
