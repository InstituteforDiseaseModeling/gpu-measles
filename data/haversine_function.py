#********************************************************************************
#
# Calculated haversine distance
#
# Python 3.6.0
#
#********************************************************************************

from math                         import sin, cos, asin, sqrt, pi

#*******************************************************************************

def hDist(lat1=0.0, long1=0.0, lat2=0.0, long2=0.0):

  # Earth radius in km
  r_earth = 6371

  # Convert degrees to radians
  lat1  = pi*lat1 /180.0
  lat2  = pi*lat2 /180.0
  long1 = pi*long1/180.0
  long2 = pi*long2/180.0

  delt = r_earth*2*asin(sqrt(sin(0.5*lat2-0.5*lat1)**2 +
                     cos(lat2)*cos(lat1)*sin(0.5*long2-0.5*long1)**2))

  return delt
 
#end-hDist

#*******************************************************************************
