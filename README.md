# Typhoonify

### Description
Processing tools for generating wind field from best track data

### Dependencies
numpy, pandas, scipy, netcdf4, matplotlib(optional)

### Installation
1. Install dependencies via pip
2. CD to folder with setup.py and run `pip install .`

### Usage
#### Set-up
```python
import numpy as np

from typhoonify.wind_maker import wind_maker

incode = 1330                                   #typhoon code on best track data
jma_file = "bst_all.txt"                        #best track data file
typ = wind_maker(jma_file, incode,
                 freq="3H", database="jma")     #initiate wind_maker function
                                                #freq - offset aliases, in this case every 3hours

lat0, lon0 = 8, 120                             #lower-left corner
lat1, lon1 = 13, 135                            #upper-right corner
dellat, dellon = 0.02, 0.02
typ.make_grid((lat0, lon0), 
              (lat1, lon1), 
              (dellat, dellon))                 #create grid for calculation of 2D field

typ.Holland_Params()                            #calculate parameters for calculationg of gradient 
                                                #winds based on Holland 1981
```

#### Calculate 1D Profile and compare to known datapoints in JMA

```python
import matplotlib.pyplot as plt

rs = np.arange(0, 300, 0.5)                   #creates an array of r distances
typ.Holland_Profile(rs)                         #calculates gradient wind at r distances away
typhoon = typ.typhoon

#plot estimated profile with jma data points
known_radii = typ.known_radii
for index, entry in typhoon.iterrows():
    if index == 0: #sample
        plt.plot(rs, entry.Vgs, "b")
        for radius in known_radii:
            key = "R" + str(radius) 
            if not np.isnan(entry[key]):
                plt.scatter(entry[key], radius*0.514444, color="g", marker="x")
            
        plt.scatter(entry.RMW, entry.Vgmax, color="r", marker="x")
        plt.grid()
        plt.xlabel("radius (km)")
        plt.ylabel("gradient wind speed (m/s)")
        plt.show()
```

#### Calculate wind and pressure fields and save to netcdf

```python
from matplotlib import animation

#Return 2D variables based on Holland Equation
typ.Holland_Field(FMA = True, WIA=True, theta_max=-115, dfm=0.5)
                                            #FMA - Apply Forward Motion Assymetry Correction
                                            #WIA - Apply Wind Inflow Angle Correction
lat = typ.grid.glat
long = typ.grid.long
wind_x = typ.wind                           #wind component along x-axis
wind_y = typ.wind_y                         #wind component along y-axis
wind_spd = typ.wind_spd                     #wind speed
wind_dir = typ.wind_dir                     #wind direction
wind_pres = typ.wind_pres                   #wind pressure field
           
#save field to netcdf file
typ.nc_save() #fname = "xxxxx.nc"

#create animation of resulting wind_field
def animate(i):
    for index, entry in typhoon.iterrows():
        if index == i:
            plt.clf()
            con = plt.contourf(long, lat, wind_spd[:, :, index], cmap="rainbow", vmin=0, vmax=40)
            m = plt.cm.ScalarMappable(cmap=plt.cm.rainbow)
            m.set_array(wind_spd[:, :, index])
            m.set_clim(0, 40)
            plt.colorbar(m, boundaries=np.linspace(0, 40, 5))
            n = 15
            plt.quiver(long[0:-1:n, 0:-1:n], lat[0:-1:n, 0:-1:n], 
                        wind_x[0:-1:n, 0:-1:n, index], wind_y[0:-1:n, 0:-1:n, index])
    return con

fig = plt.figure(figsize=(12, 4))
ax = plt.axes(xlim=(120, 125), ylim=(9, 13))
plt.axis("tight")

anim = animation.FuncAnimation(fig, animate, frames = len(typhoon))
anim.save('Haiyan.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
```
