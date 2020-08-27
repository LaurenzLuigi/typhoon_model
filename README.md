# typhoonify

## Description
Processing tools for generating wind field from best track data

## Requirements
numpy, pandas, scipy, netcdf4

## Installation
Cd to folder with setup.py and run pip install .

## Usage
#### Set-up
'''
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

from typhoonify.wind_maker import JMAto2D

incode = 1330                                   #typhoon code on best track data
jma_file = "bst_all.txt"                        #best track data file
typ = JMAto2D(jma_file, incode, freq="3H")      #initiate JMAto2D function
                                                #freq - offset aliases, in this case every 3hours

lat0, lon0 = 8, 120
lat1, lon1 = 13, 135
dellat, dellon = 0.02, 0.02
typ.make_grid((lat0, lon0), 
              (lat1, lon1), 
              (dellat, dellon))                 #create grid for calculation of 2D field

typhoon_lines = typ.Holland_Params()            #calculate parameters for calculationg of gradient 
                                                #winds based on Holland 1981
'''
####1D Profile
'''
rs = np.arange(0.1, 300, 0.1)                                                
typhoon_lines = typ.Holland_Profile(rs)

for index, typhoon in typhoon_lines.iterrows():
    if 25 > index > 18:
        plt.plot(rs, typhoon.Vgs, "b")
        if typhoon.R50 != 0:
            plt.scatter(typhoon.R50, 50.0*0.514444, color="g", marker="x")
            
        if typhoon.R30 != 0:
            plt.scatter(typhoon.R30, 30.0*0.514444, color="k", marker="x")
            
        plt.scatter(typhoon.RMW, typhoon.Vgmax, color="r", marker="x")
    
plt.show()
'''
####2D Field
'''
grid, vector, radial, pressure = typ.Holland_Field(FMA = True, WIA=True, theta_max=-115, dfm=0.5)
                                                
#save field to netcdf file
typ.nc_save() #fname = "xxxxx.nc"

#create animation of resulting wind_field
def animate(i):
    for index, typhoon in typhoon_lines.iterrows():
        if index == i:
            plt.clf()
            con = plt.contourf(grid[1], grid[0], radial[0][:, :, index], cmap="rainbow", vmin=0, vmax=40)
            m = plt.cm.ScalarMappable(cmap=plt.cm.rainbow)
            m.set_array(radial[0][:, :, index])
            m.set_clim(0, 40)
            plt.colorbar(m, boundaries=np.linspace(0, 40, 5))
            n = 15
            plt.quiver(grid[1][0:-1:n, 0:-1:n], grid[0][0:-1:n, 0:-1:n], 
                        vector[0][0:-1:n, 0:-1:n, index], vector[1][0:-1:n, 0:-1:n, index])
    return con

fig = plt.figure(figsize=(12, 4))
ax = plt.axes(xlim=(120, 125), ylim=(9, 13))
plt.axis("tight")

anim = animation.FuncAnimation(fig, animate, frames = len(typhoon_lines))
anim.save('Haiyan.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
'''
