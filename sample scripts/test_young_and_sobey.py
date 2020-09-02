# -*- coding: utf-8 -*-

import numpy as np
from typhoon_model import typhoon_model

# incode = 312013                                       #typhoon code jwtc data
# jtwc_file = "bwp"                                     #folder containing best track data
# tm = typhoon_model.Preprocessing()                      #initiate wind_maker function
# typhoon = tm.read_database(jtwc_file, incode,            #read typhoon track from best track database (jma or jtwc)
#                         outfreq="6H", database="jtwc")

incode = 1330                                           #typhoon code on best track data
jma_file = "bst_all.txt"                                #best track data file
tm = typhoon_model.Processing()                      #initiate wind_maker function
typhoon = tm.read_database(jma_file, incode,            #read typhoon track from best track database (jma or jtwc)
                        outfreq="1H", database="jma")
typhoon = tm.conv_10min_to_1min(constant = 1.08)

typhoon = tm.calc_vgmax(method="constant", 
                        constant = 0.8)                 #I recommend just using a constant value, rather than harper formulation
                                                        #Then just use constant again when you convert back from gradient to surface
typhoon = tm.resolve_vnan(method="Atk&Hol77")

yas = typhoon_model.YoungSobey(tm)
typhoon = yas.optimize(submethod = "SGP02")         #apply submethod if no data points present

#1D Profile
import matplotlib.pyplot as plt

rs = np.arange(0, 1000.0, 1.0)                   #creates an array of r distances
typhoon = yas.profiler(rs)                   #calculates gradient wind at r distances away

#plot estimated profile with jma data points
known_radii = tm.known_radii
for index, entry in typhoon.iterrows():
    # if index == 20: #sample, change or remove to iterate over all indices
    plt.plot(rs, entry.Vgs, "b")
    for radius in known_radii:
        key = f"R{radius:.3f}" 
        if not np.isnan(entry[key]):
            plt.scatter(entry[key], radius, color="g", marker="x")
    plt.title(f"t={index}hr")
    plt.scatter(entry.RMW, entry.Vgmax, color="r", marker="x")
    plt.grid()
    plt.xlabel("radius (km)")
    plt.ylabel("gradient wind speed (m/s)")
    plt.ylim([0, 90])
    plt.xlim([0, 1000])
    fname = f"YoungSobey_Profile\\t={index}hr.png"
    plt.savefig(fname)
    plt.show()
    plt.close()

typhoon.to_csv(f"YoungSobey_Profile\\00 - typhoon.csv") 

#Return 2D variables based on Holland Equation
lat0, lon0 = 8, 120                                 #lower-left corner
lat1, lon1 = 13, 135                                #upper-right corner
dellat, dellon = 0.02, 0.02
grid = tm.make_grid((lat0, lon0), 
                    (lat1, lon1), 
                    (dellat, dellon))          #create grid for calculation of 2D field

yas.field_maker(grid, north=True)                         #north is True if northern hemisphere

tm.geostrophic_correction(method="Constant", constant=0.8)   
tm.forward_assymetry(method="Har01", dfm = 0.5, theta_max=-115)
tm.inflow_angle(method="Sob77")
tm.calc_vectors()

lat = tm.grid.glat
long = tm.grid.glon
wind_x = tm.wind_x                         #wind component along x-axis
wind_y = tm.wind_y                         #wind component along y-axis
wind_spd = tm.wind_spd                     #wind speed
wind_dir = tm.wind_dir                     #wind direction
wind_pres = tm.wind_pres                   #wind pressure field
           
#save field to netcdf file
tm.nc_save() #fname = "xxxxx.nc"

#create animation of resulting wind_field
from matplotlib import animation

def animate(i):
    plt.clf()
    con = plt.contourf(long, lat, wind_spd[:, :, i], cmap="rainbow", vmin=0, vmax=70)
    m = plt.cm.ScalarMappable(cmap=plt.cm.rainbow)
    m.set_array(wind_spd[:, :, i])
    m.set_clim(0, 70)
    plt.colorbar(m, boundaries=np.linspace(0, 70, 8))
    n = 15
    plt.quiver(long[0:-1:n, 0:-1:n], lat[0:-1:n, 0:-1:n], 
                wind_x[0:-1:n, 0:-1:n, i], wind_y[0:-1:n, 0:-1:n, i], 
                scale = 1000)
    return con

fig = plt.figure(figsize=(12, 4))
ax = plt.axes(xlim=(120, 125), ylim=(9, 13))
plt.axis("tight")

anim = animation.FuncAnimation(fig, animate, frames = len(typhoon))
anim.save('Haiyan_YoungSobey.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
