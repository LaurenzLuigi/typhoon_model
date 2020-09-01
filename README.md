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
    
incode = 1330                                           #typhoon code on best track data
jma_file = "bst_all.txt"                                #best track data file
tm = typhoon_model.Preprocessing()                      #initiate wind_maker function
typhoon = tm.read_database(jma_file, incode,            #read typhoon track from best track database (jma or jtwc)
                        outfreq="6H", database="jma")
typhoon = tm.calc_vgmax(method="constant", 
                        constant = 0.8)                #I recommend just using a constant value, rather than harper formulation
                                                        #Then just use constant again when you convert back from gradient to surface
typhoon = tm.resolve_vnan(method="Atk&Hol77")

hol = typhoon_model.HolSingVor(tm)
typhoon = hol.optimize(submethod = "Vic&Wad08")         #apply submethod if no data points present
```

#### Calculate 1D Profile and compare to known datapoints in JMA

```python
import matplotlib.pyplot as plt

rs = np.arange(0, 1000, 1)                   #creates an array of r distances
typhoon = hol.profiler(rs)                   #calculates gradient wind at r distances away

#plot estimated profile with jma data points
known_radii = tm.known_radii
for index, entry in typhoon.iterrows():
    if index == 20: #sample, change or remove to iterate over all indices
        plt.plot(rs, entry.Vgs, "b")
        for radius in known_radii:
            key = f"R{radius:.3f}" 
            if not np.isnan(entry[key]):
                plt.scatter(entry[key], radius, color="g", marker="x")
        plt.title(f"B={entry.B:0.2f}")
        plt.scatter(entry.RMW, entry.Vgmax, color="r", marker="x")
        plt.grid()
        plt.xlabel("radius (km)")
        plt.ylabel("gradient wind speed (m/s)")
        plt.ylim([0, 90])
        plt.xlim([0, 1000])
        plt.show()
        plt.close()
```

#### Calculate wind and pressure fields and save to netcdf

```python
lat0, lon0 = 8, 120                                 #lower-left corner
lat1, lon1 = 13, 135                                #upper-right corner
dellat, dellon = 0.02, 0.02
hol.make_grid((lat0, lon0), 
              (lat1, lon1), 
              (dellat, dellon))                     #create grid for calculation of 2D field

hol.field_maker(north=True)                         #north is True if northern hemisphere
hol.geostrophic_correction(method="Constant", constant=0.8)   
hol.forward_assymetry(method="Harper", dfm = 0.5, theta_max=-115)
hol.inflow_angle(method="Sobey")
hol.calc_vectors()

lat = hol.grid.glat
long = hol.grid.glon
wind_x = hol.wind_x                         #wind component along x-axis
wind_y = hol.wind_y                         #wind component along y-axis
wind_spd = hol.wind_spd                     #wind speed
wind_dir = hol.wind_dir                     #wind direction
wind_pres = hol.wind_pres                   #wind pressure field
           
#save field to netcdf file
hol.nc_save() #fname = "xxxxx.nc"

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
anim.save('Haiyan.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
```
