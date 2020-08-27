# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:28:43 2020

@author: Luigi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class JMAto2D():    
    def __init__(self, jma_file, incode, inname=False, freq=False):
        self.typhoon_lines = self.read_typhoon(jma_file, incode, freq=freq)    
        
    def nc_save(self, fname = False):
        import netCDF4
        if fname is False:
            fname = f"{self.incode} {self.inname}.nc"
        lons, lats = self.grid.glat.shape
        
        long_out = self.grid.glon[:,0].tolist()
        lat_out = self.grid.glat[0,:].tolist()
        start = self.typhoon_lines.date.iloc[0]
        time_out = [(i - start).days*24 + (i - start).seconds/3600 for i in self.typhoon_lines.date]
        
        ncout = netCDF4.Dataset(fname, "w")
        
        ncout.createDimension("latitude", lats)
        ncout.createDimension("longitude", lons)
        ncout.createDimension("time", len(self.typhoon_lines))
        
        lats = ncout.createVariable("latitude","f",("latitude",))
        lons = ncout.createVariable("longitude","f",("longitude",))
        time = ncout.createVariable("time","f",("time",))
        
        U = ncout.createVariable("x_wind","f",("time","latitude","longitude"))
        V = ncout.createVariable("y_wind","f",("time","latitude","longitude"))
        pres = ncout.createVariable("air_pressure","f",("time","latitude","longitude"))
        
        lats[:] = lat_out
        lons[:] = long_out
        time[:] = time_out
        U[:,:,:] = np.swapaxes(self.wind_x, 0, -1)
        V[:,:,:] = np.swapaxes(self.wind_y, 0, -1)
        pres[:,:,:] = np.swapaxes(self.wind_pres, 0, -1) * 100
        
        lats.unit = "degree"
        lons.unit = "degree"
        time.unit = "hour since {start}"
        U.unit = "ms-1"
        V.unit = "ms-1"
        pres.unit = "Pa"
        
        ncout.close()
        
    def read_typhoon(self, jma_file, incode, inname = False, freq=False):
        txt_file = jma_file
        
        with open(txt_file) as f:
            lines = f.readlines()
        
        code_name = {}
        typhoon_dict = {}
        for line in lines:
            if int(line[0:5]) == 66666:
                code = line[6:10]
                name = line[30:50]
                code_name[code] = name.strip().capitalize()
                key = code + " " + name.strip().capitalize()
            else:
                typhoon_dict.setdefault(key, []).append(line[:-1])
        
        if inname is False:
            inname = code_name[str(incode)]
        
        print(f"Extracting Data for Typhoon {inname} with Code {incode}")
        inkey = f"{incode} " + inname
        typ = typhoon_dict[inkey]
        typ = self.decoder(typ)  
        self.inname = inname
        self.incode = incode
        
        if freq is not True:
            start = typ.date.iloc[0]
            end = typ.date.iloc[-1]
            new_range = pd.date_range(start, end, freq=freq)
            new_range = pd.DataFrame({"date": new_range})
            typ = new_range.merge(typ, on="date", how="left")
            typ.interpolate("linear", axis=0, inplace=True)
        return typ   
    
    def Holland_Params(self):
        "Calculate the wind-pressure for the set of radius given by the list"
        rho_air = 1.225
        Bs = []
        Vgmaxs = []
        RMWs = []
        self.typhoon_lines["delp"] = 1013 - self.typhoon_lines["Pc"]
        self.typhoon_lines.loc[self.typhoon_lines["delp"] < 0] = 0
        for index, typhoon in self.typhoon_lines.iterrows():
            RMW = 8
            lat = typhoon.lat
            Vgmax = typhoon.Vgmax
            
            if typhoon.R50 != 0 or typhoon.R30 != 0:  
                if typhoon.R50 != 0:
                    data = [(typhoon.R50, 50*0.5144444),
                            (typhoon.R30, 30*0.5144444)]
                else:
                    data = [(typhoon.R30, 30*0.5144444)]
                       
                fun = lambda x: self.error_calc(data, x[0], x[1], typhoon)
                opt = minimize(fun, (20, Vgmax), method="TNC", 
                               bounds=((0,100),(Vgmax-2.5*0.5144444, Vgmax+2.5*0.5144444)), tol=1e-8)
                RMW = opt.x[0]
                Vgmax = opt.x[1]
                B = Vgmax**2 * np.e * rho_air / typhoon.delp / 100
            else:
                RMW = np.exp(3.015 - 6.291*10e-5 * (typhoon.delp)**2 + 0.0337 * lat)
                B = 1.881 - 0.00557*RMW - 0.01295*typhoon.lat
                Vgmax = (typhoon.delp * 100 * B / np.exp(1)/1.15) ** 0.5                
           
            Vgmaxs.append(Vgmax)
            RMWs.append(RMW)
            Bs.append(B)
            
        self.typhoon_lines["Vgmax_old"] = self.typhoon_lines["Vgmax"]
        self.typhoon_lines["Vgmax"] = Vgmaxs
        self.typhoon_lines["RMW"] = RMWs
        self.typhoon_lines["B"] = Bs
        return self.typhoon_lines

    def Holland_Profile(self, rs):
        Vgs = []
        for index, typhoon in self.typhoon_lines.iterrows():
            Vg = []
            for r in rs:
                Vg.append(self.wind_function(r, typhoon))
            Vgs.append(Vg)
        
        self.typhoon_lines["Vgs"] = Vgs
        return self.typhoon_lines
            
    def Holland_Field(self, geo_cor="Harper", FMA=False, WIA=False, **kwargs):
        self.Holland_Params()
        wind_x = np.array([])
        wind_y = np.array([])
        wind_pres = np.array([])
        wind_spd = np.array([])
        wind_dire = np.array([])
        for index, typhoon in self.typhoon_lines.iterrows():      
            lons, lats = self.grid.glat.shape
            R = dist_calc((self.grid.glat, self.grid.glon),
                                  (typhoon.lat, typhoon.long))
            wind_speed = self.wind_function(R, typhoon)
            wind_direction = wind_dir((self.grid.glat, self.grid.glon), 
                                          (typhoon.lat, typhoon.long), 
                                          northern=True)

            if geo_cor == "Harper":
                wind_speed, _ = self.geo_Harper(wind_speed)
                
            if geo_cor == "Constant":
                constant = kwargs.get('constant')
                wind_speed = constant * (wind_speed)
                
            if FMA is True:
                if index == 0:
                    pass
                else:
                    dfm = kwargs.get('dfm')
                    theta_max = kwargs.get('theta_max')
                    wind_speed = self.FMA_Harper(wind_speed, wind_direction, typhoon, theta_max, dfm, index)
                    
            if WIA is True:
                try:
                    north = kwargs.get('north')
                    wind_direction = self.WIA_Sobey(wind_direction, typhoon.RMW, north)
                except:
                    wind_direction = self.WIA_Sobey(wind_direction, typhoon.RMW, R, True)
            
            try:
                wind_pres = np.dstack((wind_pres, self.pres_function(R, typhoon)))
                wind_spd = np.dstack((wind_spd, wind_speed))
                wind_dire = np.dstack((wind_dire, wind_direction))
                wind_x = np.dstack((wind_x, - wind_speed * np.sin(np.radians(wind_direction))))
                wind_y = np.dstack((wind_y, wind_speed * np.cos(np.radians(wind_direction))))
            except ValueError:
                wind_spd = np.expand_dims(wind_speed, axis=2)
                wind_dire = np.expand_dims(wind_direction, axis=2)
                wind_pres = np.expand_dims(self.pres_function(R, typhoon),axis=2)
                wind_x = np.expand_dims(- wind_speed * np.sin(np.radians(wind_direction)),axis=2)
                wind_y = np.expand_dims(wind_speed * np.cos(np.radians(wind_direction)),axis=2)     
                
            self.wind_spd = wind_spd
            self.wind_dire = wind_dire
            self.wind_pres = wind_pres
            self.wind_x = wind_x
            self.wind_y = wind_y
            
        return (self.grid.glat, self.grid.glon), (self.wind_x, self.wind_y), (self.wind_spd, self.wind_dire), self.wind_pres,
        
    def geo_Harper(self, ws):
        coef = np.copy(ws)
        coef[ws > 45] = 0.66
        coef[ws < 45] = (0.77 - 4.31e-3 * (ws[ws < 45] - 19.5))
        coef[ws < 19.5] = (0.81 - 2.96e-3 * (ws[ws < 19.5] - 6))
        coef[ws < 6] = 0.81
        return ws*coef, coef
    
    def FMA_Harper(self, ws, wd, typhoon, theta_max, dfm, index):
        new = self.typhoon_lines.iloc[index]
        old = self.typhoon_lines.iloc[index - 1]
        R = dist_calc((new.lat, new.long), (old.lat, old.long))
        delta_t = (new.date - old.date).seconds
        Vfm = R * 1000 / delta_t
        theta_max = (theta_max - 180) % 360
        corr = ws + (dfm * Vfm * np.cos(np.radians(theta_max - wd))) * ws / typhoon.Vgmax
        return corr    
    
    def WIA_Sobey(self, wd, RMW, R, north):
        coef = np.copy(wd)
        coef[R >= 1.2*RMW] = 25
        coef[R < 1.2 * RMW] = 10 + 75 * (R[R < 1.2 * RMW] / RMW - 1)
        coef[R < RMW] = 10 * R[R < RMW] / RMW
        if north is True:
            return wd + coef
        return wd - coef
    
    def error_calc(self, data, RMW, Vgmax, typhoon):
        error = 0
        for datum in data:
            error += (datum[1] - self.wind_optimize(RMW, datum[0], Vgmax, typhoon))**2
        return error
    
    def wind_optimize(self, RMW, r, Vgmax, typhoon):
        rho_air = 1.225
        f = 2 * 7.2921e-5 * np.sin(np.radians(typhoon.lat))
        if Vgmax != 0:
            B = Vgmax**2 * np.e * rho_air / typhoon.delp / 100
        else:
            B = 1.881 - 0.00557*RMW - 0.01295*typhoon.lat
            Vgmax = (typhoon.delp * 100 * B / np.exp(1)/1.15) ** 0.5
        
        p1 = (RMW / r) ** B
        p2 = B * typhoon.delp * np.exp(-p1) / rho_air * 100
        p3 = r**2 * f**2 / 4
        p4 = - f * r / 2
        
        return (p1*p2 + p3)**0.5 - p4    
    
    def wind_function(self, R, typhoon):
        rho_air = 1.225
        f = 2 * 7.2921e-5 * np.sin(np.radians(typhoon.lat))
        
        p1 = (typhoon.RMW / R) ** typhoon.B
        p2 = typhoon.B * typhoon.delp * np.exp(-p1) / rho_air * 100.0
        p3 = R**2 * f**2 / 4
        p4 = - f * R / 2
        
        return (p1*p2 + p3)**0.5 - p4
    
    def pres_function(self, R, typhoon):
        return typhoon.Pc + typhoon.delp * (np.exp(-typhoon.RMW / R))**typhoon.B
    
    def decoder(self, typ):
        line = []
        for entry in typ:
            #date, lat, long, Pc, Vgmax, R50, dir50, R30
            if int(entry[0:2]) > 30:
                date = "19" + entry[0:8]
            else:
                date = "20" + entry[0:8]
            lat = entry[15:18]
            long = entry[19:23]
            Pc = entry[24:28]
            Vgmax = entry[33:36]
            R50 = entry[42:46] if entry[42:46] != '    ' else 0
            dir50 = int(entry[41]) if entry[41] != ' ' else 0
            dir50 = dir50/8 * 360 if dir50 != 9 else 0
            R30 = entry[53:57] if entry[53:57] != '    ' else 0
            dir30 = int(entry[52]) if entry[52] != ' ' else 0
            dir30 = dir30/8 * 360 if dir30 != 9 else 0
            line.append([date, float(lat)/10, float(long)/10, float(Pc), float(Vgmax)*0.51444444, 
                          float(R50), int(dir50), float(R30), int(dir30)])
        
        typhoon = pd.DataFrame(line, columns = ["date", "lat", "long", "Pc", 
                                                "Vgmax", "R50", "dir50", "R30", "dir30"])
        typhoon["date"] = pd.to_datetime(typhoon.date, format="%Y%m%d%H")
        return typhoon
    
    def make_grid(self, ldown, uright, delta):
        lat0, lon0 = ldown
        lat1, lon1 = uright
        dellat, dellon = delta
    
        from collections import namedtuple
        grid_lat = np.arange(lat0, lat1, dellat)
        grid_lon = np.arange(lon0, lon1, dellon)
        mesh_lat, mesh_lon = np.meshgrid(grid_lat, grid_lon)
        grid_t = namedtuple('grid', 'glat glon')
        self.grid = grid_t(mesh_lat, mesh_lon)
        return self.grid
    
def wind_dir(COORDS1, coords0, northern=True):
    from numpy import cos, sin, radians, degrees, arctan2
    lat1, lon1 = COORDS1
    lat0, lon0 = coords0
    
    dlon = radians(lon1 - lon0)
    lat1 = radians(lat1)
    lat0 = radians(lat0)
    
    X = cos(lat1) * sin(dlon)
    Y = cos(lat0) * sin(lat1) - cos(lat1) * sin(lat0) * cos(dlon) 
    bearing = degrees(arctan2(Y, X))
    # if northern is True:
    #     wind_direction = (bearing + 90) % 360
    # else:
    #     wind_direction = (bearing - 90) % 360
    wind_direction = bearing
    return wind_direction

def dist_calc(COORDS1, coords0):
    """
    Simple conversion distance from meters to arc-degree
    """
    from numpy import cos, sin, radians, arctan2
    R = 6371 #radius of earth
    lat1, lon1 = COORDS1
    lat0, lon0 = coords0 
    dlat = radians(abs(lat1 - lat0))
    dlon = radians(abs(lon1 - lon0))
    a = sin(dlat/2)**2 + cos(radians(lat0))*cos(radians(lat1)) * sin(dlon/2)**2
    c = 2 * arctan2(a**0.5, (1-a)**0.5)
    return R * c

def embiggen(lat_rho, lon_rho):
    """
    enlarge the masking matrix by half a step in all direction to
    match with pcolormesh requirements on plottingd
    """
    dlat = lat_rho[0, 1] - lat_rho[0, 0]
    lat1 = lat_rho[:, 0] - dlat/2
    lat2 = lat_rho[:, -1] + dlat/2
    latmid = (lat_rho[:, 0:-1] + lat_rho[:, 1:])/2
    latplot = np.c_[lat1, latmid, lat2]

    dlat = latplot[1, 0] - latplot[0, 0]
    lat1 = (latplot[0, :] - dlat/2)
    lat2 = (latplot[-1, :] + dlat/2)
    latmid = (latplot[0:-1, :] + latplot[1:, :])/2
    latplot = np.r_[[lat1], latmid, [lat2]]

    dlon = lon_rho[1, 0] - lon_rho[0, 0]
    lon1 = (lon_rho[0, :] - dlon/2)
    lon2 = (lon_rho[-1, :] + dlon/2)
    lonmid = (lon_rho[0:-1, :] + lon_rho[1:, :])/2
    lonplot = np.r_[[lon1], lonmid, [lon2]]

    dlon = lonplot[0, 1] - lonplot[0, 0]
    lon1 = lonplot[:, 0] - dlon/2
    lon2 = lonplot[:, -1] + dlon/2
    lonmid = (lonplot[:, 0:-1] + lonplot[:, 1:])/2
    lonplot = np.c_[lon1, lonmid, lon2]

    return latplot, lonplot
