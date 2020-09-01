# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class wind_maker():
    def __init__(self, txt_file, incode, inname=False, freq="6H", database="jma"):
        '''
        Initiate the wind_maker class to calculate wind profile or wind field
        from different best-track data sets
        Parameters
        ----------
        txt_file : file_path
            Text file containing best track data downloadable from JMA
            or JTWC website
        incode : int
            Typhoon code based on best track data.
        inname : str, optional
            Typhoon name. The default is False.
        freq : offset alias, optional
            Time-step of output files. Missing data will be linearly 
            interpolated. The default is "6H".
        database : TYPE, optional
            Select between "jma" or "jtwc". The default is "jma".

        Returns
        -------
        None.

        '''
        self.inname = inname
        self.incode = incode
        self.txt_file = txt_file 
        self.freq = freq 
        self.database = database   
        self.HP_has_been_called = False
        self.rho_air = 1.15
        
        self.read_track()
        
    def read_track(self):
        '''
        Read tracks from file and extract typhoon based on self.incode
        Parameters
        ----------
        None.
        
        Returns
        -------
        typ : pd.DataFrame
            Typhoon data frame with inserted interpolated data

        '''
        if self.database.lower() == "jma":
            self.known_radii = [30, 50]
            self.minimum_record = 35
            typ = self.jma_decoder()
        if self.database.lower() == "jtwc":
            self.known_radii = [34, 50, 64, 100]
            self.minimum_record = 15
            typ = self.jtwc_decoder()
        if self.freq is not False:
            start = typ.date.iloc[0]
            end = typ.date.iloc[-1]
            new_range = pd.date_range(start, end, freq=self.freq)
            new_range = pd.DataFrame({"date": new_range})
            typ = new_range.merge(typ, on="date", how="left")
            typ.interpolate("linear", axis=0, inplace=True)
            
        for radius in self.known_radii:
            key = "R" + str(radius)
            typ.loc[(radius * 0.51444 >= typ.Vgmax * 0.98), key] = np.nan
      
        self.typhoon = typ
      
    def jma_decoder(self):
        '''
        Decode jma data_set to useable format

        Returns
        -------
        typhoon : pd.DataFrane
            Dataframe containing typhoon details.

        '''
        with open(self.txt_file) as f:
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
        
        if self.inname is False:
            self.inname = code_name[str(self.incode)]
        
        print(f"Extracting Data for Typhoon {self.inname} with Code {self.incode}")
        inkey = f"{self.incode} " + self.inname
        typ = typhoon_dict[inkey]
        
        line = []
        for entry in typ:
            #date, lat, long, Pc, Vgmax, R50, dir50, R30
            if int(entry[0:2]) > 50:
                date = "19" + entry[0:8]
            else:
                date = "20" + entry[0:8]
            lat = entry[15:18]
            long = entry[19:23]
            Pc = entry[24:28]
            Vmax = entry[33:36] if not entry[33:36] in ['   ', '000'] else np.nan
            R50 = entry[42:46] if not entry[42:46] in ['    ', '0000'] else np.nan
            R30 = entry[53:57] if not entry[53:57] in ['    ', '0000'] else np.nan
            line.append([date, float(lat)/10, float(long)/10, float(Pc), 
                         float(Vmax)*0.514444, float(R30) * 1.852, float(R50) * 1.852
                         ])
        typhoon = pd.DataFrame(line, columns = ["date", "lat", "long", "Pc", 
                                                "Vmax", "R30", "R50"])
        typhoon = typhoon[typhoon['Vmax'].notna()]
        typhoon["date"] = pd.to_datetime(typhoon.date, format="%Y%m%d%H")
        typhoon["Vgmax"] = [self.rev_geo_Harper(i) for i in typhoon.Vmax]
        return typhoon
    
    def jtwc_decoder(self):
        '''
        Decode jma data_set to useable format

        Returns
        -------
        typhoon : pd.DataFrane
            Dataframe containing typhoon details.

        '''
        cols = ["BASIN" , "CY" , "date" , "TECHNUM" , "TECH" , "TAU" , "lat" , "long" , "Vgmax" , "Pc" ,
                "TY" , "R" , "WINDCODE" , "R1" , "R2" , "R3" , "R4" , "RADP" , "RRP" , "MRD" , "GUSTS" , "EYE" ,
                "SUBREGION" , "MAXSEAS" , "INITIALS" , "DIR" , "SPEED" , "STORMNAME" , "DEPTH" , "SEAS" ,
                "SEASCODE" , "SEAS1" , "SEAS2" , "SEAS3" , "SEAS4"
                ]
        
        self.txt_file = self.txt_file + "\\bwp" + str(self.incode) + ".dat"
        typhoon_raw = pd.read_csv(self.txt_file, names=cols)
        
        radii = typhoon_raw[["R1", "R2", "R3", "R4"]]
        radii["Re"] = radii.min(axis=1)
        radii["R"] = typhoon_raw[["R"]]
        typhoon = typhoon_raw[["date"]]
        typhoon["R"] = typhoon_raw[["MRD"]]
        
        typhoon["date"] = pd.to_datetime(typhoon_raw.date, format="%Y%m%d%H")
        typhoon["lat"] = [float(i[:-1])/10 for i in typhoon_raw["lat"]]
        typhoon["long"] = [float(i[:-1])/10 for i in typhoon_raw["long"]]
        typhoon["Pc"] = typhoon_raw["Pc"].astype('float32')
        typhoon["Vgmax"] = typhoon_raw["Vgmax"].astype('float32') * 0.514444 / 0.93 / 0.9
 
        for radius in self.known_radii:  
            key = "R" + str(radius)
            typhoon[key] = radii.loc[(radii.R == radius), "Re"]

        return typhoon
    
    def Holland_Params(self):
        '''
        Optimize gradient wind formulation by Holland (1981) by adjusting 
        shape parameter B to minize root mean square error in comparison to
        known points based on best track data.
        
        In the absence of known points, shape parameter, and RMW was estimated based on 
        the relationship suggested by Vickery and Madhara (2003)
        
        Using the optimized gradient wind formulation, calculate the Radius of
        Maximum Winds (RMW)

        Returns
        -------
        None

        '''
        if self.HP_has_been_called is False:
            Bs = []
            Vgmaxs = []
            RMWs = []
            self.typhoon["delp"] = 1013 - self.typhoon["Pc"]
            self.typhoon.loc[self.typhoon["delp"] < 0] = 0.001
            for index, typhoon in self.typhoon.iterrows():
                RMW = 8
                lat = typhoon.lat
                Vgmax = typhoon.Vgmax
                data = []
                for speed in self.known_radii:
                    key = "R" + str(speed)
                    if not np.isnan(typhoon[key]):
                        speed = self.rev_geo_Harper(np.asarray(speed)) * 0.514444
                        # speed = speed * 0.514444  
                        data.append((typhoon[key], speed))
                    
                if data:
                    Vgmaxmin = self.rev_geo_Harper(typhoon.Vmax - 2.5*0.514444)
                    Vgmaxmax = self.rev_geo_Harper(typhoon.Vmax + 2.5*0.514444)
                    Vgmaxmin = typhoon.Vgmax - 2.5*0.514444
                    Vgmaxmax = typhoon.Vgmax + 2.5*0.514444
                    RMW_max = min([i[0] for i in data])
                    fun = lambda x: self.error_calc(data, x[0], x[1], typhoon)
                    opt = minimize(fun, (RMW, Vgmax), method="Powell", 
                                   bounds=((5, min(200,RMW_max)),(Vgmaxmin, Vgmaxmax)),
                                   tol=1e-8)
                    RMW = opt.x[0]
                    Vgmax = opt.x[1]
                    B = (Vgmax)**2 * np.e * self.rho_air / typhoon.delp / 100
                elif typhoon.Vmax > self.minimum_record * 0.514444:
                    RMW = np.exp(3.015 - 6.291*10e-5 * (typhoon.delp)**2 + 0.0337 * lat)
                    B = Vgmax**2 * np.e * self.rho_air / typhoon.delp / 100
                else:             
                    # RMW = np.exp(3.015 - 6.291*10e-5 * (typhoon.delp)**2 + 0.0337 * lat)
                    RMW = 0.676 * typhoon.Pc - 578
                    B = 2 - (typhoon.Pc - 900) / 160
                    # B = 1.881 - 0.00557*RMW - 0.01295*typhoon.lat
                    Vgmax = (typhoon.delp * 100 * B / np.e / self.rho_air) ** 0.5 
               
                Vgmaxs.append(Vgmax)
                RMWs.append(RMW)
                Bs.append(B)
                
            self.typhoon["Vgmax_old"] = self.typhoon["Vgmax"]
            self.typhoon["Vgmax"] = Vgmaxs
            self.typhoon["RMW"] = RMWs
            self.typhoon["B"] = Bs
            self.HP_has_been_called = True
        return None

    def Holland_Profile(self, rs):
        '''
        Calculates the gradient wind speed based on the formulation by Holland
        (1981), on the radiuses specified by the input list

        Parameters
        ----------
        rs : list
            List of radius where gradient wind should be calculated.

        Returns
        -------
        None

        '''
        self.Holland_Params()
        Vgs = []
        for index, typhoon in self.typhoon.iterrows():
            Vg = []
            for r in rs:
                Vg.append(self.wind_function(r, typhoon))
            Vgs.append(Vg)
        
        self.typhoon["Vgs"] = Vgs
        
        return None
            
    def Holland_Field(self, GC=False, FMA=False, WIA=False, north=True, **kwargs):
        '''
        Calculates 2D wind and pressure field based on the formulation by 
        Holland (1981)

        Parameters
        ----------
        geo_cor : str, optional
            DESCRIPTION. The default is "Harper".
        FMA : TYPE, optional
            DESCRIPTION. The default is False.
        WIA : TYPE, optional
            DESCRIPTION. The default is False.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None

        '''
        self.Holland_Params()
            
        wind_x = np.array([])
        wind_y = np.array([])
        wind_pres = np.array([])
        wind_spd = np.array([])
        wind_dir = np.array([])
                            
        for index, entry in self.typhoon.iterrows():      
            lons, lats = self.grid.glat.shape
            R = dist_calc((self.grid.glat, self.grid.glon),
                                  (entry.lat, entry.long))
            wind_speed = self.wind_function(R, entry)
            wind_direction = wind_dir_function((self.grid.glat, self.grid.glon), 
                                          (entry.lat, entry.long), 
                                          northern=north)

            if GC == "Harper":
                wind_speed, _ = self.geo_Harper(wind_speed)
                
            if GC == "Constant":
                constant = kwargs.get('constant')
                wind_speed = constant * (wind_speed)
                
            if FMA is True:
                if index == 0:
                    pass
                else:
                    dfm = kwargs.get('dfm')
                    theta_max = kwargs.get('theta_max')
                    wind_speed = self.FMA_Harper(wind_speed, wind_direction, entry, theta_max, dfm, index)
                    
            if WIA is True:
                    wind_direction = self.WIA_Sobey(wind_direction, entry.RMW, R, north)
            
            try:
                wind_pres = np.dstack((wind_pres, self.pres_function(R, entry)))
                wind_spd = np.dstack((wind_spd, wind_speed))
                wind_dir = np.dstack((wind_dir, wind_direction))
                wind_x = np.dstack((wind_x, - wind_speed * np.sin(np.radians(wind_direction))))
                wind_y = np.dstack((wind_y, wind_speed * np.cos(np.radians(wind_direction))))
            except ValueError:
                wind_spd = np.expand_dims(wind_speed, axis=2)
                wind_dir = np.expand_dims(wind_direction, axis=2)
                wind_pres = np.expand_dims(self.pres_function(R, entry),axis=2)
                wind_x = np.expand_dims(- wind_speed * np.sin(np.radians(wind_direction)),axis=2)
                wind_y = np.expand_dims(wind_speed * np.cos(np.radians(wind_direction)),axis=2)     
                
            self.wind_spd = wind_spd
            self.wind_dir = wind_dir
            self.wind_pres = wind_pres
            self.wind_x = wind_x
            self.wind_y = wind_y
            
        return None
        
    def geo_Harper(self, ws):
        '''
        Apply geostrophic correction to input wind field based on the values
        recommended by Harper et.al. (2001).

        Parameters
        ----------
        ws : 2D numpy array
            2D Wind Speed array.

        Returns
        -------
        ws*coef : 2D numpy array
            Corrected wind speed array.
        coef : 2D numpy array
            Correction coefficient.

        '''
        coef = np.copy(ws)
        coef[ws > 45] = 0.66
        coef[ws < 45] = (0.77 - 4.31e-3 * (ws[ws < 45] - 19.5))
        coef[ws < 19.5] = (0.81 - 2.96e-3 * (ws[ws < 19.5] - 6))
        coef[ws < 6] = 0.81
        ws = ws*coef
        return ws, coef
    
    def surface_to_gradient(self, ws):
        '''
        Apply geostrophic correction to input wind field based on the values
        recommended by Harper et.al. (2001).

        Parameters
        ----------
        ws : 2D numpy array
            2D Wind Speed array.

        Returns
        -------
        ws*coef : 2D numpy array
            Corrected wind speed array.
        coef : 2D numpy array
            Correction coefficient.

        '''
        error = 1
        wg = ws / 0.81
        while error > 0.0001: 
            wg_old = wg
            if wg > 45:
                wg = ws/0.66
            elif wg > 19.5:
                coef = 0.77 - 4.31e-3 * wg
                wg = ws/coef
            elif wg > 6:
                coef = 0.81 - 2.96e-3 * wg
                wg = ws/coef
            else:
                wg = ws/0.81
            error = wg - wg_old
        wg = ws / 0.81
        return wg
    
    def FMA_Harper(self, ws, wd, typhoon, theta_max, dfm, index):
        '''
        Corrects the wind field for Forward Motion Asymmetry based on the 
        equation suggested by Harper et.al. (2001)

        Parameters
        ----------
        ws : 2D numpy array
            2D Wind Speed array.
        wd : 2D numpy array
            2D Wind Direction array.
        typhoon : pd.DataFrame
            typhoon description at current time-step.
        theta_max : float
            Assumed line of maximum winds.
        dfm : float
            Correction factor.
        index : int
            index.

        Returns
        -------
        corr : 2D numpy array
            Corrected 2D Wind Speed array.

        '''
        new = self.typhoon.iloc[index]
        old = self.typhoon.iloc[index - 1]
        R = dist_calc((new.lat, new.long), (old.lat, old.long))
        delta_t = (new.date - old.date).seconds
        Vfm = R * 1000 / delta_t
        theta_max = (theta_max - 180) % 360
        ws = ws + (dfm * Vfm * np.cos(np.radians(theta_max - wd))) * ws / typhoon.Vgmax
        return ws    
    
    def WIA_Sobey(self, wd, RMW, R, north):
        '''
        Corrects wind direction by the inflow angle based on the formulation
        by Sobey et al. (1977)

        Parameters
        ----------
        wd : 2D numpy array
            2D Wind Direction array.
        RMW : float
            Radius of Maximum Winds.
        R : 2D numpy array
            2D array showing distance of grid to typhoon eye.
        north : Boolean
            True if northern hemispher, False if southern.

        Returns
        -------
        wd - coef : 2D numpy array
            Corrected Wind Direction array.

        '''
        coef = np.copy(wd)
        coef[R >= 1.2*RMW] = 25
        coef[R < 1.2 * RMW] = 10 + 75 * (R[R < 1.2 * RMW] / RMW - 1)
        coef[R < RMW] = 10 * R[R < RMW] / RMW
        if north is True:
            wd = wd + coef
            return wd
        wd = wd - coef
        return wd
    
    def nc_save(self, fname = False):
        '''
        Save the important 2D field into netcdf

        Parameters
        ----------
        fname : str, optional
            Filename. The default is False.

        Returns
        -------
        None.

        '''
        import netCDF4
        if fname is False:
            fname = f"{self.incode} {self.inname}.nc"
        lons, lats = self.grid.glat.shape
        
        long_out = self.grid.glon[:,0].tolist()
        lat_out = self.grid.glat[0,:].tolist()
        start = self.typhoon.date.iloc[0]
        time_out = [(i - start).days*24 + (i - start).seconds/3600 for i in self.typhoon.date]
        
        ncout = netCDF4.Dataset(fname, "w")
        
        ncout.createDimension("latitude", lats)
        ncout.createDimension("longitude", lons)
        ncout.createDimension("time", len(self.typhoon))
        
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
        time.unit = f"hour since {start}"
        U.unit = "ms-1"
        V.unit = "ms-1"
        pres.unit = "Pa"
        
        ncout.close()
    
    def error_calc(self, data, RMW, Vgmax, typhoon):
        error = 0
        for datum in data:
            error += (datum[1] - self.wind_optimize(RMW, datum[0], Vgmax, typhoon))**2
        error = (error / len(data)) ** 0.5
        return error
    
    def wind_optimize(self, RMW, r, Vgmax, typhoon):
        f = 2 * 7.2921e-5 * np.sin(np.radians(typhoon.lat))
        if Vgmax != 0:
            B = Vgmax**2 * np.e * self.rho_air / typhoon.delp / 100
        else:
            B = 1.881 - 0.00557*RMW - 0.01295*typhoon.lat
            Vgmax = (typhoon.delp * 100 * B / np.exp(1)/self.rho_air) ** 0.5
        
        p1 = (RMW / r) ** B
        p2 = B * typhoon.delp * np.exp(-p1) / self.rho_air * 100
        p3 = r**2 * f**2 / 4
        p4 = - abs(f) * r / 2
        
        return (p1*p2 + p3)**0.5 - p4    
    
    def wind_function(self, R, typhoon):
        f = 2 * 7.2921e-5 * np.sin(np.radians(typhoon.lat))
        
        p1 = (typhoon.RMW / R) ** typhoon.B
        p2 = typhoon.B * typhoon.delp * np.exp(-p1) / self.rho_air * 100.0
        p3 = R**2 * f**2 / 4
        p4 = - f * R / 2
        
        return (p1*p2 + p3)**0.5 - p4
    
    def pres_function(self, R, typhoon):
        return typhoon.Pc + typhoon.delp * (np.exp(-typhoon.RMW / R))**typhoon.B

    
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
    
def wind_dir_function(COORDS1, coords0, northern=True):
    from numpy import cos, sin, radians, degrees, arctan2
    lat1, lon1 = COORDS1
    lat0, lon0 = coords0
    
    dlon = radians(lon1 - lon0)
    lat1 = radians(lat1)
    lat0 = radians(lat0)
    
    X = cos(lat1) * sin(dlon)
    Y = cos(lat0) * sin(lat1) - cos(lat1) * sin(lat0) * cos(dlon) 
    bearing = degrees(arctan2(Y, X))
    if northern is True:
        wind_direction = bearing
    else:
        wind_direction = bearing - 180
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
