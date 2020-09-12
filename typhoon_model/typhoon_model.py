# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Critical_Typhoon():
    def __init__(self, proj_lat, proj_lon, radius=200):
        self.proj_lat = proj_lat
        self.proj_lon = proj_lon
        self.radius = radius
        self.amb_pres = 1010
        
    def read_database(self, txt_file, year_start, year_end, outfreq="6H", database="jma"):
        '''
        Read tracks from file and extract typhoon based on self.incode
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
        typ : pd.DataFrame
            Typhoon data frame with inserted interpolated data

        '''
        self.txt_file = txt_file
        self.outfreq = outfreq
        self.year_start = year_start
        self.year_end = year_end
        
        if database.lower() == "jma":
            self.known_radii = np.asarray([30, 50]) * 0.514444
            self.minimum_record = 35
            self.jma_decoder()
        if database.lower() == "jtwc":
            self.known_radii = np.asarray([34, 50, 64, 100]) * 0.514444
            self.minimum_record = 15
            self.jtwc_decoder()
            
        for key, typ in self.typhoon_dict.items():
            start = typ.date.iloc[0]
            end = typ.date.iloc[-1]
            new_range = pd.date_range(start, end, freq=outfreq)
            new_range = pd.DataFrame({"date": new_range})
            typhoon = new_range.merge(typ, on="date", how="left")
            typhoon.interpolate("linear", axis=0, limit_area="inside", inplace=True)
                
            self.typhoon_dict[key] = typhoon
            
        keys = []
        for key, typ in self.typhoon_dict.items():
            dist = []
            coords0 = (self.proj_lat, self.proj_lon)
            for index, entry in typ.iterrows():
                coords1 = (entry.lat, entry.lon)
                dist.append(dist_calc(coords0, coords1))
                
            typ["distance"] = dist
            typ["year"] = pd.DatetimeIndex(typ.date).year
            typ["name"] = key[5:]
            typ["code"] = key[0:4]
            if typ.distance.min() > self.radius:
                keys.append(key)
        
        for key in keys:        
            del self.typhoon_dict[key]
      
        self.conv_10min_to_1min()
        self.calc_vgmax(constant=0.8)
        self.resolve_vnan()
              
        self.calc_vsite()
        
        return self.typhoon_dict
      
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
        
        code_name = []
        self.typhoon_dict = {}
        for line in lines:
            if int(line[0:5]) == 66666:
                code = line[6:10]
                name = line[30:50]
                code_name.append((code, name.strip().capitalize()))
                key = code + " " + name.strip().capitalize()
            else:
                self.typhoon_dict.setdefault(key, []).append(line[:-1])
                
        for code, name in code_name:
            key = code + " " + name
            year = code[0:2]
            if int(year) > 50:
                year = "19" + year
            else:
                year = "20" + year
                
            if not (self.year_start <= int(year) <= self.year_end):
                del self.typhoon_dict[key]     
                
        for key, typ in self.typhoon_dict.items():    
            line = []
            for entry in typ:
                #date, lat, lon, Pc, Vgmax, R50, dir50, R30
                if int(entry[0:2]) > 50:
                    date = "19" + entry[0:8]
                else:
                    date = "20" + entry[0:8]
                lat = entry[15:18]
                lon = entry[19:23]
                Pc = entry[24:28]
                Vmax = entry[33:36] if not entry[33:36] in ['   ', '000'] else np.nan
                line.append([date, float(lat)/10, float(lon)/10, float(Pc), 
                             float(Vmax)*0.514444
                             ])
            column_names = ["date", "lat", "lon", "Pc", "Vmax"]
            
            typhoon = pd.DataFrame(line, columns = column_names)
            typhoon["date"] = pd.to_datetime(typhoon.date, format="%Y%m%d%H")
            delP = self.amb_pres - typhoon.Pc
            typhoon["delP"] = [i if i > 1 else 1 for i in delP]   
            
            self.typhoon_dict[key] = typhoon
            
    def conv_10min_to_1min(self, constant=1.08):
        for key, typ in self.typhoon_dict.items():  
            typ["Vmax"] = typ["Vmax"] * constant
            self.typhoon_dict[key] = typ
            
        return self.typhoon_dict.items()
    
    def calc_vgmax(self, method="constant", **kwargs):
        for key, typ in self.typhoon_dict.items(): 
            if method == "constant":
                constant = kwargs.get("constant")
                typ["Vgmax"] = typ["Vmax"] / constant
                self.typhoon_dict[key] = typ

        return self.typhoon_dict.items()
          
    def resolve_vnan(self, method="Atk&Hol77", **kwargs):
        if method == "remove":
            for key, typ in self.typhoon_dict.items(): 
                typ = typ.dropna(subset=["Vgmax"])
                self.typhoon_dict[key] = typ
            return self.typhoon_dict.items()
        
        if method == "Atk&Hol77":
            for key, typ in self.typhoon_dict.items(): 
                typ.loc[np.isnan(typ['Vgmax']), 'Vgmax'] = 6.7 * (typ.loc[np.isnan(typ['Vgmax']), 'delP']) ** 0.644
                self.typhoon_dict[key] = typ
            return self.typhoon_dict.items()   
           
    def calc_vsite(self):
        for key, typ in self.typhoon_dict.items(): 
            vsite = []
            for index, entry in typ.iterrows():    
                RMW = 0.4785 * entry.Pc - 413
                if entry.distance < RMW:
                    vsite.append(entry.Vgmax * (entry.distance/RMW)**7.0 * np.exp(7.0*(1.0-entry.distance/RMW)))
                else:
                    vsite.append(entry.Vgmax * np.exp((0.0025*RMW + 0.05)*(1.0-entry.distance/RMW)))  
            typ["v_site"] = [0.8*v for v in vsite]
        
class Processing():
    def __init__(self):
        '''
        Initiate the wind_maker class to calculate wind profile or wind field
        from different best-track data sets
        Parameters
        ----------
        None.
        Returns
        -------
        None.

        '''   
        self.rho_air = 1.15
        self.amb_pres = 1010
                
    def read_database(self, txt_file, incode, inname=False, outfreq="6H", database="jma"):
        '''
        Read tracks from file and extract typhoon based on self.incode
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
        typ : pd.DataFrame
            Typhoon data frame with inserted interpolated data

        '''
        self.txt_file = txt_file
        self.incode = incode
        self.inname = inname
        self.outfreq = outfreq
        
        if database.lower() == "jma":
            self.known_radii = np.asarray([30, 50]) * 0.514444
            self.minimum_record = 35
            self.typhoon = self.jma_decoder()
        if database.lower() == "jtwc":
            self.known_radii = np.asarray([34, 50, 64, 100]) * 0.514444
            self.minimum_record = 15
            self.typhoon = self.jtwc_decoder()
            
        start = self.typhoon.date.iloc[0]
        end = self.typhoon.date.iloc[-1]
        new_range = pd.date_range(start, end, freq=outfreq)
        new_range = pd.DataFrame({"date": new_range})
        self.typhoon = new_range.merge(self.typhoon, on="date", how="left")
        self.typhoon.interpolate("linear", axis=0, limit_area="inside", inplace=True)
            
        for radius in self.known_radii:
            key = f"R{radius:.3f}"
            self.typhoon.loc[(radius >= self.typhoon.Vmax * 0.98), key] = np.nan
      
        return self.typhoon
      
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
        column_names = ["date", "lat", "long", "Pc", "Vmax"]
        for radius in self.known_radii:
            column_names.append(f"R{radius:.3f}")
        
        typhoon = pd.DataFrame(line, columns = column_names)
        typhoon["date"] = pd.to_datetime(typhoon.date, format="%Y%m%d%H")
        delP = self.amb_pres - typhoon.Pc
        typhoon["delP"] = [i if i > 1 else 1 for i in delP]
        return typhoon
    
    def jtwc_decoder(self):
        '''
        Decode jma data_set to useable format

        Returns
        -------
        typhoon : pd.DataFrane
            Dataframe containing typhoon details.

        '''
        cols = ["BASIN" , "CY" , "date" , "TECHNUM" , "TECH" , "TAU" , "lat" , "long" , "Vmax" , "Pc" ,
                "TY" , "R" , "WINDCODE" , "R1" , "R2" , "R3" , "R4" , "RADP" , "RRP" , "MRD" , "GUSTS" , "EYE" ,
                "SUBREGION" , "MAXSEAS" , "INITIALS" , "DIR" , "SPEED" , "STORMNAME" , "DEPTH" , "SEAS" ,
                "SEASCODE" , "SEAS1" , "SEAS2" , "SEAS3" , "SEAS4"
                ]
        
        self.txt_file = self.txt_file + "\\bwp" + str(self.incode) + ".dat"
        typhoon_raw = pd.read_csv(self.txt_file, names=cols)
        
        radii = typhoon_raw[["R1", "R2", "R3", "R4"]]
        radii["Re"] = radii.min(axis=1) * 1.852
        radii["R"] = typhoon_raw[["R"]]
        typhoon = typhoon_raw[["date"]]
        typhoon["RMW"] = typhoon_raw[["MRD"]] * 1.852
        
        typhoon["date"] = pd.to_datetime(typhoon_raw.date, format="%Y%m%d%H")
        typhoon["lat"] = [float(i[:-1])/10 for i in typhoon_raw["lat"]]
        typhoon["long"] = [float(i[:-1])/10 for i in typhoon_raw["long"]]
        typhoon["Pc"] = typhoon_raw["Pc"].astype('float32')
        typhoon["Vmax"] = typhoon_raw["Vmax"].astype('float32') * 0.514444
 
        for radius in self.known_radii:  
            key = f"R{radius:.3f}" 
            typhoon[key] = radii.loc[(radii.R == int(radius / 0.514444)), "Re"]
            
        delP = self.amb_pres - typhoon.Pc
        typhoon["delP"] = [i if i > 1 else 1 for i in delP]
        typhoon = typhoon.groupby(["date"], as_index=False).mean()
        return typhoon
    
    def conv_10min_to_1min(self, constant=1.08):
        self.typhoon["Vmax"] = self.typhoon["Vmax"] * constant
        adj_radii = [i * constant for i in self.known_radii]
        
        rename = {}
        for rad,adj in zip(self.known_radii, adj_radii):
            orig = f"R{rad:.3f}"
            replace = f"R{adj:.3f}"
            rename[orig] = replace
        self.typhoon.rename(columns=rename, inplace=True)
        self.known_radii = adj_radii
            
        return self.typhoon
    
    def calc_vgmax(self, method="constant", **kwargs):
        if method == "constant":
            constant = kwargs.get("constant")
            self.typhoon["Vgmax"] = self.typhoon["Vmax"] / constant
            adj_radii = [i / constant for i in self.known_radii]
            
        if method == "Har01":
            Vgmax = []
            for index, typhoon in self.typhoon.iterrows():
                Vgmax.append(self.rev_harper(typhoon.Vmax))
            self.typhoon["Vgmax"] = Vgmax
            
            adj_radii = [self.rev_harper(i) for i in self.known_radii]
        
        rename = {}
        for rad,adj in zip(self.known_radii, adj_radii):
            orig = f"R{rad:.3f}"
            replace = f"R{adj:.3f}"
            rename[orig] = replace
        self.typhoon.rename(columns=rename, inplace=True)
        self.known_radii = adj_radii
        
        return self.typhoon
        
    def rev_harper(self, vmax):
        Vgmax = vmax / 0.66
        error = 1
        while error > 1e-8:
            coef = 0.66
            coef = 0.77 - 4.31e-3 * (Vgmax - 19.5) if Vgmax < 45 else coef
            coef = 0.81 - 2.96e-3 * (Vgmax - 6) if Vgmax < 19.5 else coef
            coef = 0.81 if Vgmax < 6 else coef
            
            error = abs(Vgmax - vmax / coef)
            Vgmax = vmax / coef
            
        return Vgmax
          
    def resolve_vnan(self, method="remove", **kwargs):
        if method == "remove":
            self.typhoon = self.typhoon.dropna(subset=["Vgmax"])
            return self.typhoon
        if method == "Atk&Hol77":
            self.typhoon.loc[np.isnan(self.typhoon['Vgmax']), 'Vgmax'] = 6.7 * (self.typhoon.loc[np.isnan(self.typhoon['Vgmax']), 'delP']) ** 0.644
            return self.typhoon    
            
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
        
    def geostrophic_correction(self, method="Constant", **kwargs):
        if method == "Har01":
            ws = self.wind_spd
            coef = np.copy(ws)
            coef[ws > 45] = 0.66
            coef[ws < 45] = (0.77 - 4.31e-3 * (ws[ws < 45] - 19.5))
            coef[ws < 19.5] = (0.81 - 2.96e-3 * (ws[ws < 19.5] - 6))
            coef[ws < 6] = 0.81
            
            self.wind_spd = coef * self.wind_spd
            
        if method == "Constant":
            constant = kwargs.get('constant')
            self.wind_spd = self.wind_spd * constant
            
        return None
    
    def forward_assymetry(self, method="Har01", **kwargs):
        if method == "Har01":
            for index, entry in self.typhoon.iterrows():  
                if index == 0:
                    pass
                else:
                    dfm = kwargs.get('dfm')
                    theta_max = kwargs.get('theta_max')
                    
                    ws = self.wind_spd[:, :, index]
                    new = self.typhoon.iloc[index]
                    old = self.typhoon.iloc[index - 1]
                    R = dist_calc((new.lat, new.long), (old.lat, old.long))
                    delta_t = (new.date - old.date).seconds
                    Vfm = R * 1000 / delta_t
                    theta_max = (theta_max - 180) % 360
                    ws = ws + (dfm * Vfm * np.cos(np.radians(theta_max - self.wind_dir[:, :, index]))) * ws / entry.Vgmax
                    self.wind_spd[:, :, index] = ws
        return None
    
    def inflow_angle(self, method="Sob77"):
        if method == "Sob77":
            for index, entry in self.typhoon.iterrows(): 
                coef = np.copy(self.wind_dir[:, :, index])
                RMW = entry.RMW             
                R = dist_calc((self.grid.glat, self.grid.glon),
                                      (entry.lat, entry.long))
                
                coef[R >= 1.2*RMW] = 25
                coef[R < 1.2 * RMW] = 10 + 75 * (R[R < 1.2 * RMW] / RMW - 1)
                coef[R < RMW] = 10 * R[R < RMW] / RMW
                
                if self.north is True:
                    self.wind_dir[:, :, index] = self.wind_dir[:, :, index] + coef
                else:
                    self.wind_dir[:, :, index] = self.wind_dir[:, :, index] - coef
        return None    
    
    def calc_vectors(self):
        self.wind_x = - self.wind_spd * np.sin(np.radians(self.wind_dir))
        self.wind_y = self.wind_spd * np.cos(np.radians(self.wind_dir))
                   
    def nc_save(self, fname = False):
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

class HolSingVor():
    def __init__(self, tm):
        self.tm = tm
        self.typhoon = tm.typhoon
        self.rho_air = self.tm.rho_air
        
    def optimize(self, submethod="Vic&Wad08"):
        '''
        Optimize gradient wind formulation by Holland (1981) by adjusting 
        shape RMW to minize root mean square error in comparison to
        known points based on best track data.
        
        In the absence of known points, shape parameter, and RMW was estimated based on 
        the relationship suggested by Vickery and Madhara (2003)
        
        Using the optimized gradient wind formulation, calculate the Radius of
        Maximum Winds (RMW)

        Returns
        -------
        None

        '''
        Bs = []
        RMWs = []
        if "RMW" in self.typhoon.columns:
            for index, typhoon in self.typhoon.iterrows():
                lat = typhoon.lat
                Vgmax = typhoon.Vgmax
                if np.isnan(typhoon.RMW):
                    if submethod == "Vic&Wad08":
                        RMW = np.exp(3.015 - 6.291*10e-5 * (typhoon.delP)**2 + 0.0337 * lat)
                        B = Vgmax**2 * np.e * self.rho_air / typhoon.delP / 100
                    if submethod == "Will07":
                        RMW = 46.6 * np.exp(-0.0155 * typhoon.Vgmax) + 0.0169 * lat
                        B = Vgmax**2 * np.e * self.rho_air / typhoon.delP / 100
                    if submethod == "SGP02":
                        RMW = 0.4785 * typhoon.Pc - 413
                        B = Vgmax**2 * np.e * self.rho_air / typhoon.delP / 100
                else:
                    RMW = typhoon.RMW
                    B = Vgmax**2 * np.e * self.rho_air / typhoon.delP / 100
                    
                RMWs.append(RMW)
                Bs.append(B)
        else:
            errors = []
            for index, typhoon in self.typhoon.iterrows():
                RMW = 8
                lat = typhoon.lat
                Vgmax = typhoon.Vgmax
                data = []
                for radius in self.tm.known_radii:
                    key = f"R{radius:.3f}"
                    if not np.isnan(typhoon[key]):
                        data.append((typhoon[key], radius))
                    
                if data:
                    RMW_max = min([i[0] for i in data])
                    fun = lambda x: self.error_calc(data, x, Vgmax, typhoon)
                    opt = minimize(fun, RMW, method="Powell", 
                                   bounds=((5, min(200,RMW_max)),),
                                   tol=1e-8)
                    RMW = opt.x[0]
                    errors.append(self.error_calc(data, RMW, Vgmax, typhoon))
                    B = (Vgmax)**2 * np.e * self.rho_air / typhoon.delP / 100
                else:
                    if submethod == "Vic&Wad08":
                        RMW = np.exp(3.015 - 6.291*10e-5 * (typhoon.delP)**2 + 0.0337 * lat)
                        B = Vgmax**2 * np.e * self.rho_air / typhoon.delP / 100
                    if submethod == "Will07":
                        RMW = 46.6 * np.exp(-0.0155 * typhoon.Vgmax) + 0.0169 * lat
                        B = Vgmax**2 * np.e * self.rho_air / typhoon.delP / 100
                    if submethod == "SGP02":
                        RMW = 0.4785 * typhoon.Pc - 413
                        B = Vgmax**2 * np.e * self.rho_air / typhoon.delP / 100
               
                RMWs.append(RMW)
                Bs.append(B)
             
            # key =  f"R{self.tm.known_radii[0]:.3f}"   
            # self.typhoon.loc[np.invert(np.isnan(self.typhoon[key])),"errors"] = errors
        self.typhoon["RMW"] = RMWs
        self.typhoon["B"] = Bs
        
        return self.typhoon

    def profiler(self, rs):
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
        Vgs = []
        for index, typhoon in self.typhoon.iterrows():
            Vg = []
            for r in rs:
                Vg.append(self.wind_function(r, typhoon))
            Vgs.append(Vg)
        
        self.typhoon["Vgs"] = Vgs
        
        return self.typhoon
    
    def field_maker(self, grid, north=True):
        self.grid = grid
        self.north = north
        self.tm.north = self.north
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
                                          northern=self.north)
            
            try:
                wind_pres = np.dstack((wind_pres, self.pres_function(R, entry)))
                wind_spd = np.dstack((wind_spd, wind_speed))
                wind_dir = np.dstack((wind_dir, wind_direction))
            except ValueError:
                wind_spd = np.expand_dims(wind_speed, axis=2)
                wind_dir = np.expand_dims(wind_direction, axis=2)
                wind_pres = np.expand_dims(self.pres_function(R, entry),axis=2)    
                
        self.wind_spd = wind_spd
        self.wind_dir = wind_dir
        self.wind_pres = wind_pres
        self.tm.wind_spd = self.wind_spd 
        self.tm.wind_dir = self.wind_dir 
        self.tm.wind_pres = self.wind_pres   
        
        return None
        
    def error_calc(self, data, RMW, Vgmax, typhoon):
        error = 0
        for datum in data:
            error += (datum[1] - self.wind_optimize(RMW, datum[0], Vgmax, typhoon))**2
        error = (error / len(data)) ** 0.5
        return error
    
    def wind_optimize(self, RMW, r, Vgmax, typhoon):
        f = 2 * 7.2921e-5 * np.sin(np.radians(typhoon.lat))
        if Vgmax != 0:
            B = Vgmax**2 * np.e * self.rho_air / typhoon.delP / 100
        else:
            B = 1.881 - 0.00557*RMW - 0.01295*typhoon.lat
            Vgmax = (typhoon.delP * 100 * B / np.exp(1)/self.rho_air) ** 0.5
        
        p1 = (RMW / r) ** B
        p2 = B * typhoon.delP * np.exp(-p1) / self.rho_air * 100
        p3 = r**2 * f**2 / 4
        p4 = - abs(f) * r / 2
        wind = (p1*p2 + p3)**0.5 - p4
        wind = np.nan_to_num(wind)
        
        return wind
    
    def wind_function(self, R, typhoon):
        f = 2 * 7.2921e-5 * np.sin(np.radians(typhoon.lat))
        
        p1 = (typhoon.RMW / R) ** typhoon.B
        p2 = typhoon.B * typhoon.delP * np.exp(-p1) / self.rho_air * 100.0
        p3 = R**2 * f**2 / 4
        p4 = - abs(f) * R / 2
        wind = (p1*p2 + p3)**0.5 - p4
        wind = np.nan_to_num(wind)
        
        return wind
    
    def pres_function(self, R, typhoon):
        return typhoon.Pc + typhoon.delP * (np.exp(-typhoon.RMW / R))**typhoon.B
    
class YoungSobey():
    def __init__(self, tm):
        self.tm = tm
        self.typhoon = tm.typhoon
        self.rho_air = self.tm.rho_air
        
    def optimize(self, submethod="Vic&Wad08"):
        '''
        Optimize gradient wind formulation by Holland (1981) by adjusting 
        shape RMW to minize root mean square error in comparison to
        known points based on best track data.
        
        In the absence of known points, shape parameter, and RMW was estimated based on 
        the relationship suggested by Vickery and Madhara (2003)
        
        Using the optimized gradient wind formulation, calculate the Radius of
        Maximum Winds (RMW)

        Returns
        -------
        None

        '''
        RMWs = []
        if "RMW" in self.typhoon.columns:
            for index, typhoon in self.typhoon.iterrows():
                lat = typhoon.lat
                if np.isnan(typhoon.RMW):
                    if submethod == "Vic&Wad08":
                        RMW = np.exp(3.015 - 6.291*10e-5 * (typhoon.delP)**2 + 0.0337 * lat)
                    if submethod == "Will07":
                        RMW = 46.6 * np.exp(-0.0155 * typhoon.Vgmax) + 0.0169 * lat
                    if submethod == "SGP02":
                        RMW = 0.4785 * typhoon.Pc - 413
                else:
                    RMW = typhoon.RMW
                    
                RMWs.append(RMW)
        else:
            errors = []
            for index, typhoon in self.typhoon.iterrows():
                RMW = 8
                lat = typhoon.lat
                Vgmax = typhoon.Vgmax
                data = []
                for radius in self.tm.known_radii:
                    key = f"R{radius:.3f}"
                    if not np.isnan(typhoon[key]):
                        data.append((typhoon[key], radius))
                    
                if data:
                    RMW_max = min([i[0] for i in data])
                    fun = lambda x: self.error_calc(data, x, Vgmax, typhoon)
                    opt = minimize(fun, RMW, method="Powell", 
                                   bounds=((5, min(200,RMW_max)),),
                                   tol=1e-8)
                    RMW = opt.x[0]
                    errors.append(self.error_calc(data, RMW, Vgmax, typhoon))
                else:
                    if submethod == "Vic&Wad08":
                        RMW = np.exp(3.015 - 6.291*10e-5 * (typhoon.delP)**2 + 0.0337 * lat)
                    if submethod == "Will07":
                        RMW = 46.6 * np.exp(-0.0155 * typhoon.Vgmax) + 0.0169 * lat
                    if submethod == "SGP02":
                        RMW = 0.4785 * typhoon.Pc - 413
               
                RMWs.append(RMW)
             
            # key =  f"R{self.tm.known_radii[0]:.3f}"   
            # self.typhoon.loc[np.invert(np.isnan(self.typhoon[key])),"errors"] = errors
        self.typhoon["RMW"] = RMWs
        
        return self.typhoon

    def profiler(self, rs):
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
        Vgs = []
        for index, typhoon in self.typhoon.iterrows():
            Vg = []
            for r in rs:
                Vg.append(self.wind_function(r, typhoon))
            Vgs.append(Vg)
        
        self.typhoon["Vgs"] = Vgs
        
        return self.typhoon
    
    def field_maker(self, grid, north=True):
        self.grid = grid
        self.north = north
        self.tm.north = self.north
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
                                          northern=self.north)
            try:
                wind_pres = np.dstack((wind_pres, self.pres_function(R, entry)))
                wind_spd = np.dstack((wind_spd, wind_speed))
                wind_dir = np.dstack((wind_dir, wind_direction))
            except ValueError:
                wind_spd = np.expand_dims(wind_speed, axis=2)
                wind_dir = np.expand_dims(wind_direction, axis=2)
                wind_pres = np.expand_dims(self.pres_function(R, entry),axis=2)    
                
        self.wind_spd = wind_spd
        self.wind_dir = wind_dir
        self.wind_pres = wind_pres
        self.tm.wind_spd = self.wind_spd 
        self.tm.wind_dir = self.wind_dir 
        self.tm.wind_pres = self.wind_pres   
        
        return None
        
    def error_calc(self, data, RMW, Vgmax, typhoon):
        error = 0
        for datum in data:
            error += (datum[1] - self.wind_optimize(RMW, datum[0], Vgmax, typhoon))**2
        error = (error / len(data)) ** 0.5
        return error
    
    def wind_optimize(self, RMW, r, Vgmax, typhoon):
        if r < RMW:
            wind = Vgmax * (r/RMW)**7.0 * np.exp(7.0*(1.0-r/RMW))
        else:
            wind = Vgmax * np.exp((0.0025*RMW + 0.05)*(1.0-r/RMW))
        
        return wind
    
    def wind_function(self, R, typhoon):
        wind = np.copy(R)
        wind = wind.astype(np.float32)
        
        mask = R < typhoon.RMW
        wind[mask] = typhoon.Vgmax * (R[mask]/typhoon.RMW)**7.0 * np.exp(7.0*(1.0-R[mask]/typhoon.RMW))
        mask = R >= typhoon.RMW
        wind[mask] = typhoon.Vgmax * np.exp((0.0025*typhoon.RMW + 0.05)*(1.0-R[mask]/typhoon.RMW))
        wind = np.nan_to_num(wind)
        
        return wind
    
    def pres_function(self, R, typhoon):
        return typhoon.Pc + typhoon.delP * (np.exp(-typhoon.RMW / R))
    
class Rankine():
    def __init__(self, tm):
        self.tm = tm
        self.typhoon = tm.typhoon
        self.rho_air = self.tm.rho_air
        
    def optimize(self, submethod="Vic&Wad08"):
        '''
        Optimize gradient wind formulation by Holland (1981) by adjusting 
        shape RMW to minize root mean square error in comparison to
        known points based on best track data.
        
        In the absence of known points, shape parameter, and RMW was estimated based on 
        the relationship suggested by Vickery and Madhara (2003)
        
        Using the optimized gradient wind formulation, calculate the Radius of
        Maximum Winds (RMW)

        Returns
        -------
        None

        '''
        RMWs = []
        Xs = []
        if "RMW" in self.typhoon.columns:
            for index, typhoon in self.typhoon.iterrows():
                lat = typhoon.lat
                if np.isnan(typhoon.RMW):
                    if submethod == "Vic&Wad08":
                        RMW = np.exp(3.015 - 6.291*10e-5 * (typhoon.delP)**2 + 0.0337 * lat)
                    if submethod == "Will07":
                        RMW = 46.6 * np.exp(-0.0155 * typhoon.Vgmax) + 0.0169 * lat
                    if submethod == "SGP02":
                        RMW = 0.4785 * typhoon.Pc - 413
                else:
                    RMW = typhoon.RMW
                    
                RMWs.append(RMW)
        else:
            errors = []
            for index, typhoon in self.typhoon.iterrows():
                RMW = 8
                X = 0.5
                lat = typhoon.lat
                Vgmax = typhoon.Vgmax
                data = []
                for radius in self.tm.known_radii:
                    key = f"R{radius:.3f}"
                    if not np.isnan(typhoon[key]):
                        data.append((typhoon[key], radius))
                    
                if data:
                    RMW_max = min([i[0] for i in data])
                    fun = lambda x: self.error_calc(data, x[0], x[1], Vgmax, typhoon)
                    opt = minimize(fun, (RMW, X), method="Powell", 
                                   bounds=((5, min(200,RMW_max)),(0.4,0.6),),
                                   tol=1e-8)
                    RMW = opt.x[0]
                    X = opt.x[1]
                    errors.append(self.error_calc(data, RMW, X, Vgmax, typhoon))
                else:
                    if submethod == "Vic&Wad08":
                        RMW = np.exp(3.015 - 6.291*10e-5 * (typhoon.delP)**2 + 0.0337 * lat)
                    if submethod == "Will07":
                        RMW = 46.6 * np.exp(-0.0155 * Vgmax) + 0.0169 * lat
                    if submethod == "SGP02":
                        RMW = 0.4785 * typhoon.Pc - 413
                Xs.append(X)
                RMWs.append(RMW)
             
            # key =  f"R{self.tm.known_radii[0]:.3f}"   
            # self.typhoon.loc[np.invert(np.isnan(self.typhoon[key])),"errors"] = errors
        self.typhoon["RMW"] = RMWs
        self.typhoon["X"] = Xs
        
        return self.typhoon

    def profiler(self, rs):
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
        Vgs = []
        for index, typhoon in self.typhoon.iterrows():
            Vg = []
            for r in rs:
                Vg.append(self.wind_function(r, typhoon))
            Vgs.append(Vg)
        
        self.typhoon["Vgs"] = Vgs
        
        return self.typhoon
    
    def field_maker(self, grid, north=True):
        self.grid = grid
        self.north = north
        self.tm.north = self.north
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
                                          northern=self.north)
            try:
                wind_pres = np.dstack((wind_pres, self.pres_function(R, entry)))
                wind_spd = np.dstack((wind_spd, wind_speed))
                wind_dir = np.dstack((wind_dir, wind_direction))
            except ValueError:
                wind_spd = np.expand_dims(wind_speed, axis=2)
                wind_dir = np.expand_dims(wind_direction, axis=2)
                wind_pres = np.expand_dims(self.pres_function(R, entry),axis=2)    
                
        self.wind_spd = wind_spd
        self.wind_dir = wind_dir
        self.wind_pres = wind_pres
        self.tm.wind_spd = self.wind_spd 
        self.tm.wind_dir = self.wind_dir 
        self.tm.wind_pres = self.wind_pres   
        
        return None
        
    def error_calc(self, data, RMW, X, Vgmax, typhoon):
        error = 0
        for datum in data:
            error += (datum[1] - self.wind_optimize(RMW, datum[0], X, Vgmax))**2
        error = (error / len(data)) ** 0.5
        return error
    
    def wind_optimize(self, RMW, r, X, Vgmax):
        if r < RMW:
            wind = Vgmax * (r/RMW)
        else:
            wind = Vgmax * (RMW/r) ** X
        
        return wind
    
    def wind_function(self, R, typhoon):
        wind = np.copy(R)
        wind = wind.astype(np.float32)
        
        mask = R < typhoon.RMW
        wind[mask] = typhoon.Vgmax * (R[mask]/typhoon.RMW)
        mask = R >= typhoon.RMW
        wind[mask] = typhoon.Vgmax * (typhoon.RMW/R[mask]) ** typhoon.X
        wind = np.nan_to_num(wind)
        
        return wind
    
    def pres_function(self, R, typhoon):
        return typhoon.Pc + typhoon.delP * (np.exp(-typhoon.RMW / R))
    
    
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
