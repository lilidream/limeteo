import cartopy
from matplotlib.path import Path
from cartopy.mpl.patch import geos_to_path
from cartopy import crs as ccrs
import numpy as np

def clip_by_shp(shpFile, areaName, contourf):
    """
    用SHP文件裁剪contourf或contour，不同来源的SHP文件需要修改代码
    @param shpFile: SHP文件路径
    @param areaName: 要裁剪的区域名称，字符串或字符串列表，例如"四川省"或["四川省","广东省"]
    @param contourf: 要裁剪的contourf或contour对象
    @return: None
    """
    if type(areaName) == str:
        areaName = [areaName]

    shp = cartopy.io.shapereader.Reader(shpFile)
    geo = shp.records()
    
    clip = []

    for i in geo:
        if i.attributes['FCNAME'].strip(b'\x00'.decode()) in areaName:
            clip.append(i.geometry)
    
    for c in clip:
        path=Path.make_compound_path(*geos_to_path([c]))

    for collection in contourf.collections:
        collection.set_clip_path(path, transform=contourf.axes.transData)

def clip_by_province(areaName, contourf):
    """
    用自带的SHP文件裁剪contourf或contour，不同来源的SHP文件需要修改代码
    @param areaName: 要裁剪的区域名称，字符串或字符串列表，例如"四川省"或["四川省","广东省"]
    @param contourf: 要裁剪的contourf或contour对象
    @return: None
    """
    if type(areaName) == str:
        areaName = [areaName]

    shp = cartopy.io.shapereader.Reader("./china_shapefiles/china.shp")
    geo = shp.records()
    
    clip = []

    for i in geo:
        if i.attributes['FCNAME'].strip(b'\x00'.decode()) in areaName:
            clip.append(i.geometry)
    
    for c in clip:
        path=Path.make_compound_path(*geos_to_path([c]))

    for collection in contourf.collections:
        collection.set_clip_path(path, transform=contourf.axes.transData)



def draw_shp(ax, shpFile, areaName, **kwargs):
    """
    画SHP文件中指定区域
    @param ax: 要画到的Axes对象
    @param shpFile: SHP文件路径
    @param areaName: 要画的区域名称，字符串或字符串列表，例如"四川省"或["四川省","广东省"]
    @param kwargs: 传递给ax.add_geometries的参数
    """
    if type(areaName) == str:
        areaName = [areaName]

    shp = cartopy.io.shapereader.Reader(shpFile)
    geo = shp.records()
    
    clip = []

    for i in geo:
        if i.attributes['FCNAME'].strip(b'\x00'.decode()) in areaName:
            clip.append(i.geometry)
    
    for c in clip:
        ax.add_geometries([c], ccrs.PlateCarree(), **kwargs)

def draw_province(ax, areaName, **kwargs):
    """
    画内置SHP文件中指定区域
    @param ax: 要画到的Axes对象
    @param areaName: 要画的区域名称，字符串或字符串列表，例如"四川省"或["四川省","广东省"]
    @param kwargs: 传递给ax.add_geometries的参数
    """
    if type(areaName) == str:
        areaName = [areaName]

    shp = cartopy.io.shapereader.Reader("./china_shapefiles/china.shp")
    geo = shp.records()
    
    clip = []

    for i in geo:
        if i.attributes['FCNAME'].strip(b'\x00'.decode()) in areaName:
            clip.append(i.geometry)
    
    for c in clip:
        ax.add_geometries([c], ccrs.PlateCarree(), **kwargs)



def era5_time_to_text(hoursTo1900, fmt="%Y-%m-%d %H:%M", BJT=False):
    """
    将ERA5的时间格式转换为字符串
    @param int hoursTo1900: ERA5的时间
    @param str fmt: 字符串格式
    """
    from datetime import datetime, timedelta
    if BJT:
        fmt += " BJT"
        hoursTo1900 += 8
    else:
        fmt += " UTC"
    return (datetime(1900,1,1)+timedelta(hours=int(hoursTo1900))).strftime(fmt)

def get_geo_height(lon, lat):
    """
    获取地形高度，输入格点的一维经纬度，返回格点的地形高度
    @param lon: 经度数组
    @param lat: 纬度数组
    """
    import xarray as xr
    d = xr.open_dataset("./data/geo data.nc")
    return d['z'].interp(lon=lon, lat=lat).values

class EOF:
    def __init__(self, data, lat=None):
        from eofs.standard import Eof
        # 检查数据维度，转换成四维
        if(len(data.shape) < 4):
            d = []
            for i in range(len(data)):
                d.append([data[i]])
            data = np.array(d)
        
        self.data = data

        # 实例化eof对象
        if lat is None:
            coslat = np.cos(np.deg2rad(lat)).clip(0., 1.)
            wgts = np.sqrt(coslat)[..., np.newaxis]

            self.eof = Eof(data, weights=wgts)
        else:
            self.eof = Eof(data)

    def eofs(self, neofs=None):
        """
        获取EOF空间模态
        @param int neofs: EOF模态数
        @return: EOF空间模态
        """
        return self.eof.eofsAsCovariance(neofs=neofs)
        
    def pcs(self, npcs=None, pcscaling=1):
        """
        获取EOF时间模态
        @param int neofs: EOF模态数
        @return: EOF时间模态
        """
        return self.eof.pcs(npcs=npcs, pcscaling=pcscaling)



if __name__ == "__main__":
    print(clip_by_shp('./china_shapefiles/china.shp',"四川省"))