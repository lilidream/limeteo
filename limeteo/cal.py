"""
计算相关的函数或封装
"""
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units as mpunits
import os
import gc
from tqdm import tqdm
from . import const

def butter_band_pass(data: np.ndarray | xr.DataArray | xr.Dataset, Tmin: float,
                     Tmax: float , N: int = 3, fs: int = 4, axis:int = 0
                    ) -> np.ndarray | xr.DataArray | xr.Dataset:
    """
    使用巴特沃斯带通滤波器滤波
    参考自纪献普师兄的文章: 
    https://blog.csdn.net/weixin_44237337/article/details/124135064

    一般输入一天1-4次的数据。
    
    例如过滤1日4次的某变量数据，带宽为2-10天：
    butter_band_pass(data, 2, 10) 

    ----------------------------------
        
    @param data: ArrayLike 要滤波的数据
    @param Tmin: 最小周期，一般是以天为单位
    @param Tmax: 最大周期，一般是以天为单位
    @param N: 滤波器阶数
    @param fs: 采样频率，数据一天几次就填几
    @param axis: 滤波的轴

    @return: 滤波后的数据，如果是xarray的数据，则只替换数据值，保留其他属性。
            否则返回numpy数组

    """
    import scipy.signal as signal

    b, a = signal.butter(N, [2/(Tmax*fs), 2/(Tmin*fs)], 'bandpass')

    # 如果是xarray的数据，则只替换数据值，保留其他属性
    if isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        filteredData = signal.filtfilt(b, a, data.values, axis=axis)
        data.values = filteredData
        return data
    
    else:
        filteredData = signal.filtfilt(b, a, data, axis=axis)
        return filteredData


def butter_band_pass_clip(data: xr.DataArray, Tmin: float, Tmax: float,
                          clipedMonth: list[int], N: int = 3, fs: int = 4,
                          axis: int = 0) -> xr.DataArray:
    """
    带通滤波封装，适用于每年只对几个月进行滤波的情况。
    使用 xarray 选取比滤波区间更大的时间范围输入。例如要计算678月的滤波结果，
    则传入每年的5-9月的数据，clipedMonth为需要的月份，传入[6,7,8]，计算后会自动裁剪
    为678月的数据返回。

    例：对OLR数据的678月进行滤波

    ```
    # 选择5-9月的数据
    olrData = olrData.sel(time=olrData['time.month'].isin([5,6,7,8,9]))
    # 滤波，2-10天
    olr = butter_band_pass_clip(olrData['olr'], 2, 10, [6,7,8])
    ```

    ----------------------------------

    @param data: xarray.DataArray 要滤波的数据
    @param Tmin: 最小周期，一般是以天为单位
    @param Tmax: 最大周期，一般是以天为单位
    @param clipedMonth: 要滤波/裁剪的月份
    @param N: 滤波器阶数
    @param fs: 采样频率，数据一天几次就填几
    @param axis: 滤波的轴

    @return: 滤波后的结果，会重新构建为xarray.DataArray返回

    """
    filteredData = []

    # 保存原始数据的属性
    refData = data.sel(time=data['time.month'].isin(clipedMonth))
    coords = refData.coords
    dims = refData.dims
    attrs = refData.attrs
    name = refData.name
    del refData

    data = data.groupby('time.year')
    
    # 逐年滤波
    for  year, yearData in tqdm(data, desc="带通滤波[{}]".format(name)):
        yearData = butter_band_pass(yearData, Tmin, Tmax, N, fs, axis)
        yearData = yearData.sel(time=yearData['time.month'].isin(clipedMonth))
        filteredData.append(yearData)
    
    filteredData = np.concatenate(filteredData, axis=0)
    filteredData = xr.DataArray(filteredData, coords=coords, 
                                dims=dims, attrs=attrs, name=name)

    return filteredData


def remove_ERA5_background(data: xr.DataArray) -> xr.DataArray:
    """
    使用预处理的1991-2020气候平均数据，去掉ERA5数据的气候背景。
    数据分别为 /Users/jjli/data/ERA5Climate/ 下的 all.1991-2020.nc 和
    monthly.mean.all.1991-2020.nc 文件。

    ----------------------------------

    @param data: xarray.DataArray 要去除气候背景的数据，不要转换经度。

    @return: 去除气候背景后的数据，会多出一个month维度。
    """
    climateDataPath = "/Users/jjli/data/ERA5Climate/"
    name = data.name
    # 获取metpy的单位
    units = data.metpy.units

    # 去除气候背景
    if os.path.exists(climateDataPath + 'all.1991-2020.nc'):
        climateData = xr.open_dataset(climateDataPath + 'all.1991-2020.nc')

        # 如果数据最大经度大于180，则转换经度
        if data.longitude.max() > 180:
            climateData = reverse_longitude(climateData)

        # 裁剪经纬度到相同范围
        cData = climateData[name].sel(longitude=data.longitude,
                                        latitude=data.latitude)
        data = data - cData * units
    
    else:
        print("在" + climateDataPath + 
              "下找不到气候背景数据文件 'all.1991-2020.nc'，去除气候背景跳过！")
    
    # 去除气候月平均背景
    if os.path.exists(climateDataPath + 'monthly.mean.all.1991-2020.nc'):
        climateData = xr.open_dataset(climateDataPath + 'monthly.mean.all.1991-2020.nc')
        
        # 如果数据最大经度大于180，则转换经度
        if data.longitude.max() > 180:
            climateData = reverse_longitude(climateData)

        # 裁剪经纬度到相同范围
        lon = data.longitude
        lat = data.latitude
        data['month'] = data['time.month']
        months = np.unique(data['month'])
        
        cData = climateData[name].sel(longitude=lon, latitude=lat)
        for m in months:
            data[data['month'] == m] = data[data['month'] == m] - cData.sel(month=m) * units
    
    else:
        print("在" + climateDataPath + 
              "下找不到气候月平均数据文件 'monthly.mean.all.1991-2020.nc'，"
              "去除气候月平均背景跳过！")
    
    return data


def assign_units(variables: list[xr.DataArray] | xr.DataArray, unit: str):
    """
    为数据指定单位，支持单个和多个数据同时指定单位。
    例如： u, v = assign_units([u, v], 'm/s')

    ----------------------------------

    @param variables: list[xarray.DataArray] | xarray.DataArray 要指定单位的数据
    @param unit: str 单位，例如 'm/s'

    @return: 指定单位后的数据，如果是多个数据，则返回list，否则返回单个数据。
    """
    if isinstance(variables, list):
        for var in variables:
            var = var * mpunits(unit)
    else:
        variables = variables * mpunits(unit)
    
    return variables


def vorticity(u: xr.DataArray, v: xr.DataArray, lon: np.ndarray | None = None, 
              lat: np.ndarray | None = None):
    """
    计算涡度，metpy.calc.vorticity()的封装，支持输入经纬度或者xarray.DataArray
    """
    if hasattr(u, 'longitude'):
        dx, dy = mpcalc.lat_lon_grid_deltas(u.longitude, u.latitude)
        vort = mpcalc.vorticity(u, v, dx=dx, dy=dy)

    elif hasattr(u, 'lon'):
        dx, dy = mpcalc.lat_lon_grid_deltas(u.lon, u.lat)
        vort = mpcalc.vorticity(u, v, dx=dx, dy=dy)
        
    elif lon is not None and lat is not None:
        dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
        vort = mpcalc.vorticity(u, v, dx, dy)

    else:
        raise ValueError("无经纬度信息")

    return vort


def Kmean(data, n_clusters=4):
    """
    K-means聚类
    """
    from sklearn.cluster import KMeans
    data = data.reshape(data.shape[0], -1)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(data)
    labels = kmeans.labels_

    return kmeans.labels_


def Kmean_and_silhouette(data, n_clusters=4):
    """
    进行K均值聚类，并计算轮廓系数
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    data = data.reshape(data.shape[0], -1)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(data)
    labels = kmeans.labels_

    return kmeans.labels_, silhouette_score(data, labels)


def MeanShift_and_silhouette(data,  **kwargs):
    """
    MeanShift聚类
    """
    from sklearn.cluster import MeanShift
    from sklearn.metrics import silhouette_score
    data = data.reshape(data.shape[0], -1)
    ms = MeanShift(**kwargs).fit(data)
    labels = ms.labels_

    return ms.labels_, silhouette_score(data, labels)


def reverse_longitude(data, lonName='longitude'):
    """
    将数据的经度从-180-180转换为0-360
    """
    data = data.assign_coords(longitude=(data.coords[lonName].values + 360) % 360)
    data = data.sortby(data.coords[lonName])
    return data

def lon_lat_distance(lon: np.ndarray, lat: np.ndarray) -> tuple[
        np.ndarray[float], np.ndarray[float]
    ]:
    """
    !!! 非椭球模型 !!!

    生成经纬度网格的距离，用于差分计算，如散度涡度等。使用CGCS2000的半长轴的球模型。
    
    分辨率会根据经纬度前两个值的差分别计算，只支持等经纬度格点数据。

    ----------------------------------

    @param lon: 经度，一维数组
    @param lat: 纬度，一维数组

    @return: (dx, dy) 经纬度网格的距离，单位为m，shape为(len(lat), len(lon))
    """

    resolutionX: float = np.abs(lon[1] - lon[0])
    resolutionY: float = np.abs(lat[1] - lat[0])

    dy = np.ones((len(lat), len(lon))) * const.LATITUDE_DISTANCE * resolutionY
    dx = np.ones((len(lat))) * const.LATITUDE_DISTANCE * np.cos(lat * np.pi / 180) * resolutionX
    dx = np.expand_dims(dx, axis=1).repeat(len(lon), axis=1)
    return dx, dy


def vorticity(u: xr.DataArray, v: xr.DataArray, lon: np.ndarray | None = None, 
              lat: np.ndarray | None = None):
    """
    计算涡度，使用`np.gradient()`计算差分。支持拥有longitude/lon, latitude/lat属性
    的xarray.DataArray，或者手动传入经纬度数组。

    将使用 `lon_lat_distance()` 以球面模型计算格点距离。
    自动识别数据纬度维方向，并纠正。

    ----------------------------------

    @param u: xarray.DataArray u分量
    @param v: xarray.DataArray v分量
    @param lon: optional 经度，一维数组
    @param lat: optional 纬度，一维数组

    @return: np.ndarray | xr.DataArray 涡度，shape与u和v相同，如果传入的u和v是
            xarray.DataArray，且带有经纬度，则返回xarray.DataArray，并带坐标。

    """
    if u.shape != v.shape:
        raise ValueError("请确保 u 和 v 的 shape 相同。")

    # 获取经纬度

    if lon is not None and lat is not None:
        if len(lon) != u.shape[-1] and len(lat) == u.shape[-2]:
            raise ValueError("lon 和 lat 的 shape 与 u 和 v 的 shape 不同。")
    
    elif hasattr(u, 'longitude') and isinstance(u.longitude, xr.DataArray):
        lon, lat = u.longitude.values, u.latitude.values
    
    elif hasattr(u, 'lon') and isinstance(u.lon, xr.DataArray):
        lon, lat = u.lon.values, u.lat.values

    else:
        raise ValueError("未找到经纬度信息，请手动设置 lon 和 lat 参数。")

    # 计算格点距离
    dx, dy = lon_lat_distance(lon, lat)

    # 判断纬度方向
    yDir = -1 if lat[0] > lat[1] else 1

    # 求差分涡度
    vor =  np.gradient(v, axis=-1) / dx - yDir * np.gradient(u, axis=-2) / dy

    # 如果数据带有经纬度信息，则将经纬度信息添加到结果中
    if hasattr(u, 'longitude') and isinstance(u.longitude, xr.DataArray):
        vor = xr.DataArray(vor, coords=u.coords, dims=u.dims)

    return vor


def divergence(u: xr.DataArray, v: xr.DataArray, lon: np.ndarray | None = None, 
              lat: np.ndarray | None = None):
    """
    计算散度，使用`np.gradient()`计算差分。支持拥有longitude/lon, latitude/lat属性
    的xarray.DataArray，或者手动传入经纬度数组。

    将使用 `lon_lat_distance()` 以球面模型计算格点距离。
    自动识别数据纬度维方向，并纠正。

    ----------------------------------

    @param u: xarray.DataArray u分量
    @param v: xarray.DataArray v分量
    @param lon: optional 经度，一维数组
    @param lat: optional 纬度，一维数组

    @return: np.ndarray | xr.DataArray 涡度，shape与u和v相同，如果传入的u和v是
            xarray.DataArray，且带有经纬度，则返回xarray.DataArray，并带坐标。

    """
    if u.shape != v.shape:
        raise ValueError("请确保 u 和 v 的 shape 相同。")

    # 获取经纬度

    if lon is not None and lat is not None:
        if len(lon) != u.shape[-1] and len(lat) == u.shape[-2]:
            raise ValueError("lon 和 lat 的 shape 与 u 和 v 的 shape 不同。")
    
    elif hasattr(u, 'longitude') and isinstance(u.longitude, xr.DataArray):
        lon, lat = u.longitude.values, u.latitude.values
    
    elif hasattr(u, 'lon') and isinstance(u.lon, xr.DataArray):
        lon, lat = u.lon.values, u.lat.values

    else:
        raise ValueError("未找到经纬度信息，请手动设置 lon 和 lat 参数。")

    # 计算格点距离
    dx, dy = lon_lat_distance(lon, lat)

    # 判断纬度方向
    yDir = -1 if lat[0] > lat[1] else 1

    # 求差分散度
    div =  np.gradient(u, axis=-1) / dx + yDir * np.gradient(v, axis=-2) / dy

    # 如果数据带有经纬度信息，则将经纬度信息添加到结果中
    if hasattr(u, 'longitude') and isinstance(u.longitude, xr.DataArray):
        div = xr.DataArray(div, coords=u.coords, dims=u.dims)

    return div


def loop_moving_average(d: np.ndarray | list, window: int = 3) -> np.ndarray:
    """
    循环数据的滑动平均，头尾相接滑动平均，滑动窗口为 `window * 2 + 1`。
    ----------------------------------

    @param d: np.ndarray | list 一维数据
    @param window: int 滑动窗口单侧大小

    @return: np.ndarray 滑动平均后的数据
    """

    if isinstance(d, list):
        d = np.array(d)
    
    if len(d.shape) != 1:
        raise ValueError("只支持一维数据")

    # 循环滑动平均
    d = np.concatenate((d[-window:], d, d[:window]))
    dd = np.zeros_like(d)
    for i in range(window, len(d)-window):
        dd[i] = np.mean(d[i-window:i+window+1])
    dd = dd[window:-window]
    return dd


def ttest(mean1: np.ndarray, mean2: np.ndarray, var1: np.ndarray, 
          var2: np.ndarray, n1: int, n2: int):
    """
    计算两个均值场相见后 T-test 的值

    ------------------------------

    @parma mean1: np.ndarray 第一个均值场
    @parma mean2: np.ndarray 第二个均值场
    @parma var1: np.ndarray 第一个均值场的方差场
    @parma var2: np.ndarray 第二个均值场的方差场
    @parma n1: int 第一个均值场计算均值的时间序列长度
    @parma n2: int 第二个均值场计算均值的时间序列长度

    @return np.ndarray t-test结果的值，shape与输入相同
    """
    d = mean1 - mean2
    s = np.sqrt((var1+var2) / (n1+n2-2) * (1/n1 + 1/n2))
    return d / s


def grid_in_shp(lon: np.ndarray, lat: np.ndarray, shpFile: str,
                areaName: str, radius: float = None):
    """
    判断经纬度点是否在SHP文件指定区域内（不一定适合每种shp文件）

    @param lon: np.ndarray 经度数组
    @param lat: np.ndarray 纬度数组
    @param shpFile: str SHP文件路径
    @param areaName: str 要判断的区域名称，字符串
    @param radius: 格点的半径，会判断在此半径内是否在区域内。不填则为点。

    @return: np.ndarray shape=(lat.shape, lon.shape)，在区域内值为1，否则为0.
    """
    import cartopy
    from matplotlib.path import Path
    shp = cartopy.io.shapereader.Reader(shpFile)
    geo = shp.records()
    
    clip = []
    result = np.zeros((len(lat),len(lon)))

    for i in geo:
        if i.attributes['FCNAME'].strip(b'\x00'.decode()) == areaName:
            clip.append(i.geometry)
    
    for c in clip:
        path=Path.make_compound_path(*geos_to_path([c]))
    
    for i in range(len(lat)):
        for j in range(len(lon)):
            if path.contains_point((lon[j],lat[i]), radius=radius):
                result[i,j] = 1
    
    return result


def sk(x):
    """
    MK检验的sk,输入1维时间序列
    @param x: 一维时间序列
    """
    SK = [0]
    n = len(x)
    sk = 0
    E = [0]
    Var = [0]
    UF = [0]
    for i in range(1,n):
        for j in range(i):
            if x[j] < x[i]:
                sk += 1
        SK.append(sk)
        E.append((i+0)*(i+1)/4)
        Var.append((i+1)*(i)*(2*(i+1)+5)/72)
        UF.append((SK[i]-E[i])/np.sqrt(Var[i]))
    UF = np.array(UF)
    return UF

def MK(x):
    """
    MK检验,输入1维时间序列
    @param x: 一维时间序列
    """
    UF = sk(x)
    UB = sk(x[::-1])
    UB = -UB[::-1]
    return UF, UB

def corrcoef(x,y, alpha=0.95):
    import scipy
    """
    计算相关系数
    @param x: 一维时间序列
    @param y: 一维时间序列
    @param alpha: 显著性水平
    @return: 相关系数，是否显著(bool)
    """
    r = np.corrcoef(x,y)[0,1]
    t = np.sqrt(len(x)-1) * r / np.sqrt(1-r**2)
    tLim = scipy.stats.t.sf(t, len(x)-2)
    return r, tLim >= alpha