"""
数据、配置等文件的读写
"""
import xarray as xr
import numpy as np
from metpy.units import units as mpunits
import time

ERA5_DATA_PATH = '/Datadisk/ERA5/4xdaily0.5/'

def reverse_longitude(data: xr.DataArray | xr.Dataset,
                      lonName='longitude'):
    """
    将数据的经度从-180-180转换为0-360
    """
    data = data.assign_coords(longitude=(data.coords[lonName].values + 360) % 360)
    data = data.sortby(data.coords[lonName])
    return data

def get_ERA5_data_by_month(yearRange: list[int, int], monthList: list[int], 
                           dataName: str, mapRange: list[int] | None = None,
                           levels: list[int] | int | None = None,
                           units: bool | str = True) -> xr.Dataset:
    """
    从服务器的ERA5数据库中提取合并数据，适用于提取部分月份的数据。
    数据原始范围是-180~180，如果传入经度超过180，则转换成0~360。

    @param yearRange: 要提取的年份范围，二元列表，如[1991, 2020]
    @param monthList: 要提取的月份列表，如[6,7,8]
    @param dataName: 要提取的数据的文件名称，有效值：'geop', 'uwnd', 'vwnd',
                     'vvel', 'temp', 'sph', 'rh'
    @param mapRange: 要提取的地图范围，四元列表，如[100, 180, -20, 20]，
                     不设置则不裁剪。
    @param levels: 要提取的高度，单个层或列表，如[1000, 850, 500, 200]，
    @param units: 是否为数据指定单位，True为自动指定，False为不指定，
                  输入字符串可手动指定
    
    @return: 提取的数据
    """
    _units = {
        'geop': 'm**2/s**2',
        'uwnd': 'm/s',
        'vwnd': 'm/s',
        'vvel': 'Pa/s',
        'temp': 'K',
        'sph': 'kg/kg',
        'rh': '%'
    }

    if dataName not in ['geop', 'uwnd', 'vwnd', 'vvel', 'temp', 'sph', 'rh']:
        raise ValueError("无效的ERA5数据变量名: " + dataName, + 
                         "，有效值：'geop', 'uwnd', 'vwnd', 'vvel',"
                         " 'temp', 'sph', 'rh'。")
    
    # 读取数据
    paths = []
    for year in range(yearRange[0], yearRange[1]+1):
        for month in monthList:
            paths.append(ERA5_DATA_PATH + dataName + '/' + dataName + '.' + 
                         str(year) + str(month).zfill(2) + '.nc')
    
    data = xr.open_mfdataset(paths, combine='by_coords')


    if levels is not None:
        if isinstance(levels, int):
            data = data.sel(level=[levels])
        else:
            data = data.sel(level=levels)

    if mapRange is not None:
        # 转换经度范围
        if mapRange[0] > 180 or mapRange[1] > 180:
            data = reverse_longitude(data)

        data = data.sel(latitude=slice(mapRange[3], mapRange[2]), 
                        longitude=slice(mapRange[0], mapRange[1]))
    
    if units is True:
        data = data * mpunits(_units[dataName])
    elif isinstance(units, str):
        data = data * mpunits(units)

    return data
    

# 数据保存存档系列函数
import sqlite3 as sql

DB_PATH = '/Users/jjli/archive/archive.db'
FIG_DB_PATH = '/Users/jjli/archive/figure.db'

def _connect_db():
    return sql.connect(DB_PATH)

def _create_table():
    conn = _connect_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS record
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL DEFAULT '',
        path TEXT NOT NULL,
        createTime TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
        dataDesc TEXT,
        author TEXT NOT NULL,
        fileType TEXT NOT NULL,
        desc TEXT NOT NULL);
        ''')
    conn.commit()
    conn.close()
    print("Table created successfully")

def _create_img_table():
    conn = _connect_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS image
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL DEFAULT '',
        path TEXT NOT NULL,
        class TEXT NOT NULL,
        createTime TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
        author TEXT NOT NULL,
        desc TEXT NOT NULL);
        ''')
    conn.commit()
    conn.close()
    print("Table created successfully")

def save_xr_dataset(dataset: xr.Dataset, path: str, desc: str = '',
                    name: str = '', author: str = 'jjli'):
    """
    保存xarray数据，并记录到数据库中
    
    Parameters
    ----------
    dataset : xarray.Dataset
        xarray数据
    path : str
        数据保存路径
    desc : str
        数据描述
    name : str
        数据名称，不指定则使用path中的文件名
    author : str
        作者

    """
    
    xrDesc = dataset.__repr__()
    dataset.to_netcdf(path)

    conn = _connect_db()
    
    # 检查有无path相同的数据

    c = conn.cursor()
    c.execute('''SELECT * FROM record WHERE path=?''', (path,))

    if name == '':
        name = path.split('/')[-1]

    # 如果有，更新数据，没有则插入数据

    if c.fetchone() is not None:
        c.execute('''UPDATE record SET name=?,  dataDesc=?, author=?, desc=?, createTime=(datetime('now', 'localtime')) WHERE path=?;''',
        (name, xrDesc, author, desc, path))
    else:
        c.execute('''INSERT INTO record (name, path, dataDesc, author, fileType, desc) VALUES (?, ?, ?, ?, ?, ?);''',
        (name, path, xrDesc, author, 'netcdf', desc))
    
    conn.commit()
    conn.close()

def savefig(path: str, imgClass: str, figure = None, desc: str = '',
            name: str = '', author: str = 'jjli', dpi: int = 200):
    """
    保存matplotlib图像，并记录到数据库中
    """
    from matplotlib import pyplot as plt

    if figure is None:
        figure = plt.gcf()
    
    figure.savefig(path, dpi=dpi)

    conn = _connect_db()

    # 检查有无path相同的数据

    c = conn.cursor()
    c.execute('''SELECT * FROM image WHERE path=?''', (path,))

    if name == '':
        name = path.split('/')[-1]
    
    # 如果有，更新数据，没有则插入数据

    if c.fetchone() is not None:
        c.execute('''UPDATE image SET name=?, author=?, desc=?, class=?, createTime=(datetime('now', 'localtime')) WHERE path=?;''',
        (name, author, desc, imgClass, path))
    else:
        c.execute('''INSERT INTO image (name, path, author, class, desc) VALUES (?, ?, ?, ?, ?);''',
        (name, path, author, imgClass, desc))
    
    conn.commit()
    conn.close()

def save_json(dictData: dict, path: str,  desc: str = '', 
            name: str = '', author: str = 'jjli'):
    """
    保存json数据，并记录到数据库中
    """
    import json

    with open(path, 'w') as f:
        json.dump(dictData, f)

    conn = _connect_db()

    # 检查有无path相同的数据

    c = conn.cursor()
    c.execute('''SELECT * FROM record WHERE path=?''', (path,))

    if name == '':
        name = path.split('/')[-1]

    # 如果有，更新数据，没有则插入数据

    if c.fetchone() is not None:
        c.execute('''UPDATE record SET name=?, author=?, desc=?, createTime=(datetime('now', 'localtime')) WHERE path=?;''',
        (name, author, desc, path))
    else:
        c.execute('''INSERT INTO record (name, path, author, fileType, desc) VALUES (?, ?, ?, ?, ?);''',
        (name, path, author, 'json', desc))
    
    conn.commit()
    conn.close()

def load_json(path: str):
    """
    从json文件中读取数据
    """
    import json

    with open(path, 'r') as f:
        data = json.load(f)
    
    return data

def just_save_json(dictData: dict, path: str):
    import json

    with open(path, 'w') as f:
        json.dump(dictData, f)

import json
GALLERY_PATH = '/Users/jjli/archive/gallery.json'
GALLERY_SERIES_PATH = '/Users/jjli/archive/gallery_series.json'
# 2023-11-10 重新制作图片保存，按照key=>[v1,v2,...]来分类保存，保存为yaml文件
# 2023-11-11 还是用回json
def gallery_append(path, group, desc, name, author, series):
    """
    将图片信息追加到gallery.json文件中
    """
    with open(GALLERY_PATH, 'r', encoding='utf-8') as f:
        gallery = json.loads(f.read())
    
    strTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 检查有无path相同的数据
    append = True
    for item in gallery:
        if item['path'] == path:
            # update
            item['name'] = name
            item['desc'] = desc
            item['author'] = author
            item['group'] = group
            item['time'] = strTime
            item['series'] = series
            append = False
            break
    
    if append:
        id = round(time.time() * 1000)
        item = {
            'id': id,
            'name': name,
            'path': path,
            'desc': desc,
            'author': author,
            'group': group,
            'time': strTime,
            'series': series
        }
        gallery.append(item)

    with open(GALLERY_PATH, 'w', encoding='utf-8') as f:
        json.dump(gallery, f, ensure_ascii=False, indent=4)

def gallery_series_append(series, seriesDesc):
    """
    将系列信息追加到gallery_series.json文件中
    {"series_name": {
        "name": "series_name",
        "desc": "series_desc",
        "editTime": "2021-11-11 11:11:11"
    }}
    """
    with open(GALLERY_SERIES_PATH, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    
    existed = False
    for item in data:
        if item == series:
            data[item]['desc'] = seriesDesc
            data[item]['editTime'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            existed = True
            break
    
    if not existed:
        item = {
            'name': series,
            'desc': seriesDesc,
            'editTime': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        data[series] = item
    
    with open(GALLERY_SERIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    

def savefig_in_gallery(path: str, group:dict, series:str, seriesDesc: str,
                        figure = None, desc: str = '', name: str = "",
                        author = "jjli", dpi=200):
    """
    保存并记录在json数据中，用于图库。
    series是每一次画图的一个系列区别，series是系列的名称，不能重复，seriesDesc填这次画图的详细描述。

    -------
    @param path: 保存路径
    @param group: 保存的数据，字典，key为分类，value为值
        如{"class":"MRG", "BPF": "2-10", ...}，key要确保相同，value可以是单值也可以是列表

    """
    from matplotlib import pyplot as plt

    if figure is None:
        figure = plt.gcf()
    
    figure.savefig(path, dpi=dpi)

    # 写入json文件

    if name == '':
        name = path.split('/')[-1]

    gallery_append(path, group, desc, name, author, series)

    # 写入series
    gallery_series_append(series, seriesDesc)


if __name__ == "__main__":
    # _create_table()
    _create_img_table()