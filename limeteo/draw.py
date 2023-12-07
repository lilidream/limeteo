from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from cartopy import crs as ccrs
import numpy as np
import xarray as xr
from matplotlib.contour import QuadContourSet

def basemap(fig: Figure, mapRange: list, nrows: int = 1, ncols: int = 1,
            index: int = 1, proj: ccrs.Projection = None,
            central_longitude: float = 0,
            coastlines: bool | dict = {"color": "black", "linewidth": 0.5},
            gridlines: bool | dict = {"color": "k", "linestyle": (0, (5, 5)),
                                      "linewidth": 0.5},
            lon_ticks: list | None = None, lat_ticks: list | None = None,
            ):
    """
    以 subplot 画基本地图底图

    ----------------------------------

    @param fig: Figure
    @param mapRange: 地图范围，[lon_min, lon_max, lat_min, lat_max]
    @param nrows: int, optional, add_subplot 的参数
    @param ncols: int, optional, add_subplot 的参数
    @param index: int, optional, add_subplot 的参数
    @param proj: ccrs.Projection, optional, 投影方式，默认为 PlateCarree
    @param central_longitude: float, optional, proj=None 时 PlateCarree 投影方式
        的中心经度，默认为 0。
    @param coastlines: bool | dict, optional, 是否画海岸线，如果为 dict，则作为
        ax.coastlines 的参数传入
    @param lon_ticks: list | None, optional, 指定经度刻度，不设置则5度间隔
    @param lat_ticks: list | None, optional, 指定纬度刻度，不设置则5度间隔

    """
    if proj is None:
        proj = ccrs.PlateCarree(central_longitude=central_longitude)

    ax = fig.add_subplot(nrows, ncols, index, projection=proj)
    ax.set_extent(mapRange, crs=proj)

    if coastlines:
        ax.coastlines(**coastlines)

    if lon_ticks is None:
        lon_ticks = range(int(mapRange[0]), int(mapRange[1])+1, 5)
    if lat_ticks is None:
        lat_ticks = range(int(mapRange[2]), int(mapRange[3])+1, 5)

    if gridlines:
        gl = ax.gridlines(draw_labels=True, xlocs=lon_ticks, ylocs=lat_ticks,
                          **gridlines)
        gl.top_labels = False
        gl.right_labels = False

    return ax

def quiver(x: xr.DataArray | np.ndarray, y: xr.DataArray | np.ndarray,
           u: xr.DataArray | np.ndarray, v: xr.DataArray | np.ndarray, 
           ax=None, proj: ccrs.Projection = ccrs.PlateCarree(),
           scale: float = 1, units: str = 'xy', width: float = 0.1,
           lowRes: int = 4, zorder:int = 5):
    
    if ax is None:
        ax = plt.gca()
    
    q = ax.quiver(x[::lowRes], y[::lowRes], 
                  u[::lowRes, ::lowRes], v[::lowRes, ::lowRes], 
                  transform=proj, scale=scale, units=units,
                  width=width, zorder=zorder)
    return q

def quiverkey(q, x: float = 0.92, y: float = 0.97, length: float = 1,
               unit:str = 'm/s', fontsize: int = 12):
    plt.quiverkey(q, x, y, length, "%s %s" % (length, unit), labelpos='E', 
                  coordinates='figure', fontproperties={'size': fontsize})

def colorbar(fig: Figure, axesRange: list, C, horizontal: bool = True, **kwargs):

    ax = fig.add_axes(axesRange)
    cb = plt.colorbar(C, cax=ax, 
                 orientation='horizontal' if horizontal else 'vertical',
                 **kwargs)
    return cb

def subplot_label(text: str, x: float = 0.02, y: float = 0.97, 
                  ax=None, bg: bool = True, bgPad=1, zorder=20, fontsize=12,
                  ha='left', va='top'):
    if ax is None:
        ax = plt.gca()

    if bg:
        ax.text(x, y, text, transform=ax.transAxes, bbox=dict(
            facecolor='white', edgecolor='none', pad=bgPad
            ), ha=ha, va=va, zorder=zorder, fontsize=fontsize)
    else:
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, 
                zorder=zorder, fontsize=fontsize)

def part_of_cmap(cmap, start: float = 0, end: float = 1):
    """
    裁剪自带的 colormap 的一部分，生成新的 colormap

    """
    
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    
    cmap = cm.get_cmap(cmap)
    colors = cmap(np.linspace(start, end, 256))
    new_cmap = LinearSegmentedColormap.from_list('part_of_%s' % cmap.name, colors)
    return new_cmap


def fill_land(color: str = '#888888',zorder: int = 0):
    """
    陆地填色
    """

    from cartopy.feature import LAND
    ax = plt.gca()
    ax.add_feature(LAND, facecolor=color, zorder=zorder)


def subplot_gl(ax, top=False, right=False, left=False, bottom=False,
               linestyle='--', linewidth=0.5, color='gray',
               xlocs=[-160, 120, 140, 160, 180],
               ylocs=[-20, -10, 0, 10, 20], **kwargs):
    """
    邮票图画经纬度网格
    根据情况设置是否画上下左右的经纬度标签
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(0), draw_labels=True,
                    linewidth=linewidth, color=color, linestyle=linestyle,
                    xlocs=xlocs, ylocs=ylocs, **kwargs)
    gl.top_labels = top
    gl.right_labels = right
    gl.left_labels = left
    gl.bottom_labels = bottom

def set_hatch_color(contourf: QuadContourSet, color: str | list[str],
                     borderWidth=0):
    """
    设置用 contourf 画的 hatch 的颜色

    @param contourf: contourf 的返回对象。
    @param color: hatch 的颜色，可以是单个颜色，也可以是一个 list，
        对应每个 hatch 的颜色。
    @param borderWidth: hatch 外边界宽度，一般设置为0隐藏。
    """

    for i, collection in enumerate(contourf.collections):
        if isinstance(color, list):
            collection.set_edgecolor(color[i])
        else:
            collection.set_edgecolor(color)

    for collection in contourf.collections:
        collection.set_linewidth(borderWidth)


def grid_plot(row: int, col: int, ratio: float = 1.5, 
              figsize: tuple = None, figZoom: float = 1,
              sub_adj: tuple = (0.05, 0.95, 0.05, 0.95),
              wSpace: float = 0.05, hSpace: float = 0.05,
              returnGS: bool = False, subplot={}, **kwargs):
    """
    画阵列多子图框架，使用时用 axes 制图：
    ```
    row = 5
    col = 5
    fig, axes = grid_plot(row, col)

    for i in range(row):
        for j in range(col):
            ax = axes[i, j]
            ax.plot(yourData)

    ```

    Parameters
    ----------
    row : int
        行数
    col : int
        列数
    ratio : float
        子图的长宽比，用于自动确定figsize，figsize=None时有效
    figsize : tuple, optional
        图片大小，None时自动计算, by default None
    figZoom : float, optional
        图片放大倍数, by default 1，每一张子图为 figZoom(inch) 高。
    sub_adj : tuple, optional
        子图间距，(left, right, bottom, top), 同 subplots_adjust, 
        by default (0.05, 0.95, 0.05, 0.95)
    wSpace : float, optional
        子图水平间距, by default 0.05
    hSpace : float, optional
        子图垂直间距, by default 0.05
    subplot : dict, optional
        子图参数, by default 
    returnGS : bool, optional
        是否返回GridSpec对象，否则返回subplot数组。 by default False
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        图片对象
    axes : list | matplotlib.gridspec.GridSpec
        子图数组或GridSpec对象

    """
    from matplotlib.gridspec import GridSpec

    if figsize is None:
        # 绘图区高度
        fig_ratio = row*ratio/col
        figHeight = row * figZoom / (sub_adj[3] - sub_adj[2])
        figWidth = row * figZoom * fig_ratio / (sub_adj[1] - sub_adj[0])
        figsize = (figWidth, figHeight)

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=sub_adj[0], right=sub_adj[1],
                        bottom=sub_adj[2], top=sub_adj[3])
    gs = GridSpec(row, col, figure=fig, wspace=wSpace, hspace=hSpace, **kwargs)

    if returnGS:
        return fig, gs

    axes = []
    for i in range(row):
        axes.append([])
        for j in range(col):
            axes[i].append(fig.add_subplot(gs[i, j], **subplot))
    
    return fig, axes

def emphasis_contour(ax, x, y, data, level: list[float] | np.ndarray,
                     emphasisLevel: float | list[float] | np.ndarray, 
                     colors='k', emphasisColor='r', linewidths=0.8,
                     emphasisLinewidth=1.3, contourArgs={}, clabelArgs={}) -> None:
    """
    画等值线，并强调其中某一些

    Parameters
    ----------
    ax: axes
        当前 axes
    x: array-liked
        contour的参数
    y: array-liked
        contour的参数
    data: array-liked
        contour的参数
    level: array-liked
        要画的等值线的值，包含要强调的值
    emphasisLevel: float | array-liked
        要强调的等值线的值
    colors: matplotlib colors
        非强调的等值线的颜色
    emphasisColor: matplotlib colors
        强调的等值线的颜色
    linewidths: float
        非强调的等值线宽度
    emphasisLinewidth: float
        强调的等值线宽度
    contourAugs: dict
        contour() 函数的其他参数，以字典形式传入
    clabel: dict
        clabel() 函数的其他参数，以字典形式传入

    """
    level = np.array(level)

    if not isinstance(emphasisLevel, list) and \
        not isinstance(emphasisLevel, np.ndarray):
        emphasisLevel = [emphasisLevel]

    emphasisLevel = np.array(emphasisLevel)
    level = np.delete(level, np.where(np.isin(level, emphasisLevel)))
    
    C1 = ax.contour(x, y, data, level, colors=colors, linewidths=linewidths, **contourArgs)

    C2 = ax.contour(x, y, data, emphasisLevel, colors=emphasisColor, 
               linewidths=emphasisLinewidth, **contourArgs)
    
    for C in [C1, C2]:
        clabel = ax.clabel(C, zorder=50, **clabelArgs)
        for c in clabel:
            c.set_bbox(dict(facecolor='white', edgecolor='None', pad=0.1))
    
def subplot_tick(xtick: list[float] | np.ndarray, ytick: list[float] | np.ndarray,
                 showYTickLabel: bool = True, showXTickLabel: bool = True, ax=None,
                 xAppend: str=None, yAppend: str=None, **kwargs):
    """
    多子图画坐标轴刻度，只在边缘画。例如

    ```
    for i in range(3):
        for j in range(4):
            ax = fig.add_subplots(3, 4, i*4+j)

            # 只在子图阵列的左边和下边显示 tick 的数值
            subplot_tick([0, 1, 2], [0, 1, 2],
                         showYTickLabel = j == 0,
                         showXTickLabel = i == 2)

    ```
    Parameters
    ----------
    xtick: array-liked
        x刻度值
    ytick: array-liked
        y刻度值
    showYTickLabel: bool
        是否显示y刻度的值
    showXTickLabel: bool
        是否显示x刻度的值
    ax: axes
        axes
    xAppend: str
        xtick的后缀，例如表示经度时可以加 "E"
    yAppend: str
        ytick的后缀

    """
    if ax is None:
        ax = plt.gca()

    if xAppend is not None:
        xtickLabel = [str(i)+xAppend for i in xtick]
    
    if yAppend is not None:
        ytickLabel = [str(i)+yAppend for i in ytick]

    ax.set_xticks(xtick)
    ax.set_xticklabels(xtickLabel if showXTickLabel else [], **kwargs)
    ax.set_yticks(ytick)
    ax.set_yticklabels(ytickLabel if showYTickLabel else [], **kwargs)
