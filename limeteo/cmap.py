"""
自定义色条及相关内容

2023-11-19 by Lilidream



"""

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.colors import Colormap
import numpy as np

class LmCmap:
    """
    LiMeteo 的 colormap 类，方便对 colormap 进行各类操作。
    
    -----------------------

    ## 如何使用

    ### 注册 cmap
    使用 `lmCmap.register_lm_cmap()` 将此模组的 cmap 注册到 matplotlib 中。

    使用时直接使用名称即可，默认是以 "lm_" 为前缀的，例如 "lm_twilight"。

    ![cmaps](./doc/colormaps.png)

    所有的 colormap 可以在 doc/colormaps.png 中有例图， 或通过 draw_all_cmap() 来查看。


    ### 编辑 cmap

    要使用此模块编辑 cmap，先创建一个实例，可以使用 cmap 的名字或一组颜色来创建一个 LmCamp 对象。
    ```
    myCmap1 = LmCmap("RdBu")          # 使用 matplotlib 自带的 RdBu 创建一个实例
    myCmap2 = LmCmap("lm_twilight")   # 使用 LmCamp 自带的 twilight 创建一个实例
    myCmap3 = LmCmap(['r', 'g', 'b']) # 使用一组 matplotlib 颜色创建一个实例
    ```

    创建对象后，可以通过以下方法编辑：
    - 相接：使用加号可以将 LmCmap 实例与另一个 cmap 拼接，
        可接受字符串的 cmap 名称、matplotlib 的 colormap 或 LmCmap 实例。

        ```
        newCmap = myCmap1 + myCmap2
        newCmap = myCmap1 + "jet"
        ```

    - 数乘：将 LmCmap 乘一个正整数，表示将自己重复n次。
    - 裁剪：使用 `clip()` 函数裁剪保留需要的部分。
        
        ```
        newCmap = myCamp.clip(0.2, 0.8)  # cmap 的坐标为从0~1，只保留0.2~0.8的部分
        ```

    - 减去：使用减号减一个二元元组，表示去掉 cmap 的这部分。

        ```
        newCmap = myCamp - (0.2, 0.8)  # 去掉 cmap 的 0.2~0.8 部分，剩下的自动拼接。
        ```

    - 左右移动：使用 `<<` 或 `>>` 与一个 0~1 内的数令 cmap 循环偏移（即左移右移）

        ```
        newCmap = myCamp << 0.3 
        ```

    - 取负值：前面加负号 `-` 表示翻转 cmap。
    - 使用 `concat()` 拼接多个 cmap 并可指定比例。
    - 使用以下函数可调整 cmap 整体的颜色，调整范围从 -1~1，正值增强，负值减弱：
        - `brightness()`
        - `contrast()`
        - `saturation()`
        - `hue()`
        所有函数都支持链式使用：

        ```
        newCmap = myCamp.clip(0.2, 0.5).hue(0.1).brightness(0.5) + 'jet'
        ```

    ### 使用 cmap

    要将 LmCmap 使用到绘图中，需要使用 `cmap()` 或直接调用对象，从而获得 matplotlib 的 colormap。

    ```
    myCmap = LmCmap('RdBu')
    newCmap = myCamp.clip(0.2, 0.5).hue(0.1).brightness(0.5) + 'jet'

    plt.contourf(x, y, z, cmap=newCmap.cmap())  # 使用 cmap()
    plt.contourf(x, y, z, cmap=newCmap())       # 直接调用(call)

    ```

    """

    # Limeteo 内置的 colormap
    __LM_CMAP = {
        "twilight": ["#ec865b","#EBC07E","#D7C9B0","#A7ABB8","#667b99","#3A516E",
                    "#254974"],
        "day_sky": ["#d8dfe8","#7BADF2","#5481D4","#3F65B0","#29498e"],
        "city_night": ["#df793a","#a27a67","#6b616a","#52496f","#483772"],
        "glow_sunset": ["#ee777f","#8d5e8d","#6e5a96","#3971b1","#1c4c87"],

        # div
        "garden": ["#f03628","#f85a16","#ee9c3f","#ffffff","#82cb15","#49c016",
                "#3b8609"],
        "fruit": ["#ca3e1c","#d97736","#e5b42e","#f5d924","#91c346","#309306",
                "#115a12"],
        "sky_sun": ["#0505fa","#1a90ff","#14e4ff","#ffffff","#f5ab56","#ff6d2e",
                    "#fa0505"],
        "BrBl": ["#d47216","#e19819","#e8bd45","#ffffff","#47bdf0","#42a4ff",
                "#336ee6"],
        "BrGn": ["#d47216","#e19819","#e8bd45","#ffffff","#28e684","#17dd13",
                "#00a321"],
        "RYCG": ["#e00000","#ee5e11","#ffae00","#e0d900","#ffffff","#18f7d1",
                "#18ecad","#06ea69","#00cc18"],
        "onion": ["#8704cd","#ff4dde","#ffffff","#82ee3a","#14ad00"],
        "RMBu": ["#bc1074","#ec22db","#ffffff","#379ae6","#0050d1"],
        "OrCy": ["#e64c0a","#e7650d","#ffffff","#14e1d4","#10d2e0"],
        "YROGCB": ["#a20785","#e81717","#eaa510","#ffffff","#1cd942","#07dfdb","#0f0fe6"],

        # rainbow
        "jet": ["#1600bd","#0566e6","#00aaff","#00ffee","#7FFFD4","#02ed3d",
                "#00cc03","#9fed02","#ffed29","#f7b602","#e46701","#d00101"],
        "rainbow": ["#e600d2","#b800eb","#6a16e9","#381ff4","#0566e6","#00aaff",
                    "#09ecdd","#7FFFD4","#02ed3d","#07df0a","#9fed02","#f0de19",
                    "#f7b602","#e46701","#d00101"],
        "WMBGYOR": ["#e3e3e3","#363636","#e65ce1","#dc18b8","#ad5cff","#5306c6",
                    "#57c7ff","#349af9","#0f4cae","#0de76c","#36db33","#3e9f09",
                    "#f8d80d","#b86b00","#ff8533","#d00101"],
    }


    def __init__(self, cmap: str | Colormap | list[str] | dict, 
                 lmCmapPrefix: str = "lm_", N=256, cmapType="linear", name=""):
        """
        创建一个 LmCmap 对象，并设置其 colormap。
        实例初始化时会自动注册 LiMeteo 的 colormap。

        Parameters
        ----------
        cmap : str | Colormap | list[str]
            colormap 的名称或 Colormap 对象，可以是 LiMeteo 中的 Colormap。
            也可以是一个颜色列表，例如 ["#000000", "#ffffff"] 或 ['r', 'g', 'b']，
            来创建一个 colormap，支持 Matplotlib 中所有颜色字符串。
        lmCmapPrefix : str, optional
            LiMeteo 中 colormap 的注册到 Matplotlib 中的前缀，默认为 "lm_"。
        N : int, optional
            colormap 的分段数，默认为 256。
        cmapType : str, 'linear' | 'list', optional
            创建 colormap 的类型，'linear' 表示线性插值，'list' 表示离散。
            当 cmap 为一组颜色创建 colormap 时此值有效，默认为 'linear'。
        name : str, optional
            colormap 的名称，默认为 ""，如果 cmap 为 str 或 Colormap 对象，
            则使用其原有名称。如果 cmap 为颜色列表创建新的 colormap，则使用此名称。
        """
        self.register_lm_cmap(lmCmapPrefix)
        self.N = N

        if isinstance(cmap, str):
            cmap = matplotlib.colormaps.get_cmap(cmap)
            if name != "":
                cmap.name = name

        if isinstance(cmap, list):
            if cmapType == "linear":
                cmap = LinearSegmentedColormap.from_list(name, cmap, N)
            elif cmapType == "list":
                cmap = ListedColormap(cmap, name)
            else:
                raise ValueError("cmapType 只能为 'linear' 或 'list'")
        
        self._cmap = cmap
        return None
    

    def __call__(self):
        """返回 colormap"""
        return self._cmap

    
    def cmap(self):
        """返回 colormap"""
        return self._cmap
    

    def __concat(self, a: Colormap, b: Colormap, ratio: float = 0.5) -> Colormap:
        """
        重新采样并拼接两个 colormap

        Parameters
        ----------
        ratio : float, optional
            拼接比例，0~1之间，表示 a 的比例，b 的比例为 1-ratio
        
        Returns
        -------
        Colormap
            拼接后的 colormap
        """
        ratio = max(min(ratio, 1), 0)

        if ratio <= 0.5:
            nA = self.N
            nB = int(self.N * (1 - ratio) / ratio)
        else:
            nB = self.N
            nA = int(self.N * ratio / (1 - ratio))

        colors = np.vstack((a(np.linspace(0, 1, nA)), 
                            b(np.linspace(0, 1, nB))))
        return LinearSegmentedColormap.from_list('', colors).resampled(self.N)

    
    def __add__(self, other):
        """ 用 “+” 将两个 colormap 拼接起来，拼接的比例对半 """
        if isinstance(other, str) or isinstance(other, Colormap):
            if isinstance(other, str):
                other = matplotlib.colormaps.get_cmap(other)
            self._cmap = self.__concat(self._cmap, other)
        
        elif isinstance(other, LmCmap):
            self._cmap = self.__concat(self._cmap, other._cmap)
        
        else:
            raise TypeError("只能拼接两个 Colormap、LmCmap 对象或使用 colormap 的名称")

        return self


    def __sub__(self, other):
        """
        剪去 colormap，减数为二元元组，例如 (0.2, 0.8)。

        元组的两个值为剪去的范围，0~1之间。
        元组第一个值为0时，表示从头开始剪去，第二个值为1时，表示剪到尾。

        如果剪去区间位于 0~1 之间，则表示从中间挖去这部分然后按原始比例拼接剩余部分。

        例如：
        ```python
        c = LmCmap("lm_twilight")

        # 去掉中间 0.2~0.8 的部分，等价于将 (0, 0.2) 和 (0.8, 1) 拼接。
        clipedCmap = c - (0.2, 0.8) 

        # 去掉前 0.2 的部分，等价于 clipedCmap = c.clip(0.2, 1)
        clipedCmap2 = c - (0, 0.2) 
        ```
        """
        if (not isinstance(other, tuple) and 
            not isinstance(other, list) and 
            not isinstance(other, np.ndarray)):
            raise TypeError("请使用二元元组或列表作为减数裁剪 colormap")
        
        if len(other) != 2:
            raise ValueError("请使用二元元组或列表作为减数裁剪 colormap")
        
        if other[0] < 0 or other[0] > 1 or other[1] < 0 or other[1] > 1:
            raise ValueError("请使用0~1之间的数值作为减数裁剪 colormap")
        
        if other[0] > other[1]:
            raise ValueError("请使用从小到大的数值作为减数裁剪 colormap")
        
        if other[0] == 0 and other[1] == 1:
            self._cmap = self

        if other[0] == 0:
            self.clip(other[1], 1)
        
        elif other[1] == 1:
            self.clip(0, other[0])
        
        else:
            colors = self._cmap(np.linspace(0, 1, self.N))
            colors = np.vstack((colors[:int(self.N * other[0])], 
                                colors[int(self.N * other[1]):]))
            self._cmap = LinearSegmentedColormap.from_list(self._cmap.name, 
                                                           colors)
        
        return self


    @classmethod
    def concat(self, cmaps: list[Colormap], 
               ratios: list[float] = None, N=256) -> Colormap:
        """
        拼接多个 colormap

        Parameters
        ----------
        cmaps : list[Colormap | LmCmap | str]
            colormap 列表，可以是 Colormap 对象、LmCmap 对象或 colormap 名称。

        ratios : list[float], optional
            拼接比例，长度必须与 cmaps 一致，每一个值为 i/sum(ratios)，不传入则均分。

        N : int, optional
            拼接后的 colormap 的分段数，默认为 256，当 colormap 较多时，
            建议设置更大的值以确保不丢失颜色细节。

        Returns
        -------
        Colormap
            拼接后的 colormap。
        """
        if ratios is None:
            ratios = [1] * len(cmaps)
        
        if len(cmaps) != len(ratios):
            raise ValueError("cmaps 和 ratios 长度必须一致")
        
        if len(cmaps) == 1:
            return cmaps[0]

        ratios = np.array(ratios) / sum(ratios)
        cmap = []
        for i in range(len(cmaps)):
            if isinstance(cmaps[i], str):
                cmaps[i] = matplotlib.colormaps.get_cmap(cmaps[i])
            if isinstance(cmaps[i], LmCmap):
                cmaps[i] = cmaps[i].cmap()
            cmap.append(cmaps[i](np.linspace(0, 1, int(N * ratios[i]))))
        cmap = LinearSegmentedColormap.from_list("", np.vstack(cmap))
        
        return cmap

    
    def __mul__(self, other):
        """
        数乘，用 “*” 将 colormap 重复多次
        乘数只能为正整数。
        """
        if not isinstance(other, int) or other <= 0:
            raise TypeError("只能用正整数作为乘数")
        
        self._cmap = self.concat([self._cmap] * other)
        return self
    

    def __neg__(self):
        """ 取反，用 “-” 将 colormap 反转 """
        self._cmap = self._cmap.reversed()
        return self
    

    def __lshift__(self, other):
        """
        左移，用 “<<” 将 colormap 整体颜色循环向左移动
        左移的距离只能为0~1之间的小数
        """
        if not isinstance(other, float) or other < 0 or other > 1:
            raise TypeError("只能用0~1之间的小数作为左移的距离")
        
        colors = self._cmap(np.linspace(0, 1, self.N))
        colors = np.vstack((colors[int(self.N * other):],
                            colors[:int(self.N * other)]))
        self._cmap = LinearSegmentedColormap.from_list(self._cmap.name, colors)
        return self
    

    def __rshift__(self, other):
        """
        右移，用 “>>” 将 colormap 整体颜色循环向右移动
        右移的距离只能为0~1之间的小数
        """
        if not isinstance(other, float) or other < 0 or other > 1:
            raise TypeError("只能用0~1之间的小数作为右移的距离")
        
        colors = self._cmap(np.linspace(0, 1, self.N))
        colors = np.vstack((colors[-int(self.N * other):],
                            colors[:-int(self.N * other)]))
        self._cmap = LinearSegmentedColormap.from_list(self._cmap.name, colors)
        return self


    def __process(self, func):
        """
        对 colormap 每一个颜色进行处理，合成新的 colormap

        Parameters
        ----------
        func : ((r, g, b, a)) -> (r, g, b, a)
            对颜色的处理函数，接受一个rgba颜色元组，并返回一个rgba颜色元组。
        """
        colors = self._cmap(np.linspace(0, 1, self.N))
        colors = [func(c) for c in colors]
        self._cmap = LinearSegmentedColormap.from_list(self._cmap.name, colors)


    def brightness(self, brightness: float):
        """
        改变 colormap 的亮度
        
        Parameters
        ----------
        brightness : float
            亮度，-1~1之间，正值增加亮度，负值减少亮度。
        """
        brightness = max(min(brightness, 1), -1)

        def b(rgba):
            return tuple([
                max(min(c + (1 - c) * brightness, 1), 0) for c in rgba[:3]
            ] + [rgba[3]])

        self.__process(b)
        return self

    def contrast(self, contrast: float):
        """
        改变 colormap 的对比度

        Parameters
        ----------
        contrast : float
            对比度，-1~1之间，正值增加对比度，负值减少对比度。
        """
        contrast = max(min(contrast, 1), -1)

        def cc(rgba):
            return tuple([
                max(min(c + (c - 0.5) * contrast, 1), 0) for c in rgba[:3]
            ] + [rgba[3]])
        
        self.__process(cc)
        return self


    def saturation(self, saturation: float):
        """
        改变 colormap 的饱和度

        Parameters
        ----------
        saturation : float
            饱和度，-1~1之间，正值增加饱和度，负值减少饱和度。
        """
        saturation = max(min(saturation, 1), -1)
        
        def s(rgba):
            h, s, v = matplotlib.colors.rgb_to_hsv(rgba[:3])
            s = max(min(s + s * saturation, 1), 0)
            k = matplotlib.colors.hsv_to_rgb([h, s, v])
            return tuple(list(k) + [rgba[3]])
        
        self.__process(s)
        return self
    

    def hue(self, hue: float):
        """
        改变 colormap 的色相

        Parameters
        ----------
        hue : float
            色相偏移，-1~1之间。
        """
        hue = max(min(hue, 1), -1)
        
        if hue == 0:
            return self

        def h(rgba):
            h, s, v = matplotlib.colors.rgb_to_hsv(rgba[:3])
            h += hue
            if h > 1:
                h -= 1
            elif h < 0:
                h += 1
            k = matplotlib.colors.hsv_to_rgb([h, s, v])
            return tuple(list(k) + [rgba[3]])
        
        self.__process(h)
        return self


    @classmethod
    def register_lm_cmap(self, prefix="lm_"):
        """
        将 LiMeteo 的 colormap 注册到 matplotlib 中。

        @param prefix: colormap 的前缀，默认为 "lm_"
        """
        # 获取已有的 colormap
        cmaps = matplotlib.colormaps()
        
        for name, colors in self.__LM_CMAP.items():
            if prefix + name in cmaps:
                continue

            matplotlib.colormaps.register(
                LinearSegmentedColormap.from_list(prefix + name, colors))
            # 反转
            matplotlib.colormaps.register(
                LinearSegmentedColormap.from_list(prefix + name + '_r',
                                                   colors[::-1]))


    def clip(self, vmin: float, vmax: float, N=None):
        """
        裁剪 colormap，只保留 vmin~vmax 之间的部分。

        Parameters
        ----------
        vmin : float
            裁剪的最小值，0~1之间。
        vmax : float
            裁剪的最大值，0~1之间。
        N : int, optional
            裁剪后的 colormap 的分段数，默认为 256。

        """
        if N is None:
            N = self.N

        colors = self._cmap(np.linspace(0, 1, N))
        colors = colors[int(N * vmin):int(N * vmax)]
        self._cmap = LinearSegmentedColormap.from_list(self._cmap.name, colors)
        return self
    

    @property
    def name(self):
        """ 返回 colormap 的名称 """
        return self._cmap.name
    

    @name.setter
    def name(self, name: str):
        """ 设置 colormap 的名称 """
        self._cmap.name = name
        return self

    
    def setName(self, name: str):
        """ 设置 colormap 的名称 """
        self._cmap.name = name
        return self


    @classmethod
    def draw_all_cmap(self, figsavePath):
        """ 画出所有Limeteo 的 colormap """
        fig = plt.figure(figsize=(6, 0.5*len(self.__LM_CMAP)))
        fig.subplots_adjust(left=0.2, right=0.95, bottom=0.02, top=0.97)
        for i, (name, colors) in enumerate(self.__LM_CMAP.items()):
            ax = fig.add_subplot(len(self.__LM_CMAP), 1, i+1)
            plt.colorbar(cm.ScalarMappable(
                cmap=LinearSegmentedColormap.from_list("", colors)), 
                cax=ax, orientation='horizontal', ticks=[])
            ax.text(-0.01,0.5,name, ha='right', va='center', fontsize=10)
        plt.savefig(figsavePath, dpi=100)
        plt.show()


    def show(self):
        """
        画出当前实例的 colormap 并显示(plt.show())。
        """
        fig = plt.figure(figsize=(6, 1.5))
        
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.3, top=0.8)
        ax = fig.add_subplot(1, 1, 1)

        plt.colorbar(cm.ScalarMappable(cmap=self._cmap), cax=ax, 
                     orientation='horizontal', ticks=np.linspace(0, 1, 11))
        plt.show()

    @classmethod
    def draw_some_cmap(cls, cmaps: list[str | Colormap]):
        """
        传入一组 colormap 并画出来。

        例如：
        ```python
        c = LmCmap("lm_twilight")
        LmCmap.draw_some_cmap(['jet', 'rainbow', c])
        ```

        Parameters
        ----------
        cmaps : list[str | Colormap]
            colormap 列表，可以是 Colormap 对象、LmCmap 对象或 colormap 名称。
        """
        for i in range(len(cmaps)):
            if isinstance(cmaps[i], str):
                cmaps[i] = matplotlib.colormaps.get_cmap(cmaps[i])
            if isinstance(cmaps[i], LmCmap):
                cmaps[i] = cmaps[i].cmap()

        fig = plt.figure(figsize=(6, 0.5*len(cmaps)))
        fig.subplots_adjust(left=0.2, right=0.95, bottom=0.08, top=0.92)

        for i in range(len(cmaps)):
            ax = fig.add_subplot(len(cmaps), 1, i+1)
            plt.colorbar(cm.ScalarMappable(cmap=cmaps[i]), 
                cax=ax, orientation='horizontal', ticks=[])
            ax.text(-0.01,0.5, cmaps[i].name, ha='right', va='center', fontsize=10)
        plt.show()

if __name__ == '__main__':
    # 画出所有自带的 cmap 并保存
    LmCmap.draw_all_cmap("py/limeteo/doc/colormaps.png")
