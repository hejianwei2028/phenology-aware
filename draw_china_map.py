"""
顶刊级别中国地图绘制示例：
- 使用 Cartopy + Natural Earth 10m 数据
- 统一投影、配色、边界线宽
- 增加南海诸岛插图
- 输出高分辨率出版级图片
"""

from __future__ import annotations

import pathlib

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects


def _set_publication_style() -> None:
    """设置出版级绘图风格。"""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _add_graticules(ax: plt.Axes) -> None:
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.3,
        color="0.65",
        alpha=0.7,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}


def draw_china_map(output_path: pathlib.Path) -> None:
    _set_publication_style()

    projection = ccrs.LambertConformal(
        central_longitude=105, central_latitude=35, standard_parallels=(25, 47)
    )
    fig = plt.figure(figsize=(6.4, 5.4))
    ax = plt.axes(projection=projection)

    # 聚焦中国区域
    ax.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())

    # 高分辨率自然地理要素
    land = cfeature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="none", facecolor="#f4f1ea"
    )
    ocean = cfeature.NaturalEarthFeature(
        "physical", "ocean", "10m", edgecolor="none", facecolor="#dbe8f4"
    )
    lakes = cfeature.NaturalEarthFeature(
        "physical", "lakes", "10m", edgecolor="#7aa6d9", facecolor="#a6c8e5"
    )
    borders = cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", "10m", edgecolor="#4a4a4a", facecolor="none"
    )
    coastline = cfeature.NaturalEarthFeature(
        "physical", "coastline", "10m", edgecolor="#2f2f2f", facecolor="none"
    )

    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)
    ax.add_feature(lakes, zorder=2)
    ax.add_feature(borders, linewidth=0.6, zorder=3)
    ax.add_feature(coastline, linewidth=0.6, zorder=4)

    _add_graticules(ax)

    title = ax.set_title("中国地图（出版级示例）", pad=12)
    title.set_path_effects(
        [patheffects.withStroke(linewidth=2.0, foreground="white")]
    )

    # 南海诸岛插图
    inset_ax = fig.add_axes([0.69, 0.12, 0.25, 0.25], projection=ccrs.PlateCarree())
    inset_ax.set_extent([105, 125, 3, 25])
    inset_ax.add_feature(ocean, zorder=0)
    inset_ax.add_feature(land, zorder=1)
    inset_ax.add_feature(coastline, linewidth=0.4, zorder=2)
    inset_ax.add_feature(borders, linewidth=0.4, zorder=3)
    inset_ax.outline_patch.set_linewidth(0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    draw_china_map(pathlib.Path("outputs/china_map.png"))
    print("Saved to outputs/china_map.png")
