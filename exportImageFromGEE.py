import ee
import sys
import os
import importlib
import geopandas as gpd
import json
import logging
import numpy as np
import random
import rasterio
from rasterio.warp import transform_bounds

import utils.gee_utils as gee_utils

ee.Authenticate()
ee.Initialize(project = 'ee-wangyanrs') #ee-jiashuang0207/ ee-wangyanrs / muqiurs

# region =======变量声明========================================================================
# 0.25°的allGrids共包含16186个  1°的共包含1121个
allGrids_0d25_CHN = ee.FeatureCollection("projects/ee-wangyanrs/assets/EBIA_CSM/CHN_grids_1d")

export_folder ='CHN_percentiles_'
grid_count = 1121
strat_grid = 0 # [0,112,224,336,448,560,672,784,896,1008] 
exporting_grids_num = 112
buffered_meter = 30 * 40
test_tile_id = [] # [924,242,643,638,675,924,255,181,51,357]
cloud_pct = 20

if strat_grid == 1008:
    exporting_grids_num = 113
target_years = np.full(exporting_grids_num,1992) 
# or generate random year
# candidates = np.arange(1990, 2021, 5)  # 1990, 1995, ..., 2020
# target_years = np.random.choice(candidates, size = grid_count, replace=True)

# endregion


# region =======函数定义========================================================================
def imgComposite_by_geom(roi,buffer_meter=1000):
    if buffer_meter > 0:
        roi = roi.geometry().buffer(buffer_meter).bounds()
    else:  
       roi = roi.geometry()
    ls_compositedImg = gee_utils.get_landsat_collection(start_date, end_date,roi, cloud_pct = cloud_pct, harmonization = True) \
                   .reduce(ee.Reducer.percentile([20,50,80])) \
                   .clip(roi)
    return ls_compositedImg.toInt16()

def imgCol_by_geom(roi,buffer_meter=1000):
    if buffer_meter > 0:
        roi = roi.geometry().buffer(buffer_meter).bounds()
    else:
        roi = roi.geometry()
    ls_imgCol = gee_utils.get_landsat_collection(start_date, end_date,roi, cloud_pct = cloud_pct, harmonization = True)
    return ls_imgCol

def glcfcs30d_by_geom(glcfcs30,roi,buffer_meter=1000):
    if buffer_meter > 0:
       roi = roi.geometry().buffer(buffer_meter).bounds()
    else:
       roi = roi.geometry()
    return glcfcs30.filterBounds(roi).mosaic().clip(roi)


def collect_feature_image(year, roi, buffer_meter=1000):
    grid_name = roi.get('filename')
    
    if buffer_meter > 0:
        roi = roi.geometry().buffer(buffer_meter).bounds()
    else:
        roi = roi.geometry()
    
    # year = ee.String(grid_name).split('_').get(-1)
    # year = ee.Number.parse(year)
    # start_date = ee.Algorithms.If(year.lte(2000), ee.Date.fromYMD(year.subtract(2),1,1), ee.Date.fromYMD(year.subtract(1),7,1))
    # end_date = ee.Algorithms.If(year.lte(2000), ee.Date.fromYMD(year.add(3),1,1), ee.Date.fromYMD(year.add(1),7,1))
    # log_info = ee.String('collecting images for ').cat(ee.String(grid_name))
    if year < 2000:
        # start_date = str(year-2) + '-01-01'
        # end_date = str(year+3) + '-01-01'
        start_date = str(year-1) + '-01-01'
        end_date = str(year+3) + '-01-01'
    elif year == 2000:
        start_date = str(year-1) + '-01-01'
        end_date = str(year+2) + '-01-01'
    else:
        # start_date = str(year-1) + '-07-01'
        # end_date = str(year+1) + '-07-01'
        start_date = str(year) + '-01-01'
        end_date = str(year+2) + '-01-01'
    log_info= f'collecting images from {start_date} to {end_date} of Landsat images'

    ls_imgcol = gee_utils.get_landsat_collection(start_date, end_date, roi, cloud_pct = cloud_pct)
    ls_compositedImg = ls_imgcol \
            .reduce(ee.Reducer.percentile([20, 50, 80])) \
            .clip(roi)\
            .round() \
            .toInt16()
    terrain = gee_utils.get_terrain(roi)
    imgcount = ls_imgcol\
            .select('blue')\
                .reduce(ee.Reducer.count())\
                    .clip(roi)\
                        .toInt16()\
                        .rename('obsCount')
    feature_img = ls_compositedImg.addBands(terrain).addBands(imgcount).toInt16()\
                                    .set('logInfo',log_info).set('grid_name',grid_name)
    return feature_img

def collect_seasonal_feature(year, roi,buffer_meter=1000):
    grid_name = roi.get('filename')
    if buffer_meter > 0:
        roi = roi.geometry().buffer(buffer_meter).bounds()
    else:
        roi = roi.geometry()

    ls_imgcol = gee_utils.get_landsat_collection(start_date, end_date,roi, cloud_pct = cloud_pct,harmonization = True)
    merged_img = []
    for i_season in range(4):
        season_name = ['spring','summer','fall','winter'][i_season]
        season_medainComposited = ls_imgcol.filter(ee.Filter.calendarRange(i_season*3+1, i_season*3+3, 'month'))\
                                    .median().clip(roi).toInt16()
        season_terrain = gee_utils.get_terrain(roi)
        season_count = ls_imgcol.filter(ee.Filter.calendarRange(i_season*3+1, i_season*3+3, 'month'))\
                            .select('blue')\
                                .reduce(ee.Reducer.count())\
                                    .clip(roi).toInt16()
        feature_img = season_medainComposited.addBands(season_count).toInt16()
        merged_img.append(feature_img)
    merged_img = ee.Image.cat(merged_img).addBands(gee_utils.get_terrain(roi)).toInt16()
    return merged_img

# 每个制图年都收集前后五年的单传感器图像集合并做分值合成
# 若year是1990到2010任意一年，则定义sensor= 'LT05',若year是2015到2020任意一年，则定义sensor='LC08'
def collect_feature_5year1sensor(year,roi,buffer_meter=1000):
    grid_name = roi.get('filename')
    if buffer_meter > 0:
        roi = roi.geometry().buffer(buffer_meter).bounds()
    else:
        roi = roi.geometry()

    if year <= 2010:
        sensor = 'LT05'
    if year >= 2015 and year <= 2020:
        sensor = 'LC08'
    if year >= 2025:
        sensor = 'LC09'
    
    start_date = str(year-2) + '-01-01'
    end_date = str(year+3) + '-01-01'
    log_info= f'collecting images from {start_date} to {end_date} of Sensor {sensor}'
    ls_imgcol = gee_utils.get_landsat_collection(start_date, end_date, roi, sensor = sensor, cloud_pct = cloud_pct)
    ls_compositedImg = ls_imgcol \
            .reduce(ee.Reducer.percentile([20,35,50,65,80])) \
            .clip(roi)\
            .toInt16()
    terrain = gee_utils.get_terrain(roi)
    imgcount = ls_imgcol\
            .select('blue')\
                .reduce(ee.Reducer.count())\
                    .clip(roi)\
                        .toInt16()\
                        .rename('obsCount')
    feature_img = ls_compositedImg.addBands(terrain).addBands(imgcount).toInt16()\
                                    .set('logInfo',log_info).set('grid_name',grid_name)
    return feature_img

def create_attribute(feature, index):
    return ee.Feature(feature).set('filename',ee.String('CHN_featureImage_tile_id_').cat(index.add(1).format('%05d')))

def createFeatureCollectionFromGeojson(geojson_file, start_grid_index, grid_count, exporting_years):
    """
    读取本地GeoJSON文件，逐个读取Polygon，转换成ee.Feature，合并成ee.FeatureCollection返回。

    Args:
        geojson_path (str): 本地GeoJSON文件路径
        start_grid_index (int): 开始的格网索引，从0开始
        grid_count (int): 想要提取格网用来生成FeatureCollection的数量
        exporting_years (np.array): 对应每一个fc文件内的要素，制定该要素位置要收集的图像年份

    Returns:
        ee.FeatureCollection
        size of featurecollection
    """

    with open(geojson_file, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    features = []
    for idx, feature in enumerate(geojson['features']):
        targete_year = target_years[idx]
        # 转换几何为ee.Geometry
        geom = ee.Geometry(feature['geometry'])
        # 用属性创建ee.Feature
        props = feature.get('properties', {})
        tile_id = int(os.path.splitext(props['filename'])[0].split('_')[2])
        props['tile_id'] = tile_id
        props['filename'] = f'tile_id_{tile_id}_{str(targete_year)}'
        ee_feature = ee.Feature(geom, props)
        features.append(ee_feature)

    # 合并成FeatureCollection
    fc = ee.FeatureCollection(features[start_grid_index : start_grid_index + grid_count]) #features[0:2]
    return fc

# endregion

# region =======main function==================================================================
if __name__ == "__main__":
    #====================如果是收集中国地区1d tile内的特征图像========================#
    if strat_grid == 1008:
        exporting_grids_num = 113
    allGrids_0d25_CHN_list = allGrids_0d25_CHN.toList(1121)
    allGrids_0d25_CHN = allGrids_0d25_CHN_list.map(lambda f: create_attribute(f,allGrids_0d25_CHN_list.indexOf(f)))
    exporting_grids = ee.FeatureCollection(allGrids_0d25_CHN.slice(strat_grid, strat_grid+exporting_grids_num))
    # 1. export Landsat image of each grid as the feature image
    # imgList = exporting_grids.map(lambda f: collect_feature_image(2020, f, 30*40)).toList(1)
    # print(imgList.size().getInfo())

    # export images in the list
    exporting_grids_list = exporting_grids.toList(exporting_grids_num)
    for i in range(0, exporting_grids_num, 1):
        if i + strat_grid + 1 in test_tile_id:
            # print(f'skip test tile_id: {i + strat_grid + 1}')
            continue
        grid = ee.Feature(exporting_grids_list.get(i))
        year = target_years[i]
        filename = str(year) + '_' + grid.get('filename').getInfo()
        img = collect_feature_image(year, grid, buffered_meter)
        # img = ee.Image(imgList.get(i))
        log_info = img.get('logInfo').getInfo()
        grid_name = img.get('grid_name').getInfo()
        img_geometry = img.geometry()
        # print(img_geometry.getInfo())
        task = ee.batch.Export.image.toDrive(
            image =img,
            description = filename,#'featureImage_'+str(targete_year)+'_'+"{:05d}".format(i+1+strat_grid),
            folder = export_folder + str(year),
            fileNamePrefix=filename,#'featureImage_'+str(targete_year)+'_'+"{:05d}".format(i+1+strat_grid),
            region =img_geometry,
            scale =30,
            maxPixels =1e13
        )
        task.start()
        print(f'{log_info} for tile {grid_name}')


    #====================如果是收集训练样本位置的特征图像========================#
    # ----------------可以根据矢量文件的每个polygon作为roi-----------------------
    # fc_file = r'G:\EBIA-CSM_DATA\builtup_from_products\3-builtup_mask\1000grids\label_grids.geojson'
    # exporting_grids = createFeatureCollectionFromGeojson(fc_file,strat_grid,exporting_grids_num,target_years).toList(exporting_grids_num)
    # arr = list(range(1028))    # 0到1028的列表
    # random.shuffle(arr)        # 原地随机打乱
    # for i in arr[0:129]:# range(0, exporting_grids_num, 1):
    #     exporting_grid = ee.Feature(exporting_grids.get(i))
    #     target_year = target_years[i]
    #     img = collect_feature_image(target_year, exporting_grid, buffered_meter)
    #     filename = exporting_grid.get('filename').getInfo()
    #     log_info = img.get('logInfo').getInfo()
    #     img_geometry = img.geometry()
    #     task = ee.batch.Export.image.toDrive(
    #         image =img,
    #         description = filename,#'featureImage_'+str(targete_year)+'_'+"{:05d}".format(i+1+strat_grid),
    #         folder = export_folder+str(target_year),
    #         fileNamePrefix=filename,#'featureImage_'+str(targete_year)+'_'+"{:05d}".format(i+1+strat_grid),
    #         region =img_geometry,
    #         scale =30,
    #         maxPixels =1e13
    #     )
    #     task.start()
    #     print(f'{log_info} for tile {filename}')

    # ----------------还可以参考已经有标签tif文件作为roi-----------------------
    # ref_folder = r'G:\EBIA-CSM_DATA\Dataset_for_Unet\CSM_dataset\1st_addTiles_backgroundClass\labelImage'
    # ref_files = sorted([f for f in os.listdir(ref_folder) if f.lower().endswith('.tif')])
    # for i in range(0,len(ref_files)): #len(ref_files)
    #     ref_file = ref_files[i]
    #     ref_path = os.path.join(ref_folder, ref_file)
    #     try:
    #         if target_years is not None:
    #             year = target_years[i]
    #         else:
    #             year = int(os.path.splitext(ref_file)[0].split('_')[-1])
    #         # years = np.random.choice(candidates, size = 2, replace=True)
    #         with rasterio.open(ref_path) as src:
    #             bounds = src.bounds
    #             bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)
    #             # 将rasterio的bounds转换为GEE的ee.Geometry.Rectangle
    #             roi = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
    #             roi = ee.Feature(roi, {"filename":ref_file})
    #             descriptions = ref_file.split('_')
    #             description = descriptions[0] + '_' + descriptions[1] + '_' + descriptions[2] + '_' + str(year) + '.tif'
    #             img = collect_feature_image(year, roi, buffered_meter)
    #             task = ee.batch.Export.image.toDrive(
    #                 image =img,
    #                 description = description,#'featureImage_'+str(targete_year)+'_'+"{:05d}".format(i+1+strat_grid),
    #                 folder = export_folder+str(year),
    #                 fileNamePrefix=description,#'featureImage_'+str(targete_year)+'_'+"{:05d}".format(i+1+strat_grid),
    #                 region =roi.geometry(),
    #                 scale =30,
    #                 maxPixels =1e13
    #             )
    #             task.start()
    #             print(f'exporting multi sensor composited image for training grid {ref_file} in {year}')
    #     except Exception:
    #         print(Exception)
    #         continue 



# endregion