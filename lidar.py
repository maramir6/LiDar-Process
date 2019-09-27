from laspy.file import File
from numba import jit
import numpy as np
import dask.dataframe as dd
from dask import delayed
from scipy.ndimage import label
from scipy import ndimage, stats
from skimage import measure, color
from skimage.measure import label, regionprops, find_contours, approximate_polygon, subdivide_polygon
from skimage.feature import canny, peak_local_max
from skimage.filters import threshold_otsu, rank
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.morphology import disk, square, dilation, remove_small_holes, remove_small_objects, convex_hull_object
from skimage.draw import circle_perimeter
from skimage.filters.rank import minimum, median
from skimage.util import pad
import os, time, cv2
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
import torch
from torch import nn
#from pykeops.torch import LazyTensor

class LiDar:

	def __init__(self, file, output_file, scale = 1):

		inFile = File(file, mode = "r")

		header_file = inFile.header

		self.x, self.y, self.z = inFile.x, inFile.y, inFile.z
		self.rn = inFile.return_num
		
		try:
			self.red, self.green, self.blue = inFile.red, inFile.green, inFile.blue
			print('LiDar file containing RGB channels')
		except:
			pass
		
		inFile.close()

		coords = np.vstack((self.z, self.y, self.x)).transpose()

		self.max_values = np.max(coords, axis=0).astype(int)
		self.min_values = np.min(coords, axis=0).astype(int)
		
		self.xr = np.array((1/scale)*(coords[:,2] - self.min_values[2]), dtype=np.int)
		self.yr = np.array((1/scale)*(self.max_values[1] - coords[:,1]), dtype=np.int)
		
		self.output_file = output_file
		self.scale = scale
		self.dimension = np.array((1/scale)*(self.max_values-self.min_values), dtype=np.int)
		self.n_points = len(self.xr)

		print('Metadata: ', header_file)
		print('Number of LiDar points: ', self.n_points)
		print('Maximum values in axis (z,y,x) are: ', self.max_values)
		print('Minimum values in axis (z,y,x) are: ', self.min_values)
		print('Cubic dimensions are: ', self.dimension)

	def density_first(self):

		start_time = time.time()

		index = np.where(self.rn==1)
		df = dd.from_array(np.stack((self.yr[index], self.xr[index], self.z[index]), axis=1), chunksize=int(self.n_points/10), columns=['y','x','z'])
		dz_values = df.groupby(['y','x']).agg(['min', 'max', 'count']).reset_index().values
		dz_values.compute()

		dz_values = np.asarray(dz_values).astype(int)

		z = np.full((int(3), int(self.dimension[1]+1), int(self.dimension[2]+1)), np.nan)
		z = self.height(dz_values, z)

		z_min = z[0,:,:]
		z_max = z[1,:,:]-z[0,:,:]
		z_den = z[2,:,:]

		file_name = self.output_file + '_DTM_first'
		self.rasterize(file_name, np.expand_dims(z_min, axis=0))
		
		file_name = self.output_file + '_CHM_first'
		self.rasterize(file_name, np.expand_dims(z_max, axis=0))

		file_name = self.output_file + '_DEN_first'
		self.rasterize(file_name, np.expand_dims(z_den, axis=0))

		print("Density Terrain Model, Canopy Height Model, Density Model and Boundary Polygon took: --- %s seconds ---" % (time.time() - start_time))


	def normalization(self):

		start_time = time.time()
		
		df = dd.from_array(np.stack((self.yr, self.xr, self.z), axis=1), chunksize=int(self.n_points/10), columns=['y','x','z'])
		dz_values = df.groupby(['y','x']).agg(['min', 'max', 'count']).reset_index().values
		dz_values.compute()

		dz_values = np.asarray(dz_values).astype(int)

		z = np.full((int(3), int(self.dimension[1]+1), int(self.dimension[2]+1)), np.nan)
		z = self.height(dz_values, z)

		z_min = z[0,:,:]
		z_max = z[1,:,:]-z[0,:,:]
		z_den = z[2,:,:]

		#z_min = self.filter_mean(z_min, 3)

		file_name = self.output_file + '_DTM'
		self.rasterize(file_name, np.expand_dims(z_min, axis=0))
		
		file_name = self.output_file + '_CHM'
		self.rasterize(file_name, np.expand_dims(z_max, axis=0))

		file_name = self.output_file + '_DEM'
		self.rasterize(file_name, np.expand_dims(z_den, axis=0))

		file_name = self.output_file
		self.polygon(file_name, z_min)

		print("Density Terrain Model, Canopy Height Model, Density Model and Boundary Polygon took: --- %s seconds ---" % (time.time() - start_time))

		key_kantor, values = self.kantor_encoder(dz_values[:,0], dz_values[:,1]), dz_values[:,2].astype(int)
		dictionary = dict(zip(key_kantor.tolist(), values.tolist()))
		data_kantor = self.kantor_encoder(self.yr, self.xr)

		self.zn = self.z - self.vec_translate(data_kantor, dictionary)

		print("Normalization process took: --- %s seconds ---" % (time.time() - start_time))

	def vec_translate(self, a, my_dict):
		return np.vectorize(my_dict.__getitem__)(a)

	def kantor_encoder(self, y, x):
		return 0.5*(y+x)*(y+x+1)+y

	def voxelize(self, scale, scale_z):
		
		start_time = time.time()
		self.scale = scale
		self.scale_z = scale_z
		
		coords = np.vstack((self.zn, self.y, self.x)).transpose()
		
		x = np.array((1/scale)*(coords[:,2] - self.min_values[2]), dtype=np.int)
		y = np.array((1/scale)*(self.max_values[1] - coords[:,1]), dtype=np.int)
		z = np.array((1/scale_z)*(coords[:,0] - self.min_values[0]), dtype=np.int)
		
		table = dd.from_array(np.stack((z, y, x), axis=1), columns=['z', 'y', 'x'])
		table = table.groupby(['z', 'y', 'x']).size().reset_index().rename(columns={0:'count'}).values
		table.compute()
		table = np.asarray(table).astype(int)

		self.voxel = np.zeros(self.dimension + 1)
		self.voxel = self.voxelization(table, self.voxel)
		file_name = self.output_file + '_VOX'
		
		self.rasterize(file_name,array)

		print("Voxelization process took: --- %s seconds ---" % (time.time() - start_time))
		print('Voxel raster saved as:', file_name)

	
	def KMeans(self, x, K=10, Niter=10, verbose=True):
		N, D = x.shape  # Number of samples, dimension of the ambient space

		start = time.time()
		c = x[:K, :].clone()
		x_i = LazyTensor(x[:, None, :])

		for i in range(Niter):
			c_j = LazyTensor(c[None, :, :])
			D_ij = ((x_i - c_j) ** 2).sum(-1)
			cl = D_ij.argmin(dim=1).long().view(-1)

			Ncl = torch.bincount(cl).type(torchtype[dtype])
			for d in range(D):
				c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

		end = time.time()

		if verbose:
			print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
			print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(Niter, end - start, Niter, (end-start) / Niter))

		return cl, c

	def rasterize(self, file_name, array, directory='', epsg_code=32718):

		transform = [self.min_values[2], self.scale, 0, self.max_values[1], 0, -self.scale]
		projection  = osr.SpatialReference()
		projection.ImportFromEPSG(epsg_code)
		driver = gdal.GetDriverByName('GTiff')

		raster = driver.Create(directory + file_name + '.tif', array.shape[2], array.shape[1], array.shape[0], gdal.GDT_Float32)
			
		for index_band in range(0, array.shape[0]):
			raster.GetRasterBand(index_band+1).WriteArray(array[index_band][:][:])
			raster.GetRasterBand(index_band+1).SetNoDataValue(np.nan)
			
		raster.SetGeoTransform(transform)
		raster.SetProjection(projection.ExportToWkt())
		raster.FlushCache()
		raster=None

	def pixel2coord(self, col, row):

		c, a, b, f, d, e = self.min_values[2], self.scale, 0, self.max_values[1], 0, -self.scale
		
		xp = a * col + b * row + a * 0.5 + b * 0.5 + c
		yp = d * col + e * row + d * 0.5 + e * 0.5 + f

		return(xp, yp)

	def polygon(self, file_name, img):

		img[img != np.nan] = 1
		img[img == np.nan] = 0
		img = pad(img.astype(int), pad_width=2, mode='constant')
		img = remove_small_holes(img, 100)
		contour_list = []
		
		for contour in find_contours(img, 0.5):
			contour_list.append(contour)

		coords = np.concatenate(contour_list)
		ring = ogr.Geometry(ogr.wkbLinearRing)
		
		for coord in coords:
			xp,yp = self.pixel2coord(coord[1],coord[0])
			ring.AddPoint(xp,yp)

		xp,yp = self.pixel2coord(coords[0][1],coords[0][0])
		ring.AddPoint(xp,yp)

		poly = ogr.Geometry(ogr.wkbPolygon)
		poly.AddGeometry(ring)

		driver = ogr.GetDriverByName('ESRI Shapefile')
		ds = driver.CreateDataSource(file_name + '.shp')
		layer = ds.CreateLayer('boundary', geom_type = ogr.wkbPolygon)

		idField = ogr.FieldDefn("id", ogr.OFTInteger)
		layer.CreateField(idField)
		
		defn = layer.GetLayerDefn()
		feat = ogr.Feature(defn)
		feat.SetGeometry(poly)
		feat.SetField('id',1)
		layer.CreateFeature(feat)
		feat = None

	def boundary(self, file_name, array):

		src_ds = gdal.Open(file_name + '_BDR.tif')
		srcband = src_ds.GetRasterBand(1)
		data = srcband.ReadAsArray()
		data = data > 0
		drv = ogr.GetDriverByName("ESRI Shapefile")
		dst_ds = drv.CreateDataSource( file_name + ".shp" )
		dst_layer = dst_ds.CreateLayer(file_name, srs = None )
		gdal.Polygonize(srcband, data, dst_layer, -1, [], callback=None )

	def height(self, data, array): 
		for i in range(data.shape[0]):
			array[:, data[i][0], data[i][1]] = data[i][2:] 

		return array

	def voxelization(self, data, array): 
		for i in range(data.shape[0]):
			array[data[i][0], data[i][1], data[i][2]] = data[i][3]

		return array

	def filter_mean(self, image, knn_px=3):

		tensor = torch.from_numpy(np.expand_dims(np.expand_dims(image, axis=0), axis=0)).double()
		per = 1/(knn_px**2 -1)
		masks = np.full((1,1,knn_px,knn_px), per)
		masks[0,0,int(0.5*(knn_px+1)),int(0.5*(knn_px+1))] = 0
		avg_layer = nn.Conv2d(1, 1, knn_px, padding=1)
		avg_layer.weight.data = torch.from_numpy(masks).double()
		avg_layer.bias.data = torch.from_numpy(np.zeros(1)).double()
		image_ = np.squeeze(avg_layer(tensor).detach().numpy())
		mask = np.where(image == -100, 1, 0)
		image_ = np.multiply(mask, image_).astype(int)
		
		return image + image_

if __name__ == '__main__':
	
	file = 'example/sample.las'
	output_file = 'sample'

	lidar = LiDar(file, output_file)
	lidar.normalization()
#	lidar.density_first()
