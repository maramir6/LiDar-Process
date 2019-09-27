from lidar import LiDar


if __name__ == '__main__':
	
	file = 'example/sample.las'
	output_file = 'sample'

	lidar = LiDar(file, output_file)
	lidar.normalization()