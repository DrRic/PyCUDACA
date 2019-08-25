#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PyCUDA001.py
#  
#  Copyright 2019 Dr Ricardo Colasanti <ric@Jetson>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import pycuda.driver as drv

def main(args):
	drv.init()
	for i in range(drv.Device.count()):
		dev = drv.Device(i)
		print(dev.name(),dev.compute_capability(),dev.total_memory()//(1024**2))
		dev_attrib = dev.get_attributes().items()
		compute_capability = float( '%d.%d' % dev.compute_capability() )
		dev_dict = {}
		for k,v in dev_attrib:
			dev_dict[str(k)] = v 
		num_mp = dev_dict['MULTIPROCESSOR_COUNT']
    
		# Cores per multiprocessor is not reported by the GPU!  
		# We must use a lookup table based on compute capability.
		# See the following:
		# http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

		cuda_cores_per_mp = { 5.0 : 128, 5.1 : 128, 5.2 : 128, 6.0 : 64, 6.1 : 128, 6.2 : 128}[compute_capability]
		print('\t ({}) Multiprocessors, ({}) CUDA Cores / Multiprocessor: {} CUDA Cores'.format(num_mp, cuda_cores_per_mp, num_mp*cuda_cores_per_mp))
		
		
		for k in dev_dict.keys():
			print(k,dev_dict[k])
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
