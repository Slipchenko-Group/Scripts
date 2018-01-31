import numpy as np
from sets import Set
import pickle


def cartesian_coord(fname,atom_num):
	cartesian = np.zeros((atom_num,3))
	if fname.endswith('.efp'):
		
		with open(fname,'r') as f:
			while True:
				line = f.readline()
				if not line:
					break
				if line.startswith(' COORDINATES (BOHR)'):
					for i in range(atom_num):
						line = f.readline()
						tokens = line.split()
						cartesian[i][0] = float(tokens[1])
						cartesian[i][1] = float(tokens[2])
						cartesian[i][2] = float(tokens[3])
					
	elif fname.endswith('.xyz'):
		with open(fname,'r') as f:
			line = f.readline()
			line = f.readline()
			for i in range(atom_num):
				line = f.readline()
				tokens = line.split()
				cartesian[i][0] = float(tokens[1])
				cartesian[i][1] = float(tokens[2])
				cartesian[i][2] = float(tokens[3])


	return cartesian

class Detect_bonds:
	def __init__(self,fname,atoms=None):
		if atoms == None:
			with open('./ref/ref','rb') as ref_data:
				metadata = pickle.load(ref_data)
			self.atom_num = metadata['atom_num']
			self.cartesian = cartesian_coord(fname,self.atom_num)
			self.mediums = metadata['mediums']
		else:
			self.atom_num = atoms
			self.cartesian = cartesian_coord(fname,self.atom_num)



	def get_cartesian(self):
		return self.cartesian

	def get_medium_coord(self,bonds):
		coord = []
		for i in range(len(self.mediums)):
			x = (self.cartesian[self.mediums[i][0]-1][0]+self.cartesian[self.mediums[i][1]-1][0])/2.0
			y = (self.cartesian[self.mediums[i][0]-1][1]+self.cartesian[self.mediums[i][1]-1][1])/2.0
			z = (self.cartesian[self.mediums[i][0]-1][2]+self.cartesian[self.mediums[i][1]-1][2])/2.0
			coord.append([format(x,' .10f'),format(y,' .10f'),format(z,' .10f')])
		return coord

	def calculate_bond_angles(self,a,b,c):
		ba = a-b
		bc = c-b
		cosine_angle = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
		angle = np.arccos(cosine_angle)
		#return angle
		return np.degrees(angle)

	def calculate_dihedral_angles(self,a,b,c,d):
		e0 = -1.0*(b-a)
		e1 = c-b
		e2 = d-c
		e1 /= np.linalg.norm(e1)

		v = e0 - np.dot(e0,e1)*e1
		w = e2 - np.dot(e2,e1)*e1
		x = np.dot(v,w)
		y = np.dot(np.cross(e1,v),w)
		#return np.arctan2(y,x)
		return np.degrees(np.arctan2(y,x))


	def all_lengths(self,bonds):

		length = []
		for i in range(self.atom_num):
			for j in range(i+1,self.atom_num):
				if bonds[i][j] == True:
					length.append(abs(np.linalg.norm(self.cartesian[i]-self.cartesian[j])))
		return length

	def all_bond_angles(self,bonds):

		angles = []
		for i in range(self.atom_num):
			for j in range(self.atom_num):
				if bonds[i][j] == True:
					for k in range(j+1,self.atom_num):
						if bonds[i][k] == True:
							angles.append(self.calculate_bond_angles(self.cartesian[j],
								self.cartesian[i],self.cartesian[k]))
		return angles

	def all_dihedral_angles(self,bonds):
		angles = []
		planes = []
		possibilities = []
		no_repeat = []
		for i in range(self.atom_num):
			for j in range(self.atom_num):
				if bonds[i][j] == True:
					for k in range(j+1,self.atom_num):
						if bonds[i][k] == True:
							planes.append([j,i,k])

		for i in range(len(planes)):
			for k in range(self.atom_num):
				if bonds[planes[i][2]][k] == True:
					if k not in planes[i]:
						possibilities.append([planes[i][0],planes[i][1],planes[i][2],k])
				if bonds[planes[i][0]][k] == True:
					if k not in planes[i]:
						possibilities.append([k,planes[i][0],planes[i][1],planes[i][2]])
						
		for i in range(len(possibilities)):
			item = possibilities[i]
			if item not in no_repeat and [item[3],item[2],item[1],item[0]] not in no_repeat:
				no_repeat.append(item)

		for i in range(len(no_repeat)):
			angles.append(self.calculate_dihedral_angles(self.cartesian[no_repeat[i][0]],
				self.cartesian[no_repeat[i][1]],self.cartesian[no_repeat[i][2]],
				self.cartesian[no_repeat[i][3]]))

		return angles







