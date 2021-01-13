#!/usr/bin/env python3
# convert_hipace_to_hipace++_file.py

import os
import sys
import math
import numpy as np
import re
import h5py
import openpmd_api as io
import argparse
from scipy import constants
def ps_parseargs():

    desc='This is the picpy postprocessing tool.'

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(  'path',
                          metavar = 'PATH',
                          nargs = '*',
                          help = 'Path to raw files.')
    parser.add_argument(  "--q_beam",
                          type=np.float64,
                          action='store',
                          dest="q_beam",
                          metavar="q_beam",
                          default=None,
                          help='total charge of the beam')
    parser.add_argument(  "--n0",
                          type=np.float64,
                          action='store',
                          dest="n0",
                          metavar="n0",
                          default=None,
                          help='plasma density to normalize to')
    return parser

# Dictionary for PIC codes
piccodes = { 'hipace':'hipace',
              'osiris':'osiris'
            }


class File:
    def __init__(self, file, intent, omit_hidden=True):
        self.__file = file

        if 'r' in intent:
            if not self.is_file():
                print('Error:\tFile "%s" does not exist!' % (self.get_file()) )
                sys.exit(1)

        if self.get_filename()[0] == '.' and self.get_filename()[1] != '/':
            if omit_hidden:
                print('Error:\tFile "%s" is hidden!' % (self.get_file()) )
                print('\tUse "omit_hidden=False" to read anyway!')
                sys.exit(1)

    def get_file(self):
        return self.__file

    def get_filename(self):
        return os.path.split(self.get_file())[1]

    def get_path(self):
        return os.path.split(self.get_file())[0]

    def is_file(self):
        return os.path.isfile(self.get_file())



# General HDF5 file class with routines
# to check whether file is an HDF5 file
# and to print keys of attributes and datasets
class H5File(File):
    def __init__(self, file, intent, omit_hidden=True):

        File.__init__(self, file, intent, omit_hidden=omit_hidden)

        # Allowed hdf5 extensions:
        self.__h5exts = ['.h5','.hdf5']

        if not self.is_h5_file():
            print('File "%s" is not an HDF5 file!' %(self.get_file()) )

    # Returning boolean: if file extension is hdf5 extension
    def is_h5_file(self, fext=None):
        if (fext!=None):
            return any(fext == h5ext for h5ext in self.__h5exts)
        else:
            fname, fext = os.path.splitext(self.get_file())
            return any(fext == h5ext for h5ext in self.__h5exts)

    def print_datasets(self):
        with h5py.File(self.get_file(),'r') as hf:
            # Printing attributes
            print('HDF5 file datasets:')
            for item in hf.keys():
                print('\t' + item + ":", hf[item])

    def print_attributes(self):
        with h5py.File(self.get_file(),'r') as hf:
            # Printing attributes
            print('HDF5 file attributes:')
            for item in hf.attrs.keys():
                print('\t' + item + ":", hf.attrs[item])

    def get_allowed_h5exts(self):
        return self.__h5exts


class H5PICFile(H5File):
    def __init__(self, file, h5ftype=None, omit_hidden=True):
        # initialize H5File with reading intent
        H5File.__init__(self, file, intent='r', omit_hidden=omit_hidden)

        # Grid 3D types in filenames:
        self.__g3dtypes = ['density', 'field', 'current']
        self.__g3dsubgrid_str = 'subgrid'
        # RAW types in filenames:
        self.__rawtypes = ['raw']
        self.__n_time_chars = 8

        self.__h5ftype = h5ftype

    # Returning boolean: if file name contains 'raw'
    def is_g3d_file(self, fname):
        return any((mq in fname) for mq in self.__g3dtypes)

    # Returning boolean: if file name contains 'raw'
    def is_raw_file(self, fname):
        return any((mq in fname) for mq in self.__rawtypes)

    # Returning boolean:  if file extension is hdf5 extension and
    #                     if file name contains name of grid quantity
    def is_h5g3d_file(self):
        fname, fext = os.path.splitext(self.get_file())
        return self.is_h5_file(fext=fext) and self.is_g3d_file(fname=fname)

    # Returning boolean:  if file extension is hdf5 extension and
    #                     if file name contains 'raw'
    def is_h5raw_file(self):
        fname, fext = os.path.splitext(self.get_file())
        return self.is_h5_file(fext=fext) and self.is_raw_file(fname=fname)

    def fcheck(self):
        if self.__h5ftype == 'g3d':
            return self.is_h5g3d_file()
        elif self.__h5ftype == 'raw':
            return self.is_h5raw_file()
        else:
            print('Error: No file type specified ["g3d", "raw"]!')
            sys.exit(1)

    def get_filename_time(self):
        name_w_time = os.path.splitext(self.get_filename())[0]
        stridx = [m.start() for m in re.finditer('_', name_w_time)][-1]
        return float(name_w_time[(stridx+1):])

    def get_filename_wo_time(self):
        name_w_time = os.path.splitext(self.get_filename())[0]
        stridx = [m.start() for m in re.finditer('_', name_w_time)][-1]
        name_wo_time = name_w_time[0:stridx]
        return name_wo_time

    def is_subgrid(self):
        fname = os.path.splitext(self.get_filename())[0]
        if (self.__g3dsubgrid_str in fname):
            return True
        else:
            return False


# Keys for HDF5 files
# Keys for PIC HDF5 files
class H5Keys:
    def __init__(self, piccode):

        # OSIRIS
        if piccode == piccodes['osiris']:

            # HDF5 GRID dataset keys
            self.__g3dkeys =  { 'density' : 'density'
                              }

            # HDF5 RAW dataset keys
            self.__rawkeys =  { 'x1':'x1',
                                'x2':'x2',
                                'x3':'x3',
                                'q':'q',
                                'p1':'p1',
                                'p2':'p2',
                                'p3':'p3'
                              }

            # HDF5 Attribute Keys
            self.__attrkeys = { 'nx':'NX',
                                'xmin':'XMIN',
                                'xmax':'XMAX',
                                'time':'TIME',
                                'dt':'DT',
                                'type':'TYPE',
                                'name':'NAME'
                              }
        # HiPACE
        elif piccode == piccodes['hipace']:

            # HDF5 GRID dataset keys
            self.__g3dkeys =  { 'beam_charge' : 'beam_charge',
                                'plasma_charge' : 'plasma_charge'
                              }
            # HDF5 GRID dataset types
            # Move these somewhere else...
            self.__g3dtypes = { 'density' : 'density',
                                'field' : 'field',
                                'current' : 'current'
                              }

            # HDF5 RAW dataset keys
            self.__rawkeys =  { 'x1':'x1',
                                'x2':'x2',
                                'x3':'x3',
                                'q':'q',
                                'p1':'p1',
                                'p2':'p2',
                                'p3':'p3',
                                'ipart' : 'ipart',
                                'iproc' : 'iproc'
                              }

            # HDF5 Attribute Keys
            self.__attrkeys = { 'nx':'NX',
                                'xmin':'XMIN',
                                'xmax':'XMAX',
                                'time':'TIME',
                                'dt':'DT',
                                'type':'TYPE',
                                'name':'NAME',
                                'nullcheck':'NULLCHECK'
                              }

    def get_g3dkey(self,key):
        return self.__g3dkeys[key]
    def get_g3dkeys(self):
        return self.__g3dkeys
    def print_g3dkeys(self):
        for key in self.__g3dkeys: print(key)
    def get_rawkey(self,key):
        return self.__rawkeys[key]
    def get_rawkeys(self):
        return self.__rawkeys
    def print_rawkeys(self):
        for key in self.__rawkeys: print(key)

    def get_attrkey(self, key):
        return self.__attrkeys[key]
    def get_attrkeys(self):
        return self.__attrkeys
    def print_attrkeys(self):
        for key in self.__attrkeys: print(key)

class HiFile(H5Keys, H5PICFile):
    def __init__(self, file):
        H5Keys.__init__(self, 'hipace')
        H5PICFile.__init__(self, file)
        self.__nullcheck = -1
        self.__nx = []
        self.__xmin = []
        self.__xmax = []
        self.__time = 0
        self.__dt = 0
        self.__name = ''
        # self.read_attrs()

    def read_attrs(self):
        # Reading attributes
        with h5py.File(self.get_file(),'r') as hf:

            if self.get_attrkey('nullcheck') in hf.attrs.keys():
                # If nullcheck exists, read value
                self.__nullcheck = hf.attrs[ self.get_attrkey('nullcheck') ]
            else:
                # If nullcheck doeas not exist, assume file is ok
                self.__nullcheck = 0

            self.__nx = hf.attrs[   self.get_attrkey('nx') ]
            self.__xmin = hf.attrs[ self.get_attrkey('xmin') ]
            self.__xmax = hf.attrs[ self.get_attrkey('xmax') ]
            self.__time = hf.attrs[ self.get_attrkey('time') ]
            self.__dt = hf.attrs[   self.get_attrkey('dt') ]
            type_bytes = hf.attrs[ self.get_attrkey('type') ]
            name_bytes = hf.attrs[ self.get_attrkey('name') ]

            if isinstance(type_bytes, str):
                self.__type = type_bytes
            else:
                self.__type = type_bytes[0].decode('UTF-8')
            if isinstance(name_bytes, str):
                self.__name = name_bytes
            else:
                self.__name = name_bytes[0].decode('UTF-8')

    def file_integrity_ok(self):
        if self.__nullcheck == 0:
            is_ok = True
        else:
            is_ok = False
            sys.stdout.write('Warning! File: %s is corrupted!\n' \
                    % self.get_filename())
            sys.stdout.flush()
        return is_ok

    def get_x_arr(self,dim):
        return np.linspace(self.__xmin[dim],self.__xmax[dim],self.__nx[dim])

    def get_z_arr(self):
        return np.linspace(self.__time+self.__xmin[0],self.__time+self.__xmax[0],self.__nx[0])

    def get_zeta_arr(self):
        return np.linspace(self.__xmin[0],self.__xmax[0],self.__nx[0])

    def get_xi_arr(self):
        return np.linspace(-self.__xmin[0],-self.__xmax[0],self.__nx[0])

    def get_nt(self):
        return round(self.__time/self.__dt)

    def get_time(self):
        return self.__time

    def get_nx(self,dim):
        return self.__nx[dim]

    def get_xmin(self,dim):
        return self.__xmin[dim]

    def get_xmax(self,dim):
        return self.__xmax[dim]

    def get_dx(self,dim):
        return (self.__xmax[dim]-self.__xmin[dim])/self.__nx[dim]

    def get_dt(self):
        return self.__dt

    def get_name(self):
        return self.__name

    def get_type(self):
        return self.__type

class H5FList():
    def __init__(self, paths, h5ftype=None):
        self.__paths = paths
        self.__h5ftype = h5ftype
        self.__flist = None

    def get(self, verbose=True, stride=1):
        if not self.__paths:
            print('Error: No file provided!')
            sys.exit(1)

        if isinstance(self.__paths, list):
            # if 'paths' is a list of directories
            list_of_flists = []
            for path in self.__paths:
                list_of_flists.append(self.__get_individ(path, verbose))
            flist = [item for sublist in list_of_flists for item in sublist]
        elif isinstance(self.__paths, str):
            # if 'paths' is a single directory
            flist = self.__get_individ(self.__paths, verbose)

        # Alphabetically sorting list
        self.__flist = sorted(flist)
        return self.__flist[0::stride]

    def __get_individ(self, path, verbose):
        flist = []
        if os.path.isfile(path):
            file = path
            h5f = H5PICFile(file, h5ftype=self.__h5ftype)
            if h5f.fcheck():
                flist.append(file)
            else:
                if verbose: print('Skipping: ' + file)
        elif os.path.isdir(path):
            if verbose: print('"' + path + '"' + ' is a directory.')
            if verbose:
                print('Processing all ' + self.__h5ftype +
                      ' files in the provided directory.')
            for root, dirs, files in os.walk(path):
                for filename in files:
                    file = root + '/' + filename
                    h5f = H5PICFile(file, h5ftype=self.__h5ftype)
                    if h5f.fcheck():
                        flist.append(file)
                    else:
                       if verbose: print('Skipping: ' + file)
        elif not os.path.exists(path):
            print('Error: Provided path "%s" does not exist!' % path)
            sys.exit()
        else:
            print('Error: Provided path "%s" is neither a file nor a directory!' % path)
            sys.exit()
        return flist

    def get_uniques(self, n_time_chars = 8):
        fnames = []
        if self.__flist == None:
            self.get()
        for f in self.__flist:
            h5f = H5PICFile(f)
            fnames.append(h5f.get_filename_wo_time())
        return list(set(fnames))

    def split_by_uniques(self, n_time_chars = 8):
        if self.__flist == None:
            self.get()
        uniques = self.get_uniques(n_time_chars=n_time_chars)

        # initialize and append to list of lists
        lofl = [[] for i in range(len(uniques))]
        for i in range(len(uniques)):
            for f in self.__flist:
                if uniques[i] in os.path.split(f)[1]:
                    lofl[i].append(f)
        return lofl

    def get_paths(self):
        return self.__paths

    def get_h5ftype(self):
        return self.__h5ftype


class HiRAW(HiFile):
    def __init__(self, file):
        HiFile.__init__(self, file)
        self.__npart = 0
        self.__data_is_read = False

        self.__coord2idx = {
            "x1": 0,
            "x2": 1,
            "x3": 2,
            "p1": 3,
            "p2": 4,
            "p3": 5,
            "q": 6,
            "iproc": 7,
            "ipart": 8 }

    def __save_coord(self,coord, nparray):
        self.__part[self.__coord2idx[coord],:] = nparray

    def get(self,coord=''):
        if self.__data_is_read:
            if coord=='':
                return self.__part
            else:
                return self.__part[self.__coord2idx[coord]]
        else:
            print('Error: Data not yet read!')
            sys.exit(1)

    def read_data(self, verbose=True):
        if verbose:
            print('Reading data: "%s" ...' % self.get_file())
        with h5py.File(self.get_file(),'r') as hf:

            ncoords = 7

            # Get number of particles
            dsetq = hf[self.get_rawkey('q')]
            npart = (dsetq.shape)[0]

            if self.get_rawkey('ipart') in hf.keys() \
                and self.get_rawkey('iproc') in hf.keys():
                ncoords += 2

            # Allocating array
            self.__part = np.zeros((ncoords, npart), dtype=np.float32)

            # Reading datasets
            keys = list(self.__coord2idx.keys())
            for i in range(0,ncoords):
                self.__save_coord(  keys[i],\
                                    np.array(hf.get(  self.get_rawkey(keys[i]) )))

            self.__npart = npart

        self.__data_is_read = True

        if verbose:
            print('Data is read.')

    def select_by_idx(self,idx):
        self.__part = self.__part[:,idx]

    def get_npart(self):
        self.__npart = (self.__part.shape)[1]
        return self.__npart

    def select_zeta_range(self, zeta_range, verbose=True):
        if not self.__data_is_read:
            self.read_data()
        if zeta_range != [] and len(zeta_range) == 2:
            idx = np.nonzero((self.get('x1') >= zeta_range[0]) & (self.get('x1') < zeta_range[1]))[0]
            self.__npart = np.size(idx)
            if verbose:
                print('%i particles in selected range [%0.2f, %0.2f]' \
                    % (self.__npart,zeta_range[0],zeta_range[1]))
            self.__part = self.__part[:,idx]


def main():

    parser = ps_parseargs()

    args = parser.parse_args()

    h5fl = H5FList(args.path, h5ftype='raw')
    flist = h5fl.get(verbose=False) #, stride=args.Nstride)
    if len(h5fl.get_uniques()) > 1:
        print('ERROR: Processing of multiple beams is not implemented yet!')
        print(h5fl.split_by_uniques())
        sys.exit(1)

    Nfiles = len(flist)

    if Nfiles < 1:
        print('No raw files selected!')
        print('Exiting...')
        sys.exit(1)

    # Getting file information
    raw = HiRAW(flist[0])
    # raw.read_attrs()



    sys.stdout.write('There are %i raw files to process...\n' % Nfiles)
    sys.stdout.flush()

    raw.read_data(verbose=True)

    x1 = raw.get('x1')
    x2 = raw.get('x2')
    x3 = raw.get('x3')
    p1 = raw.get('p1')
    p2 = raw.get('p2')
    p3 = raw.get('p3')
    q =  raw.get('q')

    plasma_wavenumber_in_per_meter = np.sqrt(args.n0 * (constants.e / constants.m_e) * \
                                     (constants.e / constants.epsilon_0) )/ constants.c

    if (args.q_beam):
        print("Renormalizing beam..")
        sum_of_weights = np.sum(q)
        q_SI = args.q_beam / sum_of_weights
    else:
        raw.read_attrs()
        q_SI = raw.get_dx(0) * raw.get_dx(1) * raw.get_dx(2) * constants.e * \
               args.n0 /(plasma_wavenumber_in_per_meter**3)

    series = io.Series("beam_%05T.h5", io.Access.create)

    i = series.iterations[0]

    particle = i.particles["Electrons"]

    i.set_attribute("Hipace++_Plasma_Density", args.n0)

    dataset = io.Dataset(x1.dtype,x1.shape)

    particle["r"].unit_dimension = {
        io.Unit_Dimension.L:  1,
    }

    particle["u"].unit_dimension = {
        io.Unit_Dimension.L:  1,
        io.Unit_Dimension.T: -1,
    }

    particle["q"].unit_dimension = {
        io.Unit_Dimension.I:  1,
        io.Unit_Dimension.T:  1,
    }

    particle["m"].unit_dimension = {
        io.Unit_Dimension.M:  1,
    }

    ### IMPORTANT NOTE: because HiPACE-C is C ordered and HiPACE++ is Fortran ordered
    ### the indices are switched!
    particle["r"]["x"].reset_dataset(dataset)
    particle["r"]["x"].store_chunk(x2)

    particle["r"]["y"].reset_dataset(dataset)
    particle["r"]["y"].store_chunk(x3)

    particle["r"]["z"].reset_dataset(dataset)
    particle["r"]["z"].store_chunk(x1)

    particle["u"]["x"].reset_dataset(dataset)
    particle["u"]["x"].store_chunk(p2)

    particle["u"]["y"].reset_dataset(dataset)
    particle["u"]["y"].store_chunk(p3)

    particle["u"]["z"].reset_dataset(dataset)
    particle["u"]["z"].store_chunk(p1)

    particle["q"]["q"].reset_dataset(dataset)
    particle["q"]["q"].store_chunk(q)

    particle["m"]["m"].reset_dataset(dataset)
    particle["m"]["m"].store_chunk(q)


    particle["r"]["x"].unit_SI = 1. / plasma_wavenumber_in_per_meter
    particle["r"]["y"].unit_SI = 1. / plasma_wavenumber_in_per_meter
    particle["r"]["z"].unit_SI = 1. / plasma_wavenumber_in_per_meter
    particle["u"]["x"].unit_SI = 1.
    particle["u"]["y"].unit_SI = 1.
    particle["u"]["z"].unit_SI = 1.
    particle["q"]["q"].unit_SI = q_SI
    particle["m"]["m"].unit_SI = q_SI * constants.m_e / constants.e

    series.flush()

    del series

    sys.stdout.write('Done!\n')
    sys.stdout.flush()

if __name__ == "__main__":
    main()
