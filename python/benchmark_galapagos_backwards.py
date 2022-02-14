from parcels import FieldSet, JITParticle, ScipyParticle, AdvectionRK4, Variable, StateCode, OperationCode, ErrorCode
from parcels import BenchmarkParticleSetSOA, BenchmarkParticleSetAOS, BenchmarkParticleSetNodes
from parcels import ParticleSetSOA, ParticleSetAOS, ParticleSetNodes
from parcels import GenerateID_Service, SequentialIdGenerator, LibraryRegisterC  # noqa

from datetime import timedelta as delta
from glob import glob
import numpy as np
# import itertools
# import matplotlib.pyplot as plt
import xarray as xr
import warnings
import math
import sys
import os
import gc
from argparse import ArgumentParser
import fnmatch
import time as ostime
# import dask
warnings.simplefilter("ignore", category=xr.SerializationWarning)

try:
    from mpi4py import MPI
except:
    MPI = None

# pset = None
pset_modes = ['soa', 'aos', 'nodes']
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
pset_types_dry = {'soa': {'pset': ParticleSetSOA},  # , 'pfile': ParticleFileSOA, 'kernel': KernelSOA
                  'aos': {'pset': ParticleSetAOS},  # , 'pfile': ParticleFileAOS, 'kernel': KernelAOS
                  'nodes': {'pset': ParticleSetNodes}}  # , 'pfile': ParticleFileNodes, 'kernel': KernelNodes
pset_types = {'soa': {'pset': BenchmarkParticleSetSOA},
              'aos': {'pset': BenchmarkParticleSetAOS},
              'nodes': {'pset': BenchmarkParticleSetNodes}}


def create_galapagos_fieldset(datahead, basefile_str, stokeshead, stokes_variables, stokesfile_str, meshfile,
                              periodic_wrap, period, chunk_level=0, use_stokes=False):
    files = None
    data_is_dict = False
    if type(basefile_str) == dict:
        files = {'U': sorted(glob(os.path.join(datahead, basefile_str['U']))),
                 'V': sorted(glob(os.path.join(datahead, basefile_str['V'])))}
        data_is_dict = True
    else:
        files = sorted(glob(os.path.join(datahead, basefile_str)))
    # if meshfile is None and not data_is_dict:
    #     meshfile = files
    mesh_is_dict = type(meshfile) == dict
    # ddir = os.path.join(datahead,"NEMO-MEDUSA/ORCA0083-N006/")
    # ufiles = sorted(glob(ddir+'means/ORCA0083-N06_20[00-10]*d05U.nc'))
    # vfiles = [u.replace('05U.nc', '05V.nc') for u in ufiles]
    # meshfile = glob(ddir+'domain/coordinates.nc')
    nemo_files = None
    if meshfile is None:
        nemo_files = {'U': files['U'] if data_is_dict else files,
                      'V': files['V'] if data_is_dict else files}
    else:
        nemo_files = {'U': {'lon': meshfile['U'] if mesh_is_dict else meshfile,
                            'lat': meshfile['U'] if mesh_is_dict else meshfile,
                            'data': files['U'] if data_is_dict else files},
                      'V': {'lon': meshfile['V'] if mesh_is_dict else meshfile,
                            'lat': meshfile['V'] if mesh_is_dict else meshfile,
                            'data': files['V'] if data_is_dict else files}}
    nemo_variables = {'U': 'uo', 'V': 'vo'}
    nemo_dimensions = None
    if (type(basefile_str) == dict and "ORCA" in basefile_str["U"]) or (type(basefile_str) == str and "ORCA" in basefile_str):
        nemo_dimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
    elif (type(basefile_str) == dict and "SMOC" in basefile_str["U"]) or (type(basefile_str) == str and "SMOC" in basefile_str):
        nemo_dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    # period = delta(days=366*11) if periodic_wrap else False  # 10 years period
    extrapolation = False if periodic_wrap else True
    # ==== Because the stokes data is a different grid, we actually need to define the chunking ==== #
    # fieldset_nemo = FieldSet.from_nemo(nemofiles, nemovariables, nemodimensions, field_chunksize='auto')
    nemo_chs = None
    nemo_nchs = None
    if chunk_level > 1:
        if (type(basefile_str) == dict and "ORCA" in basefile_str["U"]) or (type(basefile_str) == str and "ORCA" in basefile_str):
            nemo_chs = {'time_counter': 1, 'depthu': 75, 'depthv': 75, 'depthw': 75, 'deptht': 75, 'y': 100, 'x': 100}
            nemo_nchs = {
                'U': {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('depthu', 25), 'time': ('time_counter', 1)},  #
                'V': {'lon': ('x', 128), 'lat': ('y', 96), 'depth': ('depthv', 25), 'time': ('time_counter', 1)},  #
            }
        elif (type(basefile_str) == dict and "SMOC" in basefile_str["U"]) or (type(basefile_str) == str and "SMOC" in basefile_str):
            nemo_chs = {'time': 1, 'depth': 75, 'latitude': 100, 'longitude': 100}
            nemo_nchs = {
                'U': {'lon': ('longitude', 128), 'lat': ('latitude', 96), 'depth': ('depth', 25), 'time': ('time', 1)},  #
                'V': {'lon': ('longitude', 128), 'lat': ('latitude', 96), 'depth': ('depth', 25), 'time': ('time', 1)},  #
            }
    elif chunk_level > 0:
        nemo_chs = 'auto'
        nemo_nchs = {
            'U': 'auto',
            'V': 'auto'
        }
    else:
        nemo_chs = False
        nemo_nchs = False

    try:
        if meshfile is None:
            fieldset_nemo = FieldSet.from_netcdf(nemo_files, nemo_variables, nemo_dimensions, time_periodic=period, allow_time_extrapolation=extrapolation, field_chunksize=nemo_chs)
        else:
            fieldset_nemo = FieldSet.from_nemo(nemo_files, nemo_variables, nemo_dimensions, time_periodic=period, allow_time_extrapolation=extrapolation, field_chunksize=nemo_chs)
    except (SyntaxError, ):
        if meshfile is None:
            fieldset_nemo = FieldSet.from_netcdf(nemo_files, nemo_variables, nemo_dimensions, time_periodic=period, allow_time_extrapolation=extrapolation, chunksize=nemo_nchs)
        else:
            fieldset_nemo = FieldSet.from_nemo(nemo_files, nemo_variables, nemo_dimensions, time_periodic=period, allow_time_extrapolation=extrapolation, chunksize=nemo_nchs)

    fU = None
    fV = None
    if use_stokes:
        stokes_files = None
        if type(stokesfile_str) == dict:
            stokes_files = {'U': sorted(glob(os.path.join(stokeshead, stokesfile_str['U']))),
                            'V': sorted(glob(os.path.join(stokeshead, stokesfile_str['V'])))}
        else:
            stokes_files = sorted(glob(os.path.join(stokeshead, stokesfile_str)))
        # stokes_files = sorted(glob(os.path.join(stokeshead, "WW3-*_20[00-10]??_uss.nc")))
        # stokes_variables = {'U': 'uuss', 'V': 'vuss'}
        stokes_dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
        stokes_chs = None
        stokes_nchs = None
        if chunk_level > 1:
            if (type(basefile_str) == dict and "ORCA" in basefile_str["U"]) or (type(basefile_str) == str and "ORCA" in basefile_str):
                stokes_chs = {'time': 1, 'latitude': 16, 'longitude': 32}
                stokes_nchs = {
                    'U': {'lon': ('longitude', 32), 'lat': ('latitude', 16), 'time': ('time', 1)},
                    'V': {'lon': ('longitude', 32), 'lat': ('latitude', 16), 'time': ('time', 1)}
                }
            elif (type(basefile_str) == dict and "SMOC" in basefile_str["U"]) or (type(basefile_str) == str and "SMOC" in basefile_str):
                stokes_chs = nemo_chs
                stokes_nchs = nemo_nchs
        elif chunk_level > 0:
            stokes_chs = 'auto'
            stokes_nchs = {
                'U': 'auto',
                'V': 'auto'
            }
        else:
            stokes_chs = False
            stokes_nchs = False

        # stokes_period = delta(days=366+2*31) if periodic_wrap else False  # 14 month period
        # stokes_period = delta(days=366*11) if periodic_wrap else False  # 10 years period
        stokes_period = period if periodic_wrap else False  # 10 years period
        fieldset_stokes = None
        try:
            fieldset_stokes = FieldSet.from_netcdf(stokes_files, stokes_variables, stokes_dimensions, field_chunksize=stokes_chs, time_periodic=stokes_period, allow_time_extrapolation=extrapolation)
        except (SyntaxError, ):
            fieldset_stokes = FieldSet.from_netcdf(stokes_files, stokes_variables, stokes_dimensions, chunksize=stokes_nchs, time_periodic=stokes_period, allow_time_extrapolation=extrapolation)
        fieldset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)

        fieldset = FieldSet(U=fieldset_nemo.U+fieldset_stokes.U, V=fieldset_nemo.V+fieldset_stokes.V)
        fU = fieldset.U[0]
        fV = fieldset.V[0]
    else:
        fieldset = fieldset_nemo
        fU = fieldset.U
        fV = fieldset.V

    return fieldset, fU, fV


def perIterGC():
    gc.collect()


class GalapagosParticle(JITParticle):
    age = Variable('age', initial=0.)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=14.0*86400.0)  # np.finfo(np.float64).max


def Age(particle, fieldset, time):
    if particle.state == StateCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > particle.life_expectancy:
        particle.delete()


def DeleteParticle(particle, fieldset, time):
    particle.delete()


def periodicBC(particle, fieldSet, time):
    dlon = -89.0 + 91.8
    dlat = 0.7 + 1.4
    if particle.lon > -89.0:
        particle.lon -= dlon
    if particle.lon < -91.8:
        particle.lon += dlon
    if particle.lat > 0.7:
        particle.lat -= dlat
    if particle.lat < -1.4:
        particle.lat += dlat


if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection around an idealised peninsula")
    parser.add_argument("-s", "--stokes", dest="stokes", action='store_true', default=False, help="use Stokes' field data")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="benchmark_galapagos.png", help="image file name of the plot")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-w", "--writeout", dest="write_out", action='store_true', default=False, help="write data in outfile")
    # parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=365, help="runtime in days (default: 365)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=str, default="1*366", help="runtime in days (default: 1*365)")
    parser.add_argument("-tp", "--type", dest="pset_type", default="soa", help="particle set type = [SOA, AOS, Nodes]")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    parser.add_argument("-chs", "--chunksize", dest="chs", type=int, default=0, help="defines the chunksize level: 0=None, 1='auto', 2=fine tuned; default: 0")
    parser.add_argument("--dry", dest="dryrun", action="store_true", default=False, help="Start dry run (no benchmarking and its classes")
    args = parser.parse_args()

    pset_type = str(args.pset_type).lower()
    assert pset_type in pset_types
    ParticleSet = pset_types[pset_type]['pset']
    if args.dryrun:
        ParticleSet = pset_types_dry[pset_type]['pset']

    imageFileName=args.imageFileName
    time_in_days = int(float(eval(args.time_in_days)))
    time_in_years = int(float(time_in_days)/365.0)
    with_GC = args.useGC
    wstokes = args.stokes
    periodicFlag=args.periodic

    # ======================================================= #
    # new ID generator things
    # ======================================================= #
    idgen = None
    c_lib_register = None
    if pset_type == 'nodes':
        idgen = GenerateID_Service(SequentialIdGenerator)
        idgen.setDepthLimits(0., 1.0)
        idgen.setTimeLine(0, delta(days=time_in_days).total_seconds())
        c_lib_register = LibraryRegisterC()

    branch = "soa_benchmark"
    computer_env = "local/unspecified"
    scenario = "galapagos"
    headdir = ""
    odir = ""
    datahead = ""
    dirread_top = ""
    dirread_hydro = ""
    dirmesh = ""
    dirread_stokes = ""
    # dirread_mesh = ""
    basefile_str = None
    stokesfile_str = None
    stokes_variables = None
    period = None
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        headdir = "/scratch/{}/experiments/galapagos".format("ckehl")
        odir = os.path.join(headdir, "BENCHres", str(args.pset_type))
        datahead = "/data/oceanparcels/input_data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA', 'ORCA0083-N006')
        dirread_hydro = os.path.join(dirread_top, 'means')
        dirmesh = os.path.join(dirread_top, 'domain')
        dirread_stokes = os.path.join(datahead, 'WaveWatch3data', 'CFSR')
        computer_env = "Gemini"
        basefile_str = {
            'U': 'ORCA0083-N06_20[00-10]????d05U.nc',
            'V': 'ORCA0083-N06_20[00-10]????d05V.nc'
        }
        stokes_variables = {'U': 'uuss', 'V': 'vuss'}
        stokesfile_str = "WW3-*_2010??_uss.nc"
        period = delta(days=366*11)  # 10 years period
    elif os.uname()[1] in ["lorenz.science.uu.nl",] or fnmatch.fnmatchcase(os.uname()[1], "node*"):  # Lorenz
        CARTESIUS_SCRATCH_USERNAME = 'ckehl'
        headdir = "/storage/shared/oceanparcels/output_data/data_{}/experiments/galapagos".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "BENCHres", str(args.pset_type))
        datahead = "/storage/shared/oceanparcels/input_data/"
        dirread_top = os.path.join(datahead, 'CMEMS', 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024_SMOC')
        dirread_hydro = dirread_top
        dirmesh = dirread_top
        dirread_stokes = os.path.join(datahead, 'CMEMS', 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024_SMOC')
        # basefile_str = {
        #     'U': 'ORCA0083-N06_2004????d05U.nc',
        #     'V': 'ORCA0083-N06_2004????d05V.nc'
        # }
        basefile_str = "SMOC_2019*.nc"
        stokes_variables = {'U': 'vsdx', 'V': 'vsdy'}
        stokesfile_str = "SMOC_2019*.nc"
        period = delta(days=366)  # 1 years period
        computer_env = "Lorenz"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch/shared/{}/experiments/galapagos".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "BENCHres", str(args.pset_type))
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA', 'ORCA0083-N006')
        dirread_hydro = os.path.join(dirread_top, 'means')
        dirmesh = os.path.join(dirread_top, 'domain')
        dirread_stokes = os.path.join(datahead, 'WaveWatch3data', 'CFSR')
        basefile_str = {
            'U': 'ORCA0083-N06_20[00-10]????d05U.nc',
            'V': 'ORCA0083-N06_20[00-10]????d05V.nc'
        }
        stokes_variables = {'U': 'uuss', 'V': 'vuss'}
        stokesfile_str = "WW3-*_2010??_uss.nc"
        period = delta(days=366*11)  # 10 years period
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "int*.snellius.*") or fnmatch.fnmatchcase(os.uname()[1], "fcn*") or fnmatch.fnmatchcase(os.uname()[1], "tcn*") or fnmatch.fnmatchcase(os.uname()[1], "gcn*") or fnmatch.fnmatchcase(os.uname()[1], "hcn*"):  # Snellius
        SNELLIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch-shared/{}/experiments/galapagos".format(SNELLIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "BENCHres", str(args.pset_type))
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA', 'ORCA0083-N006')
        dirread_hydro = os.path.join(dirread_top, 'means')
        dirmesh = os.path.join(dirread_top, 'domain')
        dirread_stokes = os.path.join(datahead, 'WaveWatch3data', 'CFSR')
        basefile_str = {
            'U': 'ORCA0083-N06_20[00-10]????d05U.nc',
            'V': 'ORCA0083-N06_20[00-10]????d05V.nc'
        }
        stokes_variables = {'U': 'uuss', 'V': 'vuss'}
        stokesfile_str = "WW3-*_2010??_uss.nc"
        period = delta(days=366*11)  # 10 years period
        computer_env = "Snellius"
    else:
        headdir = "/var/scratch/galapagos"
        odir = os.path.join(headdir, "BENCHres", str(args.pset_type))
        datahead = "/data"
        dirread_top = os.path.join(datahead, 'NEMO-MEDUSA', 'ORCA0083-N006')
        dirread_hydro = os.path.join(dirread_top, 'means')
        dirmesh = os.path.join(dirread_top, 'domain')
        dirread_stokes = os.path.join(datahead, 'WaveWatch3data', 'CFSR')
        basefile_str = {
            'U': 'ORCA0083-N06_2000????d05U.nc',
            'V': 'ORCA0083-N06_2000????d05V.nc'
        }
        stokes_variables = {'U': 'uuss', 'V': 'vuss'}
        stokesfile_str = "WW3-*_2010??_uss.nc"
        period = delta(days=366)  # 1 years period

    print("running {} on {} (uname: {}) - branch '{}' - argv: {}".format(scenario, computer_env, os.uname()[1], branch, sys.argv[1:]))



    # ddir = os.path.join(datahead,"NEMO-MEDUSA/ORCA0083-N006/")
    # ufiles = sorted(glob(ddir+'means/ORCA0083-N06_20[00-10]*d05U.nc'))
    # vfiles = [u.replace('05U.nc', '05V.nc') for u in ufiles]
    # meshfile = glob(ddir+'domain/coordinates.nc')
    # create_galapagos_fieldset(datahead, basefile_str, stokeshead, stokes_variables, stokesfile_str, meshfile, periodic_wrap, period, chunk_level, use_stokes)

    wstokes &= True if dirread_stokes is not None else False
    meshfile = glob(os.path.join(dirread_top, 'domain', 'coordinates.nc'))
    meshfile = None if not len(meshfile)>0 else meshfile
    fieldset, fU, fV = create_galapagos_fieldset(dirread_hydro, basefile_str, dirread_stokes, stokes_variables, stokesfile_str, meshfile,
                                             True, period, chunk_level=args.chs, use_stokes=wstokes)

    # ======== ======== End of FieldSet construction ======== ======== #
    if os.path.sep in imageFileName:
        head_dir = os.path.dirname(imageFileName)
        if head_dir[0] == os.path.sep:
            odir = head_dir
        else:
            odir = os.path.join(odir, head_dir)
            imageFileName = os.path.split(imageFileName)[1]
    pfname, pfext = os.path.splitext(imageFileName)

    # fname = os.path.join(odir,"galapagosparticles_bwd_wstokes_v2.nc") if wstokes else os.path.join(odir,"galapagosparticles_bwd_v2.nc")
    outfile = "galapagosparticles_bwd_wstokes_v2.nc" if wstokes else "galapagosparticles_bwd_v2.nc"
    if MPI and (MPI.COMM_WORLD.Get_size()>1):
        outfile += "_MPI" + "_n{}".format(MPI.COMM_WORLD.Get_size())
    else:
        outfile += "_noMPI"
    if periodicFlag:
        outfile += '_p'
        pfname += '_p'
    if args.write_out:
        pfname += '_w'
    if time_in_years != 1:
        outfile += '_%dy' % (time_in_years, )
        pfname += '_%dy' % (time_in_years, )
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_size = mpi_comm.Get_size()
        outfile += '_n' + str(mpi_size)
        pfname += '_n' + str(mpi_size)
    if args.profiling:
        outfile += '_prof'
        pfname += 'prof'
    if with_GC:
        outfile += '_wGC'
        pfname += '_wGC'
    else:
        outfile += '_woGC'
        pfname += '_woGC'
    outfile += '_chs%d' % (args.chs)
    pfname += '_chs%d' % (args.chs)
    imageFileName = pfname + pfext

    dirwrite = odir
    if not os.path.exists(dirwrite):
        os.mkdir(dirwrite)

    galapagos_extent = [-91.8, -89, -1.4, 0.7]
    startlon, startlat = np.meshgrid(np.arange(galapagos_extent[0], galapagos_extent[1], 0.2),
                                     np.arange(galapagos_extent[2], galapagos_extent[3], 0.2))
    print("|lon| = {}; |lat| = {}".format(startlon.shape[0], startlat.shape[0]))

    pset = ParticleSet(fieldset=fieldset, pclass=GalapagosParticle, lon=startlon, lat=startlat, time=fU.grid.time[-1], repeatdt=delta(days=7), idgen=idgen, c_lib_register=c_lib_register)
    """ Kernel + Execution"""
    postProcessFuncs = None
    callbackdt = None
    if with_GC:
        postProcessFuncs = [perIterGC, ]
        callbackdt = delta(hours=12)

    output_fpath = None
    pfile = None
    if args.write_out and not args.dryrun:
        # output_fpath = fname
        output_fpath = os.path.join(dirwrite, outfile)
        pfile = pset.ParticleFile(name=output_fpath, outputdt=delta(days=1))
    kernel = pset.Kernel(AdvectionRK4)+pset.Kernel(Age)+pset.Kernel(periodicBC)

    starttime = 0
    endtime = 0
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            # starttime = MPI.Wtime()
            starttime = ostime.process_time()
    else:
        #starttime = ostime.time()
        starttime = ostime.process_time()

    pset.execute(kernel, dt=delta(hours=-1), output_file=pfile, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, postIterationCallbacks=postProcessFuncs, callbackdt=callbackdt)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            # endtime = MPI.Wtime()
            endtime = ostime.process_time()
    else:
        # endtime = ostime.time()
        endtime = ostime.process_time()

    if args.write_out and not args.dryrun:
        pfile.close()

    if not args.dryrun:
        size_Npart = len(pset.nparticle_log)
        Npart = pset.nparticle_log.get_param(size_Npart-1)
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            Npart = mpi_comm.reduce(Npart, op=MPI.SUM, root=0)
            if mpi_comm.Get_rank() == 0:
                if size_Npart>0:
                    sys.stdout.write("final # particles: {}\n".format( Npart ))
                sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime-starttime))
                avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
                sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time*1000.0))
        else:
            if size_Npart > 0:
                sys.stdout.write("final # particles: {}\n".format( Npart ))
            sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
            avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
            sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

        if MPI:
            mpi_comm = MPI.COMM_WORLD
            # mpi_comm.Barrier()
            Nparticles = mpi_comm.reduce(np.array(pset.nparticle_log.get_params()), op=MPI.SUM, root=0)
            Nmem = mpi_comm.reduce(np.array(pset.mem_log.get_params()), op=MPI.SUM, root=0)
            if mpi_comm.Get_rank() == 0:
                pset.plot_and_log(memory_used=Nmem, nparticles=Nparticles, target_N=1, imageFilePath=imageFileName, odir=odir)
        else:
            pset.plot_and_log(target_N=1, imageFilePath=imageFileName, odir=odir)

    del pset
    if idgen is not None:
        idgen.close()
        del idgen
    if c_lib_register is not None:
        c_lib_register.clear()
        del c_lib_register

    print('Execution finished')
