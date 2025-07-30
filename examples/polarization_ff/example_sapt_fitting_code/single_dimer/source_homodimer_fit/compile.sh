#!/bin/zsh

#OPT="-check bounds -check uninit -check format -traceback "
OPT=""

#ifort $OPT nrtype.f90 nrutil.f90 nr.f90 variables.f90 routines.f90 interface.f90 energy_routines.f90 eq_drude.f90  kaisq_fitting_functions.f90 frprmn.f90 main_fitting_program.f90 -o fitenergy
mpifort $OPT nrtype.f90 nrutil.f90 nr.f90 variables.f90 routines.f90 interface.f90 energy_routines.f90 eq_drude.f90  kaisq_fitting_functions.f90 frprmn.f90 main_fitting_program.f90 -o fitenergy
