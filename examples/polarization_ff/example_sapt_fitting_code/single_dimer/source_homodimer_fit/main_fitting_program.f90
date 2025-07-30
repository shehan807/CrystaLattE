!*************************************
! this program fits all individual energy components separately, to distinct force field terms
! Note that this program only works for a homomolecular dimer, but can be modified for a heteromolecular
! dimer without too much effort
!
! for electrostatics, exchange, induction, and dhf, Buckingham coefficients are fit
! for dispersion no parameters are fit, rather the energy is computed from the input Cn parameters
!
! the function minimizer used is the "frprmn" subroutine, which is a conjugate gradient algorithm from
! numerical recipes
!
! IMPORTANT:  ALL UNITS IN THIS CODE ARE ASSUMED TO BE IN ATOMIC UNITS
!**************************************

PROGRAM fit_energy_components
  use eq_drude
  USE variables
  USE routines
  CHARACTER(40)::data_file1,param_file,out_file 
  REAL*8,PARAMETER::ftol=1D-8
  INTEGER::i,j,k,iter,atoms1,atoms2,datpts,atms1,atms2,i_component
  REAL*8::fret,f0,f1,fdel,init
  REAL*8,DIMENSION(:),allocatable::df0,del,param,param1
  real*8,dimension(6)::rms
  real*8,dimension(3)::xyz

  INTERFACE
     SUBROUTINE frprmn(p,ftol,iter,fret)
       USE nrtype; USE nrutil, ONLY : nrerror
       USE nr, ONLY : linmin
       IMPLICIT NONE
       INTEGER(I4B), INTENT(OUT) :: iter
       REAL(SP), INTENT(IN) :: ftol
       REAL(SP), INTENT(OUT) :: fret
       REAL*8,DIMENSION(:), INTENT(INOUT) :: p
     END SUBROUTINE frprmn
     FUNCTION kaisq(p0)
       USE variables
       REAL*8::kaisq
       REAL*8,DIMENSION(:),INTENT(in)::p0
     END FUNCTION kaisq
     FUNCTION dkaisq(p0)
       USE variables
       REAL*8,DIMENSION(:),INTENT(in)::p0
       REAL*8,DIMENSION(SIZE(p0))::dkaisq
     END FUNCTION dkaisq
     SUBROUTINE  Efitout(rms)
       USE variables
       REAL*8,dimension(:),INTENT(inout)::rms
     END SUBROUTINE Efitout
     function E_component(type,datpt,init,atom1,atom2)
       use variables
       use interface
       real*8:: E_component
       integer,intent(in)::type,datpt,init
       integer,optional,intent(in)::atom1,atom2
     end function E_component
  END INTERFACE

 !********************************************
 !  inform about units
  write(*,*) ""
  write(*,*) "**************************************************************"
  write(*,*) " This code uses atomic units, and therefore all inputfile data"
  write(*,*) " should be in atomic units (except energy, which should be mH"
  write(*,*) "**************************************************************"
  write(*,*) ""

 !********************************************
 !  inform about minimization
  write(*,*) ""
  write(*,*) "**************************************************************"
  write(*,*) " If conjugate gradient minimization doesn't converge, try changing"
  write(*,*) " value of variable 'ftol' in main_fitting_program.f90 to a larger value"
  write(*,*) "**************************************************************"
  write(*,*) ""

 !********************************************
 ! inform about weight values
  write(*,*) ""
  write(*,*) "**************************************************************"
  write(*,*) " Parameters in weighting function are currently set to values of "
  write(*,'(A6,F8.5,A20,F8.5,A10)') " Mu = ", Eff_mu, " Hartree  and kT = ", Eff_kt , " Hartree "
  write(*,*) " please make sure these are appropriate for the current system!"
  write(*,*) " (change parameters Eff_mu and Eff_kt in variables.f90 file) "
  write(*,*) "**************************************************************"
  write(*,*) ""

  !*********************************************
  ! inform about Ecutoff
  write(*,*) "**************************************************************"
  write(*,*) " Energies will not be calculated for configurations with total energy"
  write(*,'(A15,F8.5,A20)') " greater than ", Ecutoff, " Hartree"
  write(*,*) " Modify Ecutoff parameter in variables.f90 to change this value"
  write(*,*) "**************************************************************"
  write(*,*) ""

 ! get input and output file names
  CALL getarg(1,data_file1); CALL getarg(2,param_file);CALL getarg(3,out_file)


  call initialize_data(data_file1,datpts,atoms1,atoms2)
  ! initialize the energy components to be fit,(value of init is meaningless)
  init = E_component(0,0,0) 

  ! get monomer properties, charges, exponents, drude oscillators, Cn coefficients, etc.
  call getparameters(param_file)

  !**************************** NOTE ********************************
  ! We need a mapping from the number of independent atomtypes (parameters) to the
  ! number of atoms in a molecule.  This mapping is accomplished through the use of the
  ! atomtypemon1 array which stores the atomtypes of every atom in the molecule (dimension atom1)
  ! and the atmtypemon1 array which stores only unique atom types (dimension atm1)
  !
  ! for instance, in H2, we have two identical hydrogen atoms, so atom1=2, and atomtypemon1 array
  ! will store both of these types.  But there is only one unique atom, and so atm1=1, and atmtypemon1
  ! will only store one "H"
  !******************************************************************

  ! create mapping for unique atom types
  call numindatoms(atms1,atms2)
  allocate( atmtypemon1(atms1),atmtypemon2(atms2) )
  call collapseatomtype



  shellcharge2=shellcharge1


  Select Case(drude_oscillators)
  Case("yes")
     call initialize_drude_positions(datpts)
  Case("no")
     nxyzshell2(:,:,:)=xyzmon2(:,:,:)  
     nxyzshell1(:,:,:)=xyzmon1(:,:,:)
  End Select


  !******************************************** induction first
  component_fit=3
  numparam=atms2
  allocate(df0(numparam),del(numparam),param(numparam),param1(numparam) )
  param = 1d0
  f0=kaisq(param)
  !call frprmn(param,ftol,iter,fret)


  !****** initialize high order dhf after induction penetration terms and oscillator postions have been initialized
  Select Case(drude_oscillators)
  Case("yes")
     call initialize_high_order_dhfenergy
  End Select



  !***************************************** elec
  component_fit=2
  numparam = atms2
  deallocate(df0,del,param,param1)
  allocate(df0(numparam),del(numparam),param(numparam),param1(numparam) )
  param = 1d0
  f0=kaisq(param)
  !call frprmn(param,ftol,iter,fret)


  !*******************************************dhf
  component_fit=4
  numparam = atms2
  deallocate(df0,del,param,param1)
  allocate(df0(numparam),del(numparam),param(numparam),param1(numparam) )
  param = 1d0
  f0=kaisq(param)
  !call frprmn(param,ftol,iter,fret)


  !************************************ exch
  component_fit=1
  numparam = atms2
  deallocate(df0,del,param,param1)
  allocate(df0(numparam),del(numparam),param(numparam),param1(numparam) )
  param = 1d0
  f0=kaisq(param)
  !call frprmn(param,ftol,iter,fret)


  Eout=0d0

  do i_component=1,6
     component_fit=i_component
     call Efitout(rms)
  enddo

  call writeoutput(out_file,rms)



end PROGRAM fit_energy_components
