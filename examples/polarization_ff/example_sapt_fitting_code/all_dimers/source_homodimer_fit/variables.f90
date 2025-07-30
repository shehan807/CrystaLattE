!**************************************************
! these are global variables used by many subroutines throughout
! the program
!**************************************************

MODULE variables

 integer :: component_fit
integer, parameter :: iteration_limit=15
  real*8,parameter:: force_threshold = 4.0477D-6   ! this is for rms force in e^2/A^2, equal to .1 KJ/mol/nm
 real*8,parameter::Eff_mu=.14,Eff_kt=.001
 real*8,parameter::Ecutoff=0.2 ! this only applies to rms,not fit
 real*8,dimension(:,:),allocatable :: Eout
 REAL*8,DIMENSION(:),ALLOCATABLE::chargeatoms1,chargeatoms2,shellcharge1,shellcharge2
 real*8,dimension(:),allocatable::E1exch,E1pol,E2ind,E2indexch,E2disp,E2dispexch,dhf,Etot,Etot_noind,E1tot,E2tot
 real*8,dimension(:),allocatable::E2ind21,E2ind12,high_order_dhf_energy
 real*8,dimension(:,:),allocatable::Ecomp
 CHARACTER(2),DIMENSION(:),ALLOCATABLE::atomtypemon1,atomtypemon2,atmtypemon1,atmtypemon2
 REAL*8,DIMENSION(:,:,:),ALLOCATABLE::xyzmon1,xyzmon2,nxyzshell1,nxyzshell2
  real*8,dimension(:),allocatable::Elecatom2,Inducatom2,Exchatom2,dhf2
  real*8,dimension(:),allocatable::Elecatom1,Inducatom1,Exchatom1,dhf1
  character(2),dimension(:),allocatable::hard_cons_type
  REAL*8,DIMENSION(:,:),ALLOCATABLE::exponents,Exoverlap,Elecoverlap,Inducoverlap,dhfoverlap,hard_cons_param
 real*8,dimension(:,:,:),allocatable:: Cn_cross
 REAL*8::springcon1
 INTEGER::numparam
 REAL*8,PARAMETER::deltashell=1D-8,screenlength=2.0,small_coeff=1D-10

 character(3),parameter :: elec_screen="no"   ! electrostatic screening function
 character(3),parameter :: hard_constraints="yes"
 character(3),parameter :: drude_oscillators="yes"
 character(3),parameter :: include_high_order_drude_dhf_fit="yes"


 

contains

!***************************************************
! this is a weight function (fermi-dirac distribution)
! that weights the contribution of each configuration to the
! kaisq fitting function based on its total energy
!***************************************************
function weight(Energy,Eff_mu,Eff_kt,i)
real*8::weight
real*8,dimension(:),intent(in)::Energy
real*8,intent(in)::Eff_mu,Eff_kt
integer,intent(in)::i

weight=1./(exp((Energy(i)-Eff_mu)/Eff_kt)+1.)

end function weight




END MODULE variables
