!***********************************************
! this file contains the kaisq fitting function, as well as it's derivatives
!**********************************************


!********************************************
! kaisq fitting function
!********************************************
function kaisq(param)
  use variables
  use routines
  real*8::kaisq
  real*8,dimension(:),intent(in)::param
  interface
     function E_component(type,datpt,init,atom1,atom2)
       use variables
       use interface
       real*8:: E_component
       integer,intent(in)::type,datpt,init
       integer,optional,intent(in)::atom1,atom2
     end function E_component
  end interface

  integer::i,j,k,datpts,atoms1,atoms2
  real*8,dimension(size(Ecomp(:,1)),size(Ecomp(1,:)))::E_ff
  real*8::sum

  call outparam(param)

  atoms1=size(xyzmon1(1,:,1))
  atoms2=size(xyzmon2(1,:,1))
  datpts=size(xyzmon1(:,1,1))


  ! outparam fills Exchatom2,Elecatom2, etc arrays
  ! combinegeom fills Exoverlap,Elecoverlap arrays using combination rule

  Select Case(component_fit)
  Case(1)
     call combinegeom(Exchatom1,Exchatom2,Exoverlap)
  Case(2)
     call combinegeom(Elecatom1,Elecatom2,Elecoverlap)
  Case(3)
     call combinegeom(Inducatom1,Inducatom2,Inducoverlap)
  Case(4)
     call combinegeom(dhf1,dhf2,dhfoverlap)
  End Select

  ! fix sign of coefficients
  do i=1,atoms1
     do j=1,atoms2
        Exoverlap(i,j) = abs(Exoverlap(i,j))
        Elecoverlap(i,j) = abs(Elecoverlap(i,j))
        Inducoverlap(i,j) = abs(Inducoverlap(i,j))
        dhfoverlap(i,j) = abs(dhfoverlap(i,j))
     enddo
  enddo

  sum=0.0

  j = component_fit
  do i=1,datpts
     E_ff(j,i) = E_component(j,i,1)
     sum=sum+weight(Etot,Eff_mu,Eff_kt,i) * (E_ff(j,i)-Ecomp(j,i))**2
  enddo
  kaisq=sum 

end function kaisq





!********************************************
! kaisq fitting function derivatives
!********************************************
function dkaisq(param)
  use variables
  real*8,dimension(:),intent(in)::param
  real*8,dimension(size(param))::dkaisq
  interface
     function E_component(type,datpt,init,atom1,atom2)
       use variables
       use interface
       real*8:: E_component
       integer,intent(in)::type,datpt,init
       integer,optional,intent(in)::atom1,atom2
     end function E_component
     subroutine createdkaisq(dkaisq,dparam)
       use variables
       real*8,dimension(:),intent(in)::dparam
       real*8,dimension(:),intent(inout)::dkaisq
     end subroutine createdkaisq
  end interface

  real*8,dimension(size(Ecomp(:,1)),size(Ecomp(1,:)))::E_ff
  integer::i,j,k,l,datpts,atoms1,atoms2,sign
  real*8::term,fac,sum1,penalty=1D1
  real*8,parameter::delta=1D-8
  real*8,dimension(:),allocatable::dparam


  atoms1=size(xyzmon1(1,:,1))
  atoms2=size(xyzmon2(1,:,1))
  atms2=size(atmtypemon2)
  datpts=size(xyzmon1(:,1,1))

  allocate(dparam(atoms2))

  j = component_fit
  dkaisq=0d0
  dparam=0d0
  sum1=0d0

  ! take derivatives with respect to all atom parameters, then at the end map them to unique parameters.

  do i=1,datpts
     ! rms contribution from individual components
     E_ff(j,i) = E_component(j,i,1)
     fac = 2d0 * weight(Etot,Eff_mu,Eff_kt,i)*(E_ff(j,i)-Ecomp(j,i))


     ! loop over atomtypes for monomer 2
     ! note we don't need to include a factor of 2 for monomer 1, as that has been included
     ! in the derivative subroutines
     do k =1 ,atoms2
        do l=1,atoms1
           ! call derivative for this parameter of this energy component for ith datpt
           term = E_component(j,i,2,l,k)
           dparam(k)=dparam(k) + fac * term
        enddo
     enddo

  enddo


  ! use this subroutine to map derivatives wrt all atoms, to derivatives wrt unique atom types (parameters)
  call createdkaisq(dkaisq,dparam)


end function dkaisq



subroutine createdkaisq(dkaisq,dparam)
  use variables
  real*8,dimension(:),intent(in)::dparam
  real*8,dimension(:),intent(inout)::dkaisq
  INTEGER::i,j,k,atoms2,atms2

  atoms2=size(atomtypemon2)
  atms2=size(atmtypemon2)

  do j=1,atms2
     do k=1,atoms2
        if(atomtypemon2(k) .eq. atmtypemon2(j) ) then
           dkaisq(j) = dkaisq(j) + dparam(k)
        endif
     enddo
  enddo

end subroutine createdkaisq
