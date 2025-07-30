
!****************************************************************
!
!                 Energy routines
!
!****************************************************************


!****************** Electrostatic energy *********************
FUNCTION elecenergy(n)
  USE variables
  interface
     function screen(atom1,atom2,monomer1,monomer2,dist)
       use variables
       real*8::screen
       integer,intent(in)::atom1,atom2,monomer1,monomer2
       real*8,intent(in)::dist
     end function screen
  end interface
  integer,INTENT(in)::n
  REAL*8::elecenergy
  INTEGER::atoms1,atoms2,i,j,k
  REAL*8::sum,dist,xyz(3)

  atoms1=SIZE(atomtypemon1)
  atoms2=SIZE(atomtypemon2)
  sum =0d0


  DO j=1,atoms1
     DO k=1,atoms2
        xyz(:)=xyzmon1(n,j,:)-xyzmon2(n,k,:)
        dist=sqrt(dot_product(xyz,xyz))

! see if we want electrostatic screening
    Select Case(elec_screen)
        Case("yes")
!!!!!!!!!!!!!!!!!!!coulomb terms with damping
        sum=sum+screen(j,k,1,2,dist)*chargeatoms1(j)*chargeatoms2(k)/dist
        case default
          ! no screening
        sum=sum + chargeatoms1(j)*chargeatoms2(k)/dist
    End Select

!!!!!!!!!!!!!!!!!!! charge penetration contribution
        sum = sum - Elecoverlap(j,k)*exp(-exponents(j,k)*dist)

     ENDDO
  enddo


  elecenergy=sum


END FUNCTION elecenergy


!****************** Induction energy *********************
function Inducenergy(n)
  use variables
  interface
     function screen(atom1,atom2,monomer1,monomer2,dist)
       use variables
       real*8::screen
       integer,intent(in)::atom1,atom2,monomer1,monomer2
       real*8,intent(in)::dist
     end function screen
  end interface
  real*8::Inducenergy
  integer,intent(in)::n
  real*8::dist,dist2,sum1,xyz(3)
  integer::i,j,atoms1,atoms2
  real*8,dimension(3)::store,xyzdisp


  atoms1=size(atomtypemon1)
  atoms2=size(atomtypemon2)

  sum1=0.0

!!!!!!!!!!!!!!!!!!! charge penetration contribution
  DO i=1,atoms1
     DO j=1,atoms2
        xyz(:)=xyzmon1(n,i,:)-xyzmon2(n,j,:)
        dist=sqrt(dot_product(xyz,xyz))
        sum1 = sum1 - Inducoverlap(i,j)*exp(-exponents(i,j)*dist)
     ENDDO
  enddo

!************************* DRUDE OSCILLATOR SECTION *******************************



!****************************** monomer 1 *******************************************

  do i=1,atoms1
!!!!!!!!!!!!!!!!!!!!!!!!!!spring energy 
     xyz(:)=nxyzshell1(n,i,:)-xyzmon1(n,i,:)
     dist2=dot_product(xyz,xyz)
     sum1=sum1+.5*springcon1*dist2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!interaction energy between all induction charges on monomer1
     do j=1,atoms1
        if(i.ne.j) then
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!atom-atom
           xyz(:)=xyzmon1(n,i,:)-xyzmon1(n,j,:)
           dist=sqrt(dot_product(xyz,xyz))
           sum1=sum1+.5*screen(i,j,1,1,dist)*(-shellcharge1(i))*(-shellcharge1(j))/(dist)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!shell-shell
           xyz(:)=nxyzshell1(n,i,:)-nxyzshell1(n,j,:)
           dist=sqrt(dot_product(xyz,xyz))
           sum1=sum1+.5*screen(i,j,1,1,dist)*(shellcharge1(i))*(shellcharge1(j))/(dist)
!!!!!!!!!!!!!!!!!!!!!!!!!atom-shell
           xyz(:)=xyzmon1(n,i,:)-nxyzshell1(n,j,:)
           dist=sqrt(dot_product(xyz,xyz))
           sum1=sum1+screen(i,j,1,1,dist)*(-shellcharge1(i))*(shellcharge1(j))/(dist)
        endif
     enddo
!!!!!!!!!!!!!!!!!!!!!!!!!interaction energy between monomers
     do j=1,atoms2
        xyz(:)=xyzmon1(n,i,:)-xyzmon2(n,j,:)
        dist=sqrt(dot_product(xyz,xyz))
        sum1=sum1+screen(i,j,1,2,dist)*(-shellcharge1(i))*(chargeatoms2(j))/(dist)

        xyz(:)=nxyzshell1(n,i,:)-xyzmon2(n,j,:)
        dist=sqrt(dot_product(xyz,xyz))
        sum1=sum1+screen(i,j,1,2,dist)*(shellcharge1(i))*(chargeatoms2(j))/(dist)
     enddo
  enddo         !!!!!!!!!!!!!!!!!end loop over induction sites i on monomer 1




!****************************** monomer 2 *******************************************

  do i=1,atoms2
!!!!!!!!!!!!!!!!!!!!!!!!!!spring energy 
     xyz(:)=nxyzshell2(n,i,:)-xyzmon2(n,i,:)
     dist2=dot_product(xyz,xyz)
     sum1=sum1+.5*springcon1*dist2


!!!!!!!!!!!!!!!!!!!!!!!!!!!!interaction energy between all induction charges on monomer2
     do j=1,atoms2
        if(i.ne.j) then
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!atom-atom
           xyz(:)=xyzmon2(n,i,:)-xyzmon2(n,j,:)
           dist=sqrt(dot_product(xyz,xyz))   
           sum1=sum1+.5*screen(i,j,2,2,dist)*(-shellcharge2(i))*(-shellcharge2(j))/(dist)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!shell-shell
           xyz(:)=nxyzshell2(n,i,:)-nxyzshell2(n,j,:)
           dist=sqrt(dot_product(xyz,xyz))   
           sum1=sum1+.5*screen(i,j,2,2,dist)*(shellcharge2(i))*(shellcharge2(j))/(dist)
!!!!!!!!!!!!!!!!!!!!!!!!!atom-shell
           xyz(:)=xyzmon2(n,i,:)-nxyzshell2(n,j,:)
           dist=sqrt(dot_product(xyz,xyz))   
           sum1=sum1+screen(i,j,2,2,dist)*(-shellcharge2(i))*(shellcharge2(j))/(dist)
        endif
     enddo
!!!!!!!!!!!!!!!!!!!!!!!!!interaction energy between monomers
     do j=1,atoms1

        xyz(:)=xyzmon2(n,i,:)-xyzmon1(n,j,:)
        dist=sqrt(dot_product(xyz,xyz)) 
        sum1=sum1+screen(i,j,2,1,dist)*(-shellcharge2(i))*(chargeatoms1(j))/(dist)

        xyz(:)=nxyzshell2(n,i,:)-xyzmon1(n,j,:)
        dist=sqrt(dot_product(xyz,xyz)) 
        sum1=sum1+screen(i,j,2,1,dist)*(shellcharge2(i))*(chargeatoms1(j))/(dist)
     enddo
  enddo         !!!!!!!!!!!!!!!!!end loop over induction sites i on monomer 1


  Inducenergy=sum1

end function Inducenergy




!****************** Dispersion energy *********************
function dispenergy(p)
  use variables
  real*8 :: dispenergy
  integer,intent(in)::p
 interface
     function damp(atom1,atom2,r,n)
       use variables
       real*8::damp
       integer,intent(in)::atom1,atom2,n
       real*8,intent(in)::r
     end function damp
  end interface
  INTEGER::i,j,m,n,atoms1,atoms2,order
  REAL*8::sum,dist2,dist,rij(3)


  atoms1=size(xyzmon1(1,:,1))
  atoms2=size(xyzmon2(1,:,1))

  sum=0.0
  DO m=1,atoms1
     DO n=1,atoms2
        rij(:) = xyzmon1(p,m,:)-xyzmon2(p,n,:)
        dist2= dot_product(rij,rij)
        dist = sqrt(dist2)
        do j=1,4
           ! C6, C8, C10, C12 contributions
           order = 6 + (j-1)*2
           term = damp(m,n,dist,order)* Cn_cross(m,n,j)/dist2**(order/2)
           sum=sum - term
        enddo
             
     ENDDO
  ENDDO

  dispenergy = sum


end function dispenergy



!****************************************
! this is the damping function for dispersion terms
!****************************************
function damp(atom1,atom2,r,n)
  use variables
  real*8::damp
  integer,intent(in)::atom1,atom2,n
  real*8,intent(in)::r
  real*8 :: lambda
  integer:: i
  real*8 :: sum


  lambda= exponents(atom1,atom2)
  sum=1d0
  do i=1,n
     sum=sum+((lambda*r)**i)/dble(factorial(i))
  enddo

  damp=1d0-exp(-lambda*r)*sum

contains
  function factorial(n)
    integer::factorial
    integer,intent(in)::n
    integer :: i
    factorial=1
    do i=1,n
       factorial=factorial*i
    enddo
  end function factorial

end function damp



!****************** Exchange energy *********************
function exchenergy(n)
  use variables
  real*8::exchenergy
  integer,intent(in)::n
  INTEGER::atoms1,atoms2,i,j,k
  REAL*8::sum,dist,xyz(3)

  atoms1=SIZE(atomtypemon1)
  atoms2=SIZE(atomtypemon2)

  sum=0.0
  DO i=1,atoms1
     DO j=1,atoms2
        xyz(:)=xyzmon1(n,i,:)-xyzmon2(n,j,:)
        dist=sqrt(dot_product(xyz,xyz))
        sum=sum+Exoverlap(i,j)*exp(-exponents(i,j)*dist)
     ENDDO
  enddo

  exchenergy=sum

end function exchenergy




!****************** delta HF energy *********************
function dhfenergy(n)
  use variables
  real*8::dhfenergy
  integer,intent(in)::n
  INTEGER::atoms1,atoms2,i,j,k
  REAL*8::sum,dist,xyz(3),h_energy

  atoms1=SIZE(atomtypemon1)
  atoms2=SIZE(atomtypemon2)

  sum=0.0
  DO i=1,atoms1
     DO j=1,atoms2
        xyz(:)=xyzmon1(n,i,:)-xyzmon2(n,j,:)
        dist=sqrt(dot_product(xyz,xyz))
        sum=sum - dhfoverlap(i,j)*exp(-exponents(i,j)*dist)     
     ENDDO
  enddo

  dhfenergy=sum

  ! here, if requested, include the difference between self-consistently optimizing drude-oscillators
  ! and optimizing drude-oscillators using the static field
  ! this is a higher order induction contribution, and should be included in dhf energy

  Select Case( drude_oscillators )
  Case("yes")
     Select Case( include_high_order_drude_dhf_fit )
     Case("yes")
        dhfenergy = dhfenergy + high_order_dhf_energy(n)
     End Select
  End Select

end function dhfenergy















!****************************************************************
!
!                 derivative routines
!  note, these routines account for the fact that the sign of the buckingham coefficient for
!  each type of term is explicitly accounted for in the equation, and not the parameters.  Absolute values of the
!  parameters are taken in evaluating the energy terms, and this is important in determining how the sign of the
!  derivative is treated
!
!  probably would have been a better idea to fit the squares of the parameters, that way we wouldn't have to deal with 
!  the sign issue
!****************************************************************



function dexchenergy(atom1,atom2,n)
  use variables
  real*8::dexchenergy
  integer,intent(in)::atom1,atom2,n
  real*8::dist,xyz(3)

  ! make sure derivative doesn't blow up from small coeff
  if ( ( abs(Exchatom2(atom2)) < small_coeff ) .or. ( abs(Exchatom1(atom1)) < small_coeff ) ) then
     dexchenergy = 0d0
  else
     xyz(:)=xyzmon1(n,atom1,:)-xyzmon2(n,atom2,:)
     dist=sqrt(dot_product(xyz,xyz))
     dexchenergy = exp(-exponents(atom1,atom2)*dist)
     ! check sign of Exchatom2 double contribution since two identical monomers
     if( Exchatom2(atom2) .gt. 0.0) then
        dexchenergy=dexchenergy * 2d0 * (.5d0)/sqrt(abs(Exchatom2(atom2)*Exchatom1(atom1)))*abs(Exchatom1(atom1))
     else
        dexchenergy=dexchenergy * 2d0 * -(.5d0)/sqrt(abs(Exchatom2(atom2)*Exchatom1(atom1)))*abs(Exchatom1(atom1))
     endif

  endif

end function dexchenergy


function delecenergy(atom1,atom2,n)
  use variables
  real*8::delecenergy
  integer,intent(in)::atom1,atom2,n
  real*8::dist,xyz(3)

  ! make sure derivative doesn't blow up from small coeff
  if ( ( abs(Elecatom2(atom2)) < small_coeff ) .or. ( abs(Elecatom1(atom1)) < small_coeff ) ) then
     delecenergy = 0d0
  else
     xyz(:)=xyzmon1(n,atom1,:)-xyzmon2(n,atom2,:)
     dist=sqrt(dot_product(xyz,xyz))
     delecenergy= -exp(-exponents(atom1,atom2)*dist)

     ! check sign of Elecatom2 double contribution since two identical monomers
     if( Elecatom2(atom2) .gt. 0.0) then
        delecenergy=delecenergy * 2d0 * (.5d0)/sqrt(abs(Elecatom2(atom2)*Elecatom1(atom1)))*abs(Elecatom1(atom1))
     else
        delecenergy=delecenergy * 2d0 * -(.5d0)/sqrt(abs(Elecatom2(atom2)*Elecatom1(atom1)))*abs(Elecatom1(atom1))
     endif

  endif
end function delecenergy



function dinducenergy(atom1,atom2,n)
  use variables
  real*8::dinducenergy
  integer,intent(in)::atom1,atom2,n
  real*8::dist,xyz(3)

  ! make sure derivative doesn't blow up from small coeff
  if ( ( abs(Inducatom2(atom2)) < small_coeff ) .or. ( abs(Inducatom1(atom1)) < small_coeff ) ) then
     dinducenergy = 0d0
  else
     xyz(:)=xyzmon1(n,atom1,:)-xyzmon2(n,atom2,:)
     dist=sqrt(dot_product(xyz,xyz))
     dinducenergy= -exp(-exponents(atom1,atom2)*dist)

     ! check sign double contribution since two identical monomers
     if( Inducatom2(atom2) .gt. 0.0) then
        dinducenergy=dinducenergy * 2d0 * (.5d0)/sqrt(abs(Inducatom2(atom2)*Inducatom1(atom1)))*abs(Inducatom1(atom1))
     else
        dinducenergy=dinducenergy * 2d0 * -(.5d0)/sqrt(abs(Inducatom2(atom2)*Inducatom1(atom1)))* abs(Inducatom1(atom1))
     endif

  endif
end function dinducenergy


function ddhf(atom1,atom2,n)
  use variables
  real*8::ddhf
  integer,intent(in)::atom1,atom2,n
  real*8::dist,xyz(3)

  ! make sure derivative doesn't blow up from small coeff
  if ( ( abs(dhf2(atom2)) < small_coeff ) .or. ( abs(dhf1(atom1)) < small_coeff ) ) then
     ddhf = 0d0
  else
     xyz(:)=xyzmon1(n,atom1,:)-xyzmon2(n,atom2,:)
     dist=sqrt(dot_product(xyz,xyz))
     ddhf= -exp(-exponents(atom1,atom2)*dist)

     ! check sign, double contribution since two identical monomers
     if( dhf2(atom2) .gt. 0.0) then
        ddhf=ddhf * 2d0 * (.5d0)/sqrt(abs(dhf2(atom2)*dhf1(atom1)))*abs(dhf1(atom1))
     else
        ddhf=ddhf * 2d0 * -(.5d0)/sqrt(abs(dhf2(atom2)*dhf1(atom1)))*abs(dhf1(atom1))
     endif


  endif

end function ddhf






!************************************************
! this function determines which energy routines to call based on
! which energy components are to be fit
! init =0 initializes the fit (sets value of Ecomp)
! init =1 calls fit functions
! init =2 calls derivatives
! atom1,atom2, are for derivatives
!***********************************************
function E_component(type,datpt,init,atom1,atom2)
  use variables
  use interface
  real*8:: E_component
  integer,intent(in)::type,datpt,init
  integer,optional,intent(in)::atom1,atom2
  integer :: terms

  ! arguments atom1,atom2 only need to be present if init=2, meaning derivatives are being called

  Select case (init)
  case(0)
     ! fit is initialized
     Ecomp(1,:) = E1exch(:)
     Ecomp(2,:) = E1pol(:)
     Ecomp(3,:) = E2ind(:) + E2indexch(:)
     Ecomp(4,:) = dhf(:)
     Ecomp(5,:) = E2disp(:) + E2dispexch(:)
     ! this return value is meaningless
     E_component =0.0
  case(1)
     ! call fit functions

     ! always 1=exch,2=elec,3=ind, 4=dhf, 5=disp
     Select case(type)
     case(1)
        E_component = exchenergy(datpt)
     case(2)
        E_component = elecenergy(datpt)
     case(3)
        E_component = inducenergy(datpt)
     case(4)
        E_component = dhfenergy(datpt)
     case(5)
        E_component = dispenergy(datpt)
     end select

  case(2)
     ! arguments atom1,atom2 need to be present
     if (present(atom1) .and. present(atom2)) then
        ! nothing
     else
        stop "E_component is being called for derivatives without specifying atom1,atom2"
     endif

     ! call derivative functions

     Select case(type)
     case(1)
        E_component = dexchenergy(atom1,atom2,datpt)
     case(2)
        E_component = delecenergy(atom1,atom2,datpt)
     case(3)
        E_component = dinducenergy(atom1,atom2,datpt)
     case(4)
        E_component = ddhf(atom1,atom2,datpt)
     end select

  case default
     stop "unknown value of init in call to function E_component"
  end select

end function E_component


!*********************************************
! this subroutine computes the final energies of the
! different terms after all parameters have been optimized
!*********************************************
subroutine Efitout(rms)
  use variables
  use routines
  real*8,dimension(:),intent(inout)::rms
  interface
     function E_component(type,datpt,init,atom1,atom2)
       use variables
       use interface
       real*8:: E_component
       integer,intent(in)::type,datpt,init
       integer,optional,intent(in)::atom1,atom2
     end function E_component
  end interface
  integer::i,j,k,datpts,count,atoms1,atoms2
  real*8::sumE,sum,sum1,dist
  real*8,dimension(3)::xyz

  atoms1=size(xyzmon1(1,:,1))
  atoms2=size(xyzmon2(1,:,1))
  datpts=size(xyzmon1(:,1,1))

  sumE=0d0
  count=0
  j = component_fit

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



  if (component_fit < 6 ) then
     do i=1,datpts
        if ( Etot(i) < Ecutoff) then
           Eout(j,i) = E_component(j,i,1)
           count=count+1
           sumE = sumE + (Eout(j,i)-Ecomp(j,i))**2
        endif
     enddo

     rms(j) = (sumE/real(count))**.5


  else

        do i=1,datpts

           if ( Etot(i) < Ecutoff) then
              Eout(j,i) = E_component(1,i,1) + E_component(2,i,1) + E_component(3,i,1) + E_component(4,i,1) + E_component(5,i,1)
              count=count+1
              sumE = sumE + (Eout(j,i)-Etot(i))**2
           endif
        enddo

        rms(j) = (sumE/real(count))**.5

  endif


end subroutine Efitout



!****************************************
! Here are the electrostatic screen functions, that are
! used for both static charges, as well as drude oscillator charges
!
! note that only intramolecular electrostatic interactions occur between
! drude oscillators, and so if monomer1 equals monomer2, then this is
! a intra molecular drude oscillator interaction, and we use the Thole screening
! function
!
! otherwise, it is an intermolecular interaction, and we use the Tang-Toennies screening function
!*****************************************
function screen(atom1,atom2,monomer1,monomer2,dist)
  use variables
  real*8::screen
  integer,intent(in)::atom1,atom2,monomer1,monomer2
  real*8,intent(in)::dist
  integer::i,j
  real*8::pol1,pol2,a
  real*8,parameter::small=1D-6
  
  a=screenlength
  i= atom1
  j= atom2

  ! set screen = zero for zero charge otherwise blows up

  if( monomer1 .eq. monomer2 ) then
     if((abs(shellcharge1(i)).lt. small).or.( abs(shellcharge1(j)).lt. small)) then
        screen=0d0
     else
      pol1=(shellcharge1(i)**2)/springcon1
      pol2=(shellcharge1(j)**2)/springcon1
      screen=1.0-(1.0+(a*dist)/(2.*(pol1*pol2)**(1./6.)))*exp(-a*dist/(pol1*pol2)**(1./6.))
      endif
  elseif ((monomer1.eq.2).and.(monomer2.eq.1))then

      screen=1D0-(1D0+exponents(j,i)*dist)*exp(-exponents(j,i)*dist)

  elseif ((monomer1.eq.1).and.(monomer2.eq.2))then

      screen=1D0-(1D0+exponents(i,j)*dist)*exp(-exponents(i,j)*dist)

  endif
 

end function screen


!**********************************************
! this is the gradient of the above screening function, used for forces
!**********************************************
function  dscreen(atom1,atom2,monomer1,monomer2,xyz)
  use variables
  real*8,dimension(3)::dscreen
  integer,intent(in)::atom1,atom2,monomer1,monomer2
  real*8,dimension(3),intent(in)::xyz
  integer::i,j
  real*8::pol1,pol2,dist,fac,Ex,a
  real*8,parameter::small=1D-6

  a=screenlength
  i= atom1
  j= atom2

  dist=(xyz(1)**2+xyz(2)**2+xyz(3)**2)**.5


  if( monomer1 .eq. monomer2) then
     ! set dscreen = zero for zero charge otherwise blows up
     if((abs(shellcharge1(i)).lt. small).or.( abs(shellcharge1(j)).lt. small)) then
        dscreen =0d0
     else
        pol1=(shellcharge1(i)**2)/springcon1
        pol2=(shellcharge1(j)**2)/springcon1

        fac=a/(pol1*pol2)**(1./6.)
        Ex=exp(-fac*dist)
        dscreen(:)=(xyz(:)/dist)*(fac*(1.0+fac*dist/2.)*Ex-(fac/2.)*Ex)

     endif
  elseif ((monomer1.eq.2).and.(monomer2.eq.1))then
     fac= exponents(j,i)
     Ex=exp(-fac*dist)
     dscreen(:)=(xyz(:)/dist)*(fac*(1.0+fac*dist)*Ex- fac*Ex )

  elseif ((monomer1.eq.1).and.(monomer2.eq.2))then
     fac= exponents(i,j)
     Ex=exp(-fac*dist)
     dscreen(:)=(xyz(:)/dist)*(fac*(1.0+fac*dist)*Ex- fac*Ex )
  endif


end function dscreen
