!***************************************************************
! this module controls the optimization of drude oscillators
!
! note that this seems more complicated than it is, because we've broken
! up the drude oscillator response into a linear response, and a
! self-consistent "infinite" order response
! this is because we use the linear response for the 2nd order induction energy,
! and the higher order response ( scf - linear ) for the dhf component
! 
!***************************************************************

module eq_drude
integer :: flag_intra ! set to zero if intra molecular electrostatics shouldn't be considered (no drudes)

contains


!*********************************************************************
!  This subroutine finds the linear response positions of the drude oscillators for
!  each data point
!*********************************************************************
  subroutine initialize_drude_positions(datpts)
    use variables
    integer,intent(in) :: datpts

    integer :: i

    do i=1,datpts
!!!!!!!!!!!!for each data point start with shells close to atom
       nxyzshell1(i,:,:)=xyzmon1(i,:,:)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       call findpositionshell(i,1)
       nxyzshell2(i,:,:)=xyzmon2(i,:,:)
       call findpositionshell(i,2)
    enddo

  end subroutine initialize_drude_positions


!*********************************************************************
!  This subroutine finds the "infinite" response positions of the drude oscillators for
!  each data point
!*********************************************************************
subroutine initialize_high_order_dhfenergy
  use variables
  INTEGER::datpts,i
  real*8 :: h_energy

  datpts=size(xyzmon1(:,1,1))

  if ( allocated(high_order_dhf_energy) ) then
     deallocate(high_order_dhf_energy)
  endif
   allocate(high_order_dhf_energy(datpts))

  ! set flag_intra=1 for drude oscillator calculations
   flag_intra=1

  ! here, if requested, include the difference between self-consistently optimizing drude-oscillators
  ! and optimizing drude-oscillators using the static field
  ! this is a higher order induction contribution, and probably should be included in dhf energy

  Select Case( drude_oscillators )
  Case("yes")
     Select Case( include_high_order_drude_dhf_fit )
     Case("yes")
        do i=1, datpts
        Call high_order_drude_energy(i,h_energy)
        high_order_dhf_energy(i) = h_energy
        enddo
     End Select
  End Select

end subroutine initialize_high_order_dhfenergy



!***************************************
! this subroutine calculates the higher order drude
! oscillator energy, by subtracting energies from 
! the "infinite" response, and the linear response
! of the drude oscillators
!***************************************
subroutine high_order_drude_energy (i_data,energy)
  use routines
  use variables
  interface
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
     end function Inducenergy
  end interface
    integer,intent(in) :: i_data
    real*8,intent(out) :: energy

    real*8,dimension(:,:,:), allocatable :: xyz,xyz_drude,force_atoms
    real*8,dimension(:,:), allocatable :: tot_chg,store_exponents
    integer,dimension(:),allocatable :: n_atom
    integer :: i,j,atoms,ind1,ind2,i_atom, j_atom,tot_atoms,iteration
    real*8 :: energy_high_order, energy_2nd_order,energy_tot,energy_nopol,dist
    real*8,dimension(3) :: dx

    atoms = size (xyzmon1(1,:,1))
    tot_atoms = 2*atoms

    ! first initialize data structures that we'll need 

    allocate(xyz(2,tot_atoms,3),xyz_drude(2,atoms,3),tot_chg(2,tot_atoms),n_atom(2),store_exponents(atoms,atoms))

    do i=1, atoms
       xyz(1,i,:) = xyzmon1(i_data,i,:)
       xyz(2,i,:) = xyzmon2(i_data,i,:)
       tot_chg(1,i) = chargeatoms1(i) - shellcharge1(i)
       tot_chg(2,i) = chargeatoms2(i) - shellcharge2(i)
       tot_chg(1,atoms+i) = shellcharge1(i)
       tot_chg(2,atoms+i) = shellcharge2(i)
    enddo

    n_atom = atoms

    ! global variable array "exponents" will be used in electrostatic screening functions.  For drude oscillator screening,
    ! the easiest thing to do is to duplicate these exponents for the drude oscillators
    store_exponents=exponents
    deallocate(exponents)
    allocate(exponents(tot_atoms,tot_atoms))

    do i_atom=1,tot_atoms
       if ( i_atom > atoms) then
          ind1 = i_atom - atoms
       else
          ind1 = i_atom
       endif
       do j_atom=1, tot_atoms
          if ( j_atom > atoms ) then
             ind2 = j_atom - atoms
          else
             ind2 = j_atom
          endif

          exponents(i_atom,j_atom) = store_exponents(ind1,ind2)

       enddo
    enddo


    ! note scf_drude routine has been modified to output energy in au
    call scf_drude(energy_tot, energy_nopol ,xyz,tot_chg,2,n_atom,iteration,xyz_drude)
    energy_high_order = energy_tot - energy_nopol

    ! fix exponents array
    deallocate(exponents)
    allocate(exponents(atoms,atoms))
    exponents = store_exponents


    ! get 2nd order energy
  call combinegeom(Inducatom1,Inducatom2,Inducoverlap)
    energy_2nd_order = Inducenergy(i_data)

    ! get rid of charge penetration contribution
    DO i=1,atoms
       DO j=1,atoms
          dx(:) = xyzmon1(i_data,i,:)-xyzmon2(i_data,j,:)
          dist= sqrt(dot_product(dx,dx))
          energy_2nd_order = energy_2nd_order - ( - Inducoverlap(i,j)*exp(-exponents(i,j)*dist) )
       ENDDO
    enddo


    energy = energy_high_order - energy_2nd_order



  end subroutine high_order_drude_energy




!*************************************************
!
!
!
!        THIS SECTION OF SUBROUTINES CONTROLS THE "INFINITE", SELF-CONSISTENT
!        DRUDE OSCILLATOR RESPONSE
!
!
!
!**************************************************





!******************************************************************!
! this subroutine finds optimal drude oscillator positions
! for a given configuration of molecules.  The notation used in these subroutines
! is different from the rest of the code, because these subroutines were
! adapted from the Monte Carlo code
!
!  global variables used:
!  real*8,parameter::springcon (.1)
!  real*8,parameter::thole     (2.)
!  real*8,parameter:: force_threshold
!  currently,this assumes an oscillator on all atoms, so that
!  there is a one to one correspondence
!  therefore, if there isn't supposed to be an oscillator on an atom,
!  make sure the corresponding chg is zero
!
!  work in atomic units : NOTE this is different than subroutine
!  in monte carlo code where units are in e^2/A
! 
!  input array "chg" should be total charges including drudes
!
! this subroutine uses a conjugate gradient method to find equilibrium 
! drude positions
! based off Lindan, P.J.D, Gillan,M.J., J. Phys.: Condens. Matter 5 (1993) 1019-1030
!******************************************************************!
  subroutine scf_drude(energy,energy_nopol,xyz,chg,n_mole,n_atom,iteration,xyz_drude)
    use variables
    real*8,intent(out)::energy,energy_nopol
    integer,intent(in)::n_mole
    integer,intent(out)::iteration
    integer,dimension(:),intent(in)::n_atom
    real*8, dimension(:,:,:),intent(in) :: xyz
    real*8, dimension(:,:,:),intent(out) :: xyz_drude
    real*8, dimension(:,:),intent(in) :: chg
    real*8,dimension(size(xyz(:,1,1)),size(xyz(1,:,1)),size(xyz(1,1,:)))::force,force_old,force_new,spring_force
    real*8, dimension(size(xyz(:,1,1)),size(xyz(1,:,1)),size(xyz(1,1,:))) ::tot_xyz
    integer,dimension(size(xyz(:,1,1)),size(xyz(1,:,1)))::drude_atoms
    real*8,dimension(size(xyz(:,1,1)),size(xyz(1,:,1)))::temp_chg
    integer, dimension(size(xyz(:,1,1))) :: tot_n_atom
    real*8,dimension(n_mole,maxval(n_atom),3)::search_d
    integer::i,i_mole,i_atom,i_drude,converged,tot_drudes,flag
    real*8::Beta,lambda1,sum_f_old,sum_f_new,sum_lambda_n,sum_lambda_d,Ewald_self_store
    real*8,dimension(3)::delta_xyz,disp_xyz,unit
    real*8,parameter::drude_initial=.005
    real*8::springE,springcon
    real*8,parameter::drude_max=1.

    springcon = springcon1

!!!!!!!!!!!!!!!! create tot_n_atom list, assuming an oscillator on every non-framework atom
    tot_n_atom = n_atom
    do i_mole=1,n_mole
       tot_n_atom(i_mole) =2*tot_n_atom(i_mole)
    enddo
    ! make sure to include framework positions
    tot_xyz=xyz

    drude_atoms=0
    iteration=0
    converged=0
!!!!!!!!!!!!!! combine positions of atoms and oscillators into one array to pass to force subroutine
!!!!!!!!!!!!!! remember that chg on atom equals permanent chg - charge on drude oscillator
!!!!!!!!!!!!!! while looping, construct drude_atoms array, which tells force routine for which atoms to compute forces


!!!!!!!!!!!!!! if first call to this subroutine, put drude oscillators on respective atoms, otherwise use passed array
! put drude oscillators slightly off atoms so that erf(x)/x doesn't blow up, also distribute oscillators randomly
!so no net dipole of system
!!$
       do i_mole=1,n_mole
          do i_atom=1,n_atom(i_mole)
             tot_xyz(i_mole,i_atom,:)=xyz(i_mole,i_atom,:)
             call random_unit_vector(unit)
             xyz_drude(i_mole,i_atom,:)= xyz(i_mole,i_atom,:) + drude_initial * unit

             tot_xyz(i_mole,n_atom(i_mole)+i_atom,:)=xyz_drude(i_mole,i_atom,:)
             drude_atoms(i_mole,i_atom)=n_atom(i_mole)+i_atom

          enddo
       enddo

    call F_elec_cutoff( force, drude_atoms, n_mole, tot_n_atom, chg, tot_xyz) 

!!!!!!!!!no unit conversion here as force is output in au


    do while(converged.eq.0)
       iteration=iteration+1
       converged=1


       if(iteration .eq. 1) then
          sum_lambda_n=0.D0
          lambda1 = 1.D0/springcon
          springE=0.D0
          sum_f_new=0.D0
          tot_drudes=0
!!!!!!!!!!!!if first iteration, search direction is in direction of force
!!!!!!!!!!add spring force,and first search direction is only determined by initial force
          do i_mole=1,n_mole
             do i_atom=1,n_atom(i_mole)
                tot_drudes=tot_drudes+1
                disp_xyz(:)=xyz_drude(i_mole,i_atom,:)-xyz(i_mole,i_atom,:)
!!!!!!!!!!total force including spring
                force(i_mole,i_atom,:)=force(i_mole,i_atom,:)-springcon*disp_xyz(:)
                sum_f_new = sum_f_new + dot_product(force(i_mole,i_atom,:),force(i_mole,i_atom,:))
!!!!!!!!! first trial positions determined by steepest descent
                search_d(i_mole,i_atom,:)=force(i_mole,i_atom,:)
                delta_xyz(:)= lambda1*search_d(i_mole,i_atom,:)
                xyz_drude(i_mole,i_atom,:)=xyz_drude(i_mole,i_atom,:)+delta_xyz(:)

                spring_force(i_mole,i_atom,:) = -springcon*disp_xyz(:)

                tot_xyz(i_mole,n_atom(i_mole)+i_atom,:)=xyz_drude(i_mole,i_atom,:)
                sum_lambda_n=sum_lambda_n+dot_product(force(i_mole,i_atom,:),force(i_mole,i_atom,:))
                disp_xyz(:)=xyz_drude(i_mole,i_atom,:)-xyz(i_mole,i_atom,:)
                springE = springE + .5D0*springcon*dot_product(disp_xyz,disp_xyz)
             enddo
          enddo
!!!!!!!!!! check total net force, if rms force < threshold, converged
          sum_f_new = sum_f_new/dble(tot_drudes)

          if ( sum_f_new > force_threshold**2) then
             converged=0
          endif
!!!!!!!!!!!!!!!!!!if converged, lets get outta here
          if(converged.eq.1) then
             goto 100
          endif
       else

!!!!!!!!!!if not the first iteration, search direction is determined using information from previous step
!!!!calculate Beta, which is contribution of previous direction
          sum_f_old=0.;sum_f_new=0.
          do i_mole=1,n_mole
             do i_atom=1,n_atom(i_mole)
                disp_xyz(:)=xyz_drude(i_mole,i_atom,:)-xyz(i_mole,i_atom,:)
                spring_force(i_mole,i_atom,:) = -springcon*disp_xyz(:)
                force(i_mole,i_atom,:)=force(i_mole,i_atom,:)-springcon*disp_xyz(:)
                sum_f_new = sum_f_new+dot_product(force(i_mole,i_atom,:),force(i_mole,i_atom,:))
                sum_f_old = sum_f_old+dot_product(force_old(i_mole,i_atom,:),force_old(i_mole,i_atom,:))
             enddo
          enddo
          Beta=sum_f_new/sum_f_old
!!!!!!!!!! check total net force, if rms force < threshold, converged
          sum_f_new = sum_f_new / dble(tot_drudes)
          if ( sum_f_new > force_threshold**2) then
             converged=0
          endif
!!!!!!!!!!!!!!!!!!if converged, lets get outta here
          if(converged.eq.1) then
             goto 100
          endif

!!!!!!!!!calculate directions of moves, and lambda
          sum_lambda_n=0.D0;sum_lambda_d=0.D0
          do i_mole=1,n_mole
             do i_atom=1,n_atom(i_mole)
!!!!!!!!!!!!!!!!now iterations,like conjugate gradient, use direction of previous move
                search_d(i_mole,i_atom,:)=force(i_mole,i_atom,:)+Beta*search_d(i_mole,i_atom,:)
                sum_lambda_n=sum_lambda_n+dot_product(force(i_mole,i_atom,:),search_d(i_mole,i_atom,:))
                sum_lambda_d=sum_lambda_d+springcon*dot_product(search_d(i_mole,i_atom,:),search_d(i_mole,i_atom,:))
             enddo
          enddo

          lambda1=sum_lambda_n/sum_lambda_d
          springE=0.D0
!!!!!!!!!!!!!!calculate positions of oscillators for this value of lambda
          do i_mole=1,n_mole
             do i_atom=1,n_atom(i_mole)
                delta_xyz(:) = lambda1 * search_d(i_mole,i_atom,:)
                xyz_drude(i_mole,i_atom,:)=xyz_drude(i_mole,i_atom,:)+delta_xyz(:)
                tot_xyz(i_mole,n_atom(i_mole)+i_atom,:)=xyz_drude(i_mole,i_atom,:)
                disp_xyz(:)=xyz_drude(i_mole,i_atom,:)-xyz(i_mole,i_atom,:)
                springE=springE + .5D0*springcon*dot_product(disp_xyz,disp_xyz)
             enddo
          enddo

       endif



!!!!!!!!!!!!!!!!!!!!!now calculate forces at new drude positions
    call F_elec_cutoff( force_new, drude_atoms, n_mole, tot_n_atom, chg, tot_xyz) 

  ! again no unit conversion here

       do i_mole=1,n_mole
          do i_atom=1,n_atom(i_mole)
          force_old(i_mole,i_atom,:) = force(i_mole,i_atom,:)
          force(i_mole,i_atom,:) = force_new(i_mole,i_atom,:)
          enddo
       enddo

100    continue

!!!!!!!!!!!!!!!!!!!!! if drude oscillators are not converging

       if(iteration .gt. iteration_limit) then
          exit
       endif


    enddo


 !when we get here, iterations have either converged or passed the max limit, calculate forces on atoms to output for hybrid mc/md
    energy = E_elec_cutoff( n_mole, tot_n_atom, chg, tot_xyz )

!!!!!!!!!!!!!!!!!!!!!add spring terms, no unit conversion
        energy=energy + springE 

        ! now energy without drudes
        temp_chg = chg
       do i_mole=1,n_mole
          do i_atom=1,n_atom(i_mole)
             temp_chg(i_mole,i_atom) = chg(i_mole,i_atom) + chg(i_mole, n_atom(i_mole) + i_atom)
          enddo
       enddo

       ! at this point, no drude oscillators, do not include intramolecular electrostatics
       flag_intra=0
    energy_nopol = E_elec_cutoff( n_mole, n_atom, temp_chg, xyz )
      ! set back
       flag_intra=1

    end subroutine scf_drude


!*****************************
! this subroutine calculates electrostatic force
! and outputs in atomic units, as input should be in au
!****************************
 subroutine F_elec_cutoff( elec_force,target_atoms, n_mole, n_atom, chg, xyz) 
    implicit none
    interface
     function screen(atom1,atom2,monomer1,monomer2,dist)
       use variables
       real*8::screen
       integer,intent(in)::atom1,atom2,monomer1,monomer2
       real*8,intent(in)::dist
     end function screen
     function  dscreen(atom1,atom2,monomer1,monomer2,xyz)
       use variables
       real*8,dimension(3)::dscreen
       integer,intent(in)::atom1,atom2,monomer1,monomer2
       real*8,dimension(3),intent(in)::xyz
     end function dscreen
  end interface
    real*8,dimension(:,:,:),intent(out):: elec_force
    integer, intent(in) :: n_mole
    integer, intent(in), dimension(:) :: n_atom
    real*8, intent(in), dimension(:,:) :: chg
    integer, dimension(:,:),intent(in):: target_atoms
    real*8, intent(in), dimension(:,:,:) :: xyz
    integer :: i_mole, i_atom, j_mole, j_atom, t_atom, thread_id, i_thread
    real*8, dimension(3) :: rij, r_com_i, r_com_j, dr_com, shift,f_ij
    real*8, dimension(n_mole,maxval(n_atom),3) :: local_force
    real*8 :: norm_dr,Electro_cutoff_use

    elec_force = 0D0
    local_force = 0.D0


    local_force = 0.D0

    do i_mole = 1,n_mole
       do j_mole = i_mole, n_mole

                if ( i_mole /= j_mole ) then  ! if it is not self interaction
                      do i_atom = 1, n_atom( i_mole )
                         do j_atom = 1, n_atom( j_mole )
                            rij = xyz(i_mole,i_atom,:) - xyz(j_mole,j_atom,:)
                            norm_dr = sqrt( dot_product( rij, rij ) )
                            !! add "screened" real space interaction
                            f_ij = chg(i_mole,i_atom) * chg(j_mole,j_atom) * rij * screen(i_atom,j_atom,i_mole,j_mole,norm_dr) / &
                                    norm_dr**3
                            f_ij = f_ij - chg(i_mole,i_atom) * chg(j_mole,j_atom) * dscreen(i_atom,j_atom,i_mole,j_mole,rij) /&
                                    norm_dr
                            local_force(i_mole,i_atom,:) = local_force(i_mole,i_atom,:) + f_ij(:)
                            local_force(j_mole,j_atom,:) = local_force(j_mole,j_atom,:) - f_ij(:)
                         enddo
                      enddo
                elseif (i_mole == j_mole) then
                   if(n_atom(i_mole) .gt. 1) then
                      do i_atom=1,n_atom(i_mole)-1
                         do j_atom=i_atom+1,n_atom(i_mole)
                            call intra_elec_force(f_ij,xyz,chg,i_mole,n_atom,i_atom,j_atom)
                            local_force(i_mole,i_atom,:)=local_force(i_mole,i_atom,:) + f_ij(:)
                            local_force(i_mole,j_atom,:)=local_force(i_mole,j_atom,:) - f_ij(:)                         
                         enddo
                      enddo
                   endif
                end if
       end do
    enddo

    elec_force = local_force

  ! no conversion here

    !! reorganize force array so that this is consistent with output of pme_force
    do i_mole=1,n_mole
       do i_atom=1,n_atom(i_mole)
          if(target_atoms(i_mole,i_atom).eq.0) then
             goto 200
          endif
          elec_force(i_mole,i_atom,:)=elec_force(i_mole,target_atoms(i_mole,i_atom),:)
       enddo
200    continue
    enddo

  end subroutine F_elec_cutoff


!************************ 
! this function calculates electrostatic energy and outputs in 
! atomic units, as input should be in atomic units
!************************
 real*8 function E_elec_cutoff( n_mole, n_atom, chg, xyz )
    implicit none
     interface
     function screen(atom1,atom2,monomer1,monomer2,dist)
       use variables
       real*8::screen
       integer,intent(in)::atom1,atom2,monomer1,monomer2
       real*8,intent(in)::dist
     end function screen
     end interface
    integer, intent(in) :: n_mole
    integer, intent(in), dimension(:) :: n_atom
    real*8, intent(in), dimension(:,:) :: chg
    real*8, intent(in), dimension(:,:,:) :: xyz
    integer :: i_mole, i_atom, j_mole, j_atom
    real*8, dimension(3) :: dr, r_com_i, r_com_j, dr_com, shift
    real*8 :: norm_dr,E_intra

    E_elec_cutoff=0d0

    do i_mole = 1, n_mole
       do j_mole = i_mole, n_mole

                if ( i_mole /= j_mole ) then  ! if it is not self interaction
                   do i_atom = 1, n_atom(i_mole)
                      do j_atom = 1, n_atom(j_mole)
                         dr = xyz(i_mole,i_atom,:) - xyz(j_mole,j_atom,:)
                         norm_dr = sqrt( dot_product( dr, dr ) )
                         E_elec_cutoff = E_elec_cutoff + chg(i_mole,i_atom)*chg(j_mole,j_atom)* &
                                 screen(i_atom,j_atom,i_mole,j_mole,norm_dr) / norm_dr
                      end do
                   end do
                else if (i_mole == j_mole) then
                   if(n_atom(i_mole) .gt. 1) then
                      do i_atom=1,n_atom(i_mole)-1
                         do j_atom=i_atom+1,n_atom(i_mole)  
                            call intra_elec_energy(E_intra,xyz,chg,i_mole,i_atom,j_atom,n_atom)
                            E_elec_cutoff = E_elec_cutoff + E_intra
                         enddo
                      enddo
                   endif
                end if
       end do
    end do


   ! no unit conversion here

  end function E_elec_cutoff



 !*************************************************************
  !  this subroutine adds intra molecular polarization (drude oscillator) electrostatic interactions
 ! it is meant to be used with Electrostatic cutoff routine, as no reciprocal space terms are subtracted
  !
  !  Ewald and pme electrostatics should use intra_pme_energy routine instead
  !*************************************************************
  subroutine intra_elec_energy(E_intra,xyz,chg,i_mole,i_atom,j_atom,n_atom)
    use variables
    implicit none
    interface
     function screen(atom1,atom2,monomer1,monomer2,dist)
       use variables
       real*8::screen
       integer,intent(in)::atom1,atom2,monomer1,monomer2
       real*8,intent(in)::dist
     end function screen
  end interface
    real*8, intent(out):: E_intra
    integer, intent(in), dimension(:) :: n_atom
    real*8, intent(in), dimension(:,:) :: chg
    real*8, intent(in), dimension(:,:,:) :: xyz
    integer,intent(in)::i_mole,i_atom,j_atom

    real*8,dimension(3)::rij
    integer::i_drude,j_drude,i_fix,j_fix,n_pairs,sign_chg_i,sign_chg_j,drude_p
    real*8::norm_dr,pol1,pol2,springcon
    real*8,parameter::small=1D-8

    springcon = springcon1

! decide whether we need to include intra-molecular electrostatics for drude oscillators
    Select Case(flag_intra)
       Case(0)
          E_intra=0d0
          
       Case(1)

!!!!!!!!!determine whether i_atom,j_atom are atoms or shells, this is needed because only shell charge is used for intra-molecular 
!!!!!!!!!screened interactions, and this charge needs to be taken from the corresponding drude_oscillator
       n_pairs=n_atom(i_mole)/2
       if(i_atom-n_pairs > 0) then
          i_fix=i_atom-n_pairs
          i_drude=i_atom
          sign_chg_i=1
       else
          i_fix=i_atom
          i_drude=i_atom+n_pairs
          sign_chg_i=-1
       endif

       if(j_atom-n_pairs > 0) then
          j_fix=j_atom-n_pairs
          j_drude=j_atom
          sign_chg_j=1
       else
          j_fix=j_atom
          j_drude=j_atom+n_pairs
          sign_chg_j=-1
       endif

       rij(:) = xyz(i_mole,i_atom,:) - xyz(i_mole, j_atom,:)
       norm_dr = sqrt( dot_product( rij, rij ) )

!!! if j_atom is not the drude_oscillator attached to i_atom (if i_atom is even a drude oscillator) (remember i_atom < j_atom)
       if (j_atom .ne. i_atom + n_pairs) then
!!!!!!!!!!!!!!! add screened atom-atom, get atom chg from corresponding oscillator
          pol1=chg(i_mole,i_drude)**2/springcon; pol2=chg(i_mole,j_drude)**2/springcon
          E_intra = dble(sign_chg_i*sign_chg_j)*chg(i_mole,i_drude) * chg(i_mole,j_drude) * &
                  screen(i_fix,j_fix,i_mole,i_mole,norm_dr)/ norm_dr


       else
          E_intra = 0D0
       endif

End Select

  end subroutine intra_elec_energy




 !*************************************************************
  !  this subroutine adds intra molecular polarization (drude oscillator) electrostatic forces
  !*************************************************************
  subroutine intra_elec_force(f_ij,xyz,chg,i_mole,n_atom,i_atom,j_atom)
    use variables
    implicit none
    interface
     function screen(atom1,atom2,monomer1,monomer2,dist)
       use variables
       real*8::screen
       integer,intent(in)::atom1,atom2,monomer1,monomer2
       real*8,intent(in)::dist
     end function screen
     function  dscreen(atom1,atom2,monomer1,monomer2,xyz)
       use variables
       real*8,dimension(3)::dscreen
       integer,intent(in)::atom1,atom2,monomer1,monomer2
       real*8,dimension(3),intent(in)::xyz
     end function dscreen
  end interface
    real*8, dimension(:), intent(out):: f_ij
    integer, intent(in), dimension(:) :: n_atom
    real*8, intent(in), dimension(:,:) :: chg
    real*8, intent(in), dimension(:,:,:) :: xyz
    integer,intent(in)::i_mole,i_atom,j_atom

    real*8,dimension(3)::rij
    integer::i_drude,j_drude,i_fix,j_fix,n_pairs,sign_chg_i,sign_chg_j,drude_p
    real*8::norm_dr,pol1,pol2,springcon
    real*8,parameter::small=1D-8

    springcon = springcon1

       n_pairs=n_atom(i_mole)/2

       if(i_atom-n_pairs > 0) then
          i_fix = i_atom-n_pairs
          i_drude=i_atom
          sign_chg_i=1
       else
          i_fix = i_atom
          i_drude=i_atom+n_pairs
          sign_chg_i=-1
       endif

       if(j_atom-n_pairs > 0) then
          j_fix = j_atom-n_pairs
          j_drude=j_atom
          sign_chg_j=1
       else
          j_fix= j_atom
          j_drude=j_atom+n_pairs
          sign_chg_j=-1
       endif

       rij(:) = xyz(i_mole,i_atom,:) - xyz(i_mole, j_atom,:)
       norm_dr = sqrt( dot_product( rij, rij ) )

!!!!!!!!!!!!!!! if j_atom is not the drude_oscillator attached to i_atom (or vice-versa)
       if ((j_atom .ne. i_atom + n_pairs).and.(i_atom .ne. j_atom + n_pairs)) then
!!!!!!!!!!!!add screened atom-drude, get atom chg from corresponding oscillator
          pol1=chg(i_mole,i_drude)**2/springcon; pol2=chg(i_mole,j_drude)**2/springcon
          f_ij= dble(sign_chg_i*sign_chg_j)*chg(i_mole,i_drude) * chg(i_mole,j_drude) *&
         (rij * screen(i_fix,j_fix,i_mole,i_mole,norm_dr)/ norm_dr**3 - dscreen(i_fix,j_fix,i_mole,i_mole,rij) / norm_dr)
       else
          f_ij =0D0
       endif


  end subroutine intra_elec_force

  !*************************************************************************
  ! This subroutine generates a random unit vector
  ! from 'computer simulation of liquids', Allen,M.P.,Tildesley,D.J.
  !*************************************************************************
  subroutine random_unit_vector(unit)
    real*8,dimension(3),intent(out)::unit
    real*8::ransq,ran1,ran2,ranh
    real*8,dimension(3)::randnum3
    ransq=2.
    do while (ransq.ge.1)
       call random_number(randnum3)
       ran1=1.-2.*randnum3(1)
       ran2=1.-2.*randnum3(2)
       ransq=ran1*ran1+ran2*ran2
    enddo
    ranh=2.*sqrt(1.-ransq)
    unit(1)=ran1*ranh
    unit(2)=ran2*ranh
    unit(3)=(1.-2.*ransq)

  end subroutine random_unit_vector










!*************************************************
!
!
!
!        THIS SECTION OF SUBROUTINES CONTROLS THE LINEAR, FIRST ORDER
!        DRUDE OSCILLATOR RESPONSE
!
!
!
!**************************************************







 !*****************************************************
 ! this subroutine finds the linear response optimal positions of the drude oscillators
 ! for the given data point "n", and the given molecule "mon"
 !*****************************************************
  recursive subroutine findpositionshell(n,mon)
    use variables
    integer,intent(in)::n,mon
    real*8,dimension(:,:),allocatable::forces1,forces2
    integer::atoms1,atoms2,i,j,k,l,iter,converged,anharm
    integer,parameter :: maxiterations=1000
    real*8::delta,r,sum
    real*8,dimension(3)::dist,xxout,del
    real*8,parameter::step=0.1,step1=.005,anharmonic=3.,anharmdist=0.4,small=1D-7
    integer,dimension(10000,2),save:: store_iterations=0
      

    atoms1=size(atomtypemon1)
    atoms2=size(atomtypemon2)


    if(mon.eq.1) then
       allocate(forces1(atoms1,3))
       converged=1
       anharm=0

       do i=1,atoms1
          dist(:)=nxyzshell1(n,i,:)-xyzmon1(n,i,:)
          r =sqrt(dot_product(dist,dist))

          forces1(i,:)=shellcharge1(i)*field1(i,1,n)


          do j=1,3
             delta=abs(forces1(i,j)-springcon1*dist(j))
             if(delta > deltashell) converged=0
          enddo
       enddo
       if (converged.eq.0) then
          do i=1,atoms1
             xxout(:)=forces1(i,:)/springcon1
             del(:)=xyzmon1(n,i,:)+xxout(:)-nxyzshell1(n,i,:)
             nxyzshell1(n,i,:)=nxyzshell1(n,i,:)+step*del(:)
          enddo

          deallocate(forces1)
          call findpositionshell(n,1)
       endif

    else

       allocate(forces2(atoms2,3))
       converged=1
       anharm=0

       do i=1,atoms2
          dist(:)=nxyzshell2(n,i,:)-xyzmon2(n,i,:)
          r=sqrt(dot_product(dist,dist))

          forces2(i,:)=shellcharge2(i)*field1(i,2,n)


          do j=1,3
             delta=abs(forces2(i,j)-springcon1*dist(j))
             if(delta > deltashell) converged=0
          enddo
       enddo
       if (converged.eq.0) then
          do i=1,atoms2
             xxout(:)=forces2(i,:)/springcon1
             del(:)=xyzmon2(n,i,:)+xxout(:)-nxyzshell2(n,i,:)

             nxyzshell2(n,i,:)=nxyzshell2(n,i,:)+step*del(:)

          enddo

          deallocate(forces2)
          call findpositionshell(n,2)
       endif

    endif

    ! check iteration

    if ( store_iterations(n,mon) > maxiterations ) then
       write(*,*) " drude oscillator minimization for ", mon," molecule of "
       write(*,*) n, " datapoint isn't converging.  Exceeded maximum iteration number of ", maxiterations
       stop
    else
       store_iterations(n,mon) = store_iterations(n,mon) + 1
    endif

  end subroutine findpositionshell


!****************************************
! this function computes the electrostatic field on an atom,
! due to the other monomer, as well as the intra molecular drude oscillators
!***************************************
function field1(shell,monomer,n)
  use variables
  interface
     function screen(atom1,atom2,monomer1,monomer2,dist)
       use variables
       real*8::screen
       integer,intent(in)::atom1,atom2,monomer1,monomer2
       real*8,intent(in)::dist
     end function screen
     function  dscreen(atom1,atom2,monomer1,monomer2,xyz)
       use variables
       real*8,dimension(3)::dscreen
       integer,intent(in)::atom1,atom2,monomer1,monomer2
       real*8,dimension(3),intent(in)::xyz
     end function dscreen
  end interface

  real*8,dimension(3)::field1
  integer,intent(in)::shell,monomer,n
  real*8::dist
  real*8,dimension(3)::xyz,summ
  integer::i,atoms1,atoms2

  atoms1=size(atomtypemon1)
  atoms2=size(atomtypemon2)

if(monomer .eq. 1) then  ! monomer 1
  summ=0.0

  do i=1,atoms2
!!!!!!!!!!!!!!!field at shell due to atoms on other monomer
     xyz(:)=nxyzshell1(n,shell,:)-xyzmon2(n,i,:)
     dist=sqrt(dot_product(xyz,xyz))

     summ(:)=summ(:) + screen(shell,i,1,2,dist)*(chargeatoms2(i))*xyz(:)/(dist**3)
     summ(:)=summ(:) + (-1.)*dscreen(shell,i,1,2,xyz)*(chargeatoms2(i))/dist
  enddo

! ********* field due to intramolecular drude oscillator contributions
  do i=1,atoms1
     if(i .ne.shell) then
        xyz(:) = nxyzshell1(n,shell,:)-nxyzshell1(n,i,:)
        dist=sqrt(dot_product(xyz,xyz))

        summ(:)=summ(:) + screen(shell,i,1,1,dist)*(shellcharge1(i))*xyz(:)/(dist**3)
        summ(:)=summ(:) + (-1)*dscreen(shell,i,1,1,xyz)*(shellcharge1(i))/dist

        xyz(:)=nxyzshell1(n,shell,:)-xyzmon1(n,i,:)
        dist=sqrt(dot_product(xyz,xyz))

        summ(:)=summ(:) + screen(shell,i,1,1,dist)*(-shellcharge1(i))*xyz(:)/(dist**3)
        summ(:)=summ(:) + (-1)*dscreen(shell,i,1,1,xyz)*(-shellcharge1(i))/dist
     endif
  enddo

  field1(:)=summ(:)

else    ! monomer 2

  summ(:)=0.0

  do i=1,atoms1
!!!!!!!!!!!!!!!field at shell due to atoms on other monomer
     xyz(:)=nxyzshell2(n,shell,:)-xyzmon1(n,i,:)
     dist=sqrt(dot_product(xyz,xyz))

     summ(:)=summ(:) + screen(shell,i,1,2,dist)*(chargeatoms1(i))*xyz(:)/(dist**3)
     summ(:)=summ(:) + (-1.)*dscreen(shell,i,1,2,xyz)*(chargeatoms1(i))/dist
  enddo

! ********* field due to intramolecular drude oscillator contributions
  do i=1,atoms2
     if(i .ne.shell) then
        xyz(:) = nxyzshell2(n,shell,:)-nxyzshell2(n,i,:)
        dist=sqrt(dot_product(xyz,xyz))

        summ(:)=summ(:) + screen(shell,i,2,2,dist)*(shellcharge2(i))*xyz(:)/(dist**3)
        summ(:)=summ(:) + (-1)*dscreen(shell,i,2,2,xyz)*(shellcharge2(i))/dist

        xyz(:)=nxyzshell2(n,shell,:)-xyzmon2(n,i,:)
        dist=sqrt(dot_product(xyz,xyz))

        summ(:)=summ(:) + screen(shell,i,2,2,dist)*(-shellcharge2(i))*xyz(:)/(dist**3)
        summ(:)=summ(:) + (-1)*dscreen(shell,i,2,2,xyz)*(-shellcharge2(i))/dist
     endif
  enddo

  field1(:)=summ(:)

endif

end function field1



end module eq_drude
