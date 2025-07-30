MODULE routines

  !****************************************
  ! this module contains miscellaneous subroutines that are used in the program
  !****************************************

CONTAINS


  !***********************************************************
  ! this subroutine gets the number of atoms for each molecules, as well as the number
  ! of datapoints in the file.  The number of data points is recognized as the number of lines that
  ! start with an integer, divided by two
  !***********************************************************
  SUBROUTINE getnumatoms(inpfile,datpts,atoms1,atoms2)
    CHARACTER(*),INTENT(in)::inpfile
    CHARACTER(50)::line, entry1
    INTEGER,INTENT(out)::datpts,atoms1,atoms2
    INTEGER::i,inputstatus,nargs
    integer,parameter :: max_param=20
    character(50),dimension(max_param)::args

    OPEN(unit=4,file=inpfile,status='old')


    ! geometry block
    read(4,*) atoms1
    do i=1,atoms1
       read(4,*) line
    enddo
    read(4,*) atoms2
    do i=1,atoms2
       read(4,*) line
    enddo

    datpts=2

    ! now read until end of file, counting the number of integer lines, and assume number of 
    ! data points is this number divided by 2

    do 
       READ(4,'(A)',Iostat=inputstatus) line
       IF(inputstatus < 0) EXIT       

       call parse(line," ",args,nargs)
       ! see if first entry is an integer
       entry1=args(1)
       if (index("0123456789",entry1(1:1)) .ne. 0 ) then
          ! this is an integer!
          datpts = datpts + 1
       endif
    enddo

    datpts = datpts / 2

    CLOSE(4)

  END SUBROUTINE getnumatoms



  !*****************************************************
  ! This subroutine reads the input file that contains all the data points to be fit
  ! and gets the energies and configurations for all these points
  !*****************************************************
  SUBROUTINE getdata(inpfile,datpts,atoms1,atoms2,E1pol,E1exch,E2ind,E2indexch,E2disp,E2dispexch,E1tot,E2tot,Etot,E2ind12,&
                  E2ind21,dhf,xyzmon1,xyzmon2,atomtypemon1,atomtypemon2)
    IMPLICIT NONE
    CHARACTER(*),INTENT(in)::inpfile
    INTEGER,INTENT(in)::datpts
    INTEGER,INTENT(inout)::atoms1,atoms2
    REAL*8,DIMENSION(:),INTENT(out)::E1pol,E1exch,E2ind,E2indexch,E2disp,E2dispexch,E1tot,E2tot,Etot,E2ind21,E2ind12,dhf
    REAL*8,DIMENSION(:,:,:),INTENT(out)::xyzmon1,xyzmon2
    CHARACTER(2),DIMENSION(:),INTENT(out)::atomtypemon1,atomtypemon2
    CHARACTER(50)::line
    CHARACTER(20)::junk
    INTEGER::ind,i,j,count,inputstatus
    E2disp=0.
    E2dispexch=0.
    dhf=0.

    OPEN(unit=4,file=inpfile,status='old')
    DO
       READ(4,'(A)',Iostat=inputstatus) line
       IF(inputstatus < 0) EXIT
       !        ind=INDEX(line,'E2disp(unc)')
       ind=INDEX(line,'E2ind(unc)')
       IF(ind .NE. 0) EXIT
    ENDDO

    count=1
    DO WHILE(count < 6 )
       !       DO WHILE(count < 9 )
       BACKSPACE(4)
       count=count+1
    ENDDO
    count=1
    DO WHILE(count < (atoms1+atoms2+2))
       BACKSPACE(4)
       count=count+1
    ENDDO
    count=1

    DO j=1,datpts
       READ(4,*) atoms1
       DO i=1,atoms1
          READ(4,*) atomtypemon1(i),xyzmon1(j,i,1),xyzmon1(j,i,2),xyzmon1(j,i,3)
       ENDDO
       READ(4,*) atoms2
       DO i=1,atoms2
          READ(4,*) atomtypemon2(i),xyzmon2(j,i,1),xyzmon2(j,i,2),xyzmon2(j,i,3)
       ENDDO
       READ(4,*) junk,E1pol(j)
       READ(4,*) junk,E1exch(j)
       READ(4,'(A)') line
       READ(4,'(A)') line
       READ(4,*) junk,E2ind(j)
       READ(4,*) junk,E2indexch(j)
       READ(4,'(A)') line
       READ(4,*) junk,E2disp(j)
       READ(4,'(A)') line
       READ(4,*) junk,E2dispexch(j)
       READ(4,*) junk,E1tot(j)
       READ(4,*) junk,E2tot(j)
       READ(4,*) junk,Etot(j)
       READ(4,*) junk,E2ind21(j)
       READ(4,*) junk,E2ind12(j)
       READ(4,'(A)') line
       READ(4,'(A)') line
       READ(4,*) junk,dhf(j)
       IF(j < datpts) THEN
          READ(4,'(A)') line
       ENDIF

    ENDDO
    CLOSE(4)
  END SUBROUTINE getdata


  !*****************************************
  ! this subroutine determines the number of independent
  ! atom types to fit in each molecule, by the number of 
  ! unique atom names
  !*****************************************
  SUBROUTINE numindatoms(atms1,atms2)
    use variables
    INTEGER,INTENT(out)::atms1,atms2
    CHARACTER(2)::value
    INTEGER::count1,count2,atoms1,atoms2,i,j
    CHARACTER(3),DIMENSION(:),ALLOCATABLE::typemon1,typemon2
    atoms1=SIZE(atomtypemon1)
    atoms2=SIZE(atomtypemon2)

    ALLOCATE(typemon1(atoms1),typemon2(atoms2))

    DO i=1,atoms1
       WRITE(typemon1(i),'(A2)') atomtypemon1(i)
    ENDDO
    DO j=1,atoms2
       WRITE(typemon2(j),'(A2)') atomtypemon2(j)
    ENDDO
    count1=0
    count2=0
    DO i=1,atoms1
       IF(i < atoms1) THEN 
          DO j=i+1,atoms1
             IF(typemon1(i) .EQ. typemon1(j)) THEN
                count1=count1+1
                WRITE(value,'(I2)') j
                typemon1(j)='z'//value
             ENDIF
          ENDDO
       ENDIF
    ENDDO
    DO i=1,atoms2
       IF(i < atoms2) THEN
          DO j=i+1,atoms2
             IF(typemon2(i) .EQ. typemon2(j)) THEN
                count2=count2+1
                WRITE(value,'(I2)') j
                typemon2(j)='z'//value
             ENDIF
          ENDDO
       ENDIF
    ENDDO


    ! if hard constraints, subtract those atom types
    if ( hard_constraints .eq. "yes" ) then
       count2=count2+size(hard_cons_type)
       count1=count1+size(hard_cons_type)
    endif

    atms1=atoms1-count1
    atms2=atoms2-count2

  END SUBROUTINE numindatoms



  !************************************
  ! this subroutine constructs the atmtypemon1(2) array that
  ! contains a list of all the unique atom types to fit
  ! in each molecule
  !************************************
  subroutine collapseatomtype
    use variables
    INTEGER::count,countt,atoms1,atoms2,atms1,atms2,i,j,k

    atoms1=SIZE(atomtypemon1);atoms2=SIZE(atomtypemon2)
    atms1=SIZE(atmtypemon1);atms2=SIZE(atmtypemon2)


    count=0
    countt=0

    DO i=1,atms1
       count=countt
       DO j=i+count,atoms1
!!! check if this is a hard constraint
          if ( hard_constraints .eq. "yes" ) then
             do k=1,size(hard_cons_type)
                if(atomtypemon1(j).eq.hard_cons_type(k)) then
                   countt=countt+1
                   go to 200 
                endif
             enddo
          endif
          DO k=1,j-1
             IF(atomtypemon1(k).EQ.atomtypemon1(j)) THEN
                countt=countt+1
                go to 200 
             ENDIF
          ENDDO
          atmtypemon1(i)=atomtypemon1(j)
          go to 201
200       CONTINUE
       ENDDO
201    CONTINUE
    ENDDO

    count=0
    countt=0

    DO i=1,atms2
       count=countt
       DO j=i+count,atoms2
!!! check if this is a hard constraint
          if ( hard_constraints .eq. "yes" ) then
             do k=1,size(hard_cons_type)
                if(atomtypemon2(j).eq.hard_cons_type(k)) then
                   countt=countt+1
                   go to 202 
                endif
             enddo
          endif
          DO k=1,j-1
             IF(atomtypemon2(k).EQ.atomtypemon2(j)) THEN
                countt=countt+1
                go to 202
             ENDIF
          ENDDO
          atmtypemon2(i)=atomtypemon2(j)
          go to 203
202       CONTINUE
       ENDDO
203    CONTINUE
    ENDDO


  END SUBROUTINE collapseatomtype



  !***********************************************
  ! this subroutine reads in force field parameters, as well as any 
  ! hard constraints
  !
  !**********************************************
  subroutine getparameters(param_file)
    use variables
    CHARACTER(*),INTENT(in)::param_file
    INTEGER::i,j,k,atoms1,atoms2,inputstatus,atom,site1,site2,ind=0,flag
    CHARACTER(50)::line
    character(5)::junk1
    character(len(atomtypemon1(1)))::test1
    REAL*8::temp,junk,exch,e1,e2,temp_pen(4)
    real*8,dimension(:),allocatable::exp_molecule

    atoms1=size(chargeatoms1)
    atoms2=SIZE(chargeatoms2)

    OPEN(unit=7,file=param_file,status='old')


    !********************************* hard constraints******************************
    if ( hard_constraints .eq. "yes" ) then
       read(7,*) atom 
       allocate(hard_cons_type(atom),hard_cons_param(atom,4)) 
       do i=1,atom
          read(7,*) hard_cons_type(i),(hard_cons_param(i,j),j=1,4)
       enddo

       ! fill in hard constraints for both molecules
       do i=1,atoms2
          do j=1,size(hard_cons_type)
             if ( atomtypemon2(i) .eq. hard_cons_type(j) ) then
                Exchatom2(i) = hard_cons_param(j,1)
                Elecatom2(i) = hard_cons_param(j,2)
                Inducatom2(i) = hard_cons_param(j,3)
                dhf2(i) = hard_cons_param(j,4)
             endif
          enddo
       enddo
       do i=1,atoms1
          do j=1,size(hard_cons_type)
             if ( atomtypemon1(i) .eq. hard_cons_type(j) ) then
                Exchatom1(i) = hard_cons_param(j,1)
                Elecatom1(i) = hard_cons_param(j,2)
                Inducatom1(i) = hard_cons_param(j,3)
                dhf1(i) = hard_cons_param(j,4)
             endif
          enddo
       enddo

    endif


!!!!!!!!!!!!!!!!!!!read exchange exponents, create cross terms
    allocate(exp_molecule(atoms1))
    READ(7,'(A)') line
    DO i=1,atoms1
       READ(7,*) junk1,exp_molecule(i)
    ENDDO
    write(*,*) ""
    write(*,*) "**************************************************************"
    write(*,*) "explicitly creating cross-term exponents using combination rule"
    write(*,*) "exp_tot = (exp1 + exp2 ) * exp1*exp2 / (exp1^2 + exp2^2 )"
    write(*,*) "**************************************************************"  
    write(*,*) "" 
    do i=1,atoms1
       e1 = exp_molecule(i)
       do j=1,atoms2
          e2=exp_molecule(j)
          exponents(i,j) = (e1+e2) * (e1*e2)/(e1**2+e2**2)
       enddo
    enddo


!!!!!!!!!!!!!!!!!!!!! read dispersion coefficients, C6-C12, create cross terms
    READ(7,'(A)') line
    DO i=1,atoms1
       READ(7,*) junk1, (Cn_cross(i,i,j),j=1,4)
    ENDDO
    write(*,*) ""
    write(*,*) "**************************************************************"
    write(*,*) "explicitly creating cross-term dispersion coeffs using combination rule"
    write(*,*) "Cn(AB) = sqrt(Cn(A) * Cn(B))"
    write(*,*) "**************************************************************"  
    write(*,*) "" 
    do i=1,atoms1
       do j=1,atoms2
          do k=1,4
             Cn_cross(i,j,k) = dsqrt( Cn_cross(i,i,k) * Cn_cross(j,j,k) )
          enddo
       enddo
    enddo




!!!!!!!!!!!!!!!!!!!!!!read charges
    READ(7,'(A)') line
    DO i=1,atoms1
       READ(7,*) atom, chargeatoms1(i)
    ENDDO
    chargeatoms2=chargeatoms1


!!!!!!!!!!!!!!!!!!!!!!! read drude charges and springcon
    READ(7,'(A)') line
    DO i=1,atoms1
       READ(7,*) atom, shellcharge1(i)
    ENDDO
    READ(7,'(A)') line
    READ(7,*) springcon1      

    close(7)



  END SUBROUTINE getparameters




  !******************************************
  ! this subroutine writes the results of the fitting program
  ! to an output file
  !******************************************
  subroutine writeoutput(out_file,rms)
    use variables
    character(*),intent(in) :: out_file
    real*8,dimension(:),intent(in)::rms
    integer:: i,j,atoms1,atoms2,datpts

    atoms1=size(atomtypemon1);atoms2=size(atomtypemon2);
    datpts=size(xyzmon1(:,1,1))

    open(unit=7,file=out_file,status='new')

    ! write all parameters, then energy for all components

    write(7,*) "-------------------------------------------------------------------"
    write(7,*) "--------------       exchange         -----------------------------"
    write(7,*) "-------------------------------------------------------------------"
    write(7,*) ""
    write(7,*) "Exchange buckingham coefficients"
    do i=1,atoms1
       write(7,*) atomtypemon1(i),Exchatom1(i)
    enddo
    write(7,*) "Exch energy: SAPT , FIT"
    do i=1,datpts
       write(7,*) Ecomp(1,i),Eout(1,i)
    enddo
    write(7,*) "rms for this component is", rms(1)

    write(7,*) "-------------------------------------------------------------------"
    write(7,*) "--------------       electrostatic    -----------------------------"
    write(7,*) "-------------------------------------------------------------------"
    write(7,*) ""
    write(7,*) "Electrostatic buckingham coefficients"
    do i=1,atoms1
       write(7,*) atomtypemon1(i),Elecatom1(i)
    enddo
    write(7,*) "Elec energy: SAPT , FIT"
    do i=1,datpts
       write(7,*) Ecomp(2,i),Eout(2,i)
    enddo
    write(7,*) "rms for this component is", rms(2)

    write(7,*) "-------------------------------------------------------------------"
    write(7,*) "--------------       induction        -----------------------------"
    write(7,*) "-------------------------------------------------------------------"
    write(7,*) ""
    write(7,*) "Induction buckingham coefficients"
    do i=1,atoms1
       write(7,*) atomtypemon1(i),Inducatom1(i)
    enddo
    write(7,*) "Induc energy: SAPT , FIT"
    do i=1,datpts
       write(7,*) Ecomp(3,i),Eout(3,i)
    enddo
    write(7,*) "rms for this component is", rms(3)


    write(7,*) "-------------------------------------------------------------------"
    write(7,*) "--------------       dhf              -----------------------------"
    write(7,*) "-------------------------------------------------------------------"
    write(7,*) ""
    Select Case(include_high_order_drude_dhf_fit)
    Case("yes")
       write(7,*) "We have included higher order drude oscillator contributions in the dhf fit.  These contributions &
    are energy differences between fully self-consistent optimized drude positions and drude responses to only the static charges"
    End Select
    write(7,*) "dhf buckingham coefficients"
    do i=1,atoms1
       write(7,*) atomtypemon1(i),dhf1(i)
    enddo
    write(7,*) "dhf energy: SAPT , FIT"
    do i=1,datpts
       write(7,*) Ecomp(4,i),Eout(4,i)
    enddo
    write(7,*) "rms for this component is", rms(4)



    write(7,*) "-------------------------------------------------------------------"
    write(7,*) "--------------       dispersion       -----------------------------"
    write(7,*) "-------------------------------------------------------------------"
    write(7,*) ""
    write(7,*) "Cn coefficients (C6,C8,C10,C12)"
    do i=1,atoms1
       write(7,*) atomtypemon1(i),(Cn_cross(i,i,j),j=1,4)
    enddo

    write(7,*) "disp energy: SAPT , FIT"
    do i=1,datpts
       write(7,*) Ecomp(5,i),Eout(5,i)
    enddo
    write(7,*) "rms for this component is", rms(5)


    write(7,*) "-------------------------------------------------------------------"
    write(7,*) "--------------      total energy       ----------------------------"
    write(7,*) "-------------------------------------------------------------------"
    write(7,*) ""
      do i=1,datpts
       write(7,*) Etot(i),Eout(6,i)
    enddo
    write(7,*) "rms for this component is", rms(6)

    close(7)

  end subroutine writeoutput



  !**********************************************************
  ! this subroutine initializes data
  !*********************************************************
  subroutine initialize_data(data_file, datpts,atoms1,atoms2)
    use variables
    character(*),intent(in) :: data_file
    integer,intent(out) :: datpts,atoms1,atoms2

    CALL getnumatoms(data_file,datpts,atoms1,atoms2)

    !  allocate everything
       ALLOCATE(atomtypemon1(atoms1),atomtypemon2(atoms2),chargeatoms1(atoms1),chargeatoms2(atoms2),shellcharge1(atoms1),&
       shellcharge2(atoms2),exponents(atoms1,atoms2),Exoverlap(atoms1,atoms2),Elecoverlap(atoms1,atoms2),&
       Inducoverlap(atoms1,atoms2),dhfoverlap(atoms1,atoms2),Elecatom1(atoms1),Elecatom2(atoms2),Inducatom1(atoms1),&
       Inducatom2(atoms2),Exchatom1(atoms1),Exchatom2(atoms2),dhf1(atoms1),dhf2(atoms2),Cn_cross(atoms1,atoms2,4))

    ALLOCATE(xyzmon1(datpts,atoms1,3),xyzmon2(datpts,atoms2,3), E1pol(datpts),E1exch(datpts),E2ind(datpts),E2indexch(datpts),&
       E2disp(datpts),E2dispexch(datpts),Ecomp(6,datpts),Eout(6,datpts),E1tot(datpts),E2tot(datpts),Etot(datpts),E2ind12(datpts),&
       E2ind21(datpts),dhf(datpts),nxyzshell1(datpts,atoms1,3),nxyzshell2(datpts,atoms2,3),Etot_noind(datpts))

    CALL getdata(data_file,datpts,atoms1,atoms2,E1pol,E1exch,E2ind,E2indexch,E2disp,E2dispexch,E1tot,E2tot,Etot,E2ind12,E2ind21,&
            dhf,xyzmon1,xyzmon2,atomtypemon1,atomtypemon2)


write(*,*) ""
write(*,*) "*****************************************************"
write(*,*) "input energies are assumed to be in mH, and are converted"
write(*,*) "to Hartree"
write(*,*) "*****************************************************"
write(*,*) ""

!!!!!!!!!convert to hartree
    E1pol(:)=E1pol(:)/1000.
    E1exch(:)=E1exch(:)/1000.
    E2ind(:)=E2ind(:)/1000.
    E2indexch(:)=E2indexch(:)/1000.
    E2disp(:)=E2disp(:)/1000.
    E2dispexch(:)=E2dispexch(:)/1000.
    E1tot(:)=E1tot(:)/1000.
    E2tot(:)=E2tot(:)/1000.
    Etot(:)=Etot(:)/1000.
    E2ind21(:)=E2ind21(:)/1000.
    E2ind12(:)=E2ind12(:)/1000.
    dhf(:)=dhf(:)/1000.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!add dhf to total energy
    Etot=Etot+dhf
    Etot_noind = Etot - E2ind - E2indexch


  end subroutine initialize_data




!******************************************
! this subroutine distributes the fitting parameters among Elecatom2,Inducatom2,Exchatom2,dhf2
!*****************************************
subroutine outparam(p0)
  use variables
  real*8,dimension(:),intent(in)::p0

  integer::i,j,atms2,atoms2,atoms1,atms1

  atoms2=size(atomtypemon2)
  atms2=size(atmtypemon2)
  atoms1=size(atomtypemon1)
  atms1=size(atmtypemon1)

  Select Case(component_fit)
  Case(1)
     do i=1,atms2
        do j=1,atoms2
           if (atomtypemon2(j) .eq. atmtypemon2(i)) then
              Exchatom2(j) = p0(i)
           endif
        enddo
     enddo
       Exchatom1=Exchatom2
  Case(2)
     do i=1,atms2
        do j=1,atoms2
           if (atomtypemon2(j) .eq. atmtypemon2(i)) then
              Elecatom2(j) = p0(i)
           endif
        enddo
     enddo
     Elecatom1=Elecatom2
  Case(3)
     do i=1,atms2
        do j=1,atoms2
           if (atomtypemon2(j) .eq. atmtypemon2(i)) then
              Inducatom2(j) = p0(i)
           endif
        enddo
     enddo
     Inducatom1=Inducatom2
  Case(4)
     do i=1,atms2
        do j=1,atoms2
           if (atomtypemon2(j) .eq. atmtypemon2(i)) then
              dhf2(j) = p0(i)
           endif
        enddo
     enddo
     dhf1=dhf2

  case default
     stop "component_fit setting not recognized in outparam"

  End Select

end subroutine outparam


  !****************************************
  ! this subroutine uses combination rule to generate pair parameters
  !****************************************
subroutine combinegeom(paramA1,paramA2,paramAA)
  use variables
  real*8,dimension(:),intent(in)::paramA1
  real*8,dimension(:),intent(in)::paramA2
  real*8,dimension(:,:),intent(out)::paramAA
  integer::i,j,atoms1,atoms2
  real*8:: A1,A2


  atoms1=size(paramA1);atoms2=size(paramA2(:))

  do i=1,atoms1
     A1 = paramA1(i)
     do j=1,atoms2
        A2 = paramA2(j)
        ! if same atom, don't use combination rule as this might mess up sign
        if (atomtypemon1(i) .eq. atomtypemon2(j) ) then
           paramAA(i,j) = abs(A1)
        else
              paramAA(i,j)=abs(A1*A2)**.5
!           if( A1 * A2 > 0.) then
!              paramAA(i,j)=(A1*A2)**.5
!           else
!              paramAA(i,j)=-(abs(A1*A2))**.5  
!           endif
        endif
     enddo
  enddo

end subroutine combinegeom






  !******************************************************
  !        Here are some string manipulation routines 
  !        written by Dr. George Benthien and taken from
  !        http://www.gbenthien.net/strings/index.html
  !******************************************************

  subroutine parse(str,delims,args,nargs)

    ! Parses the string 'str' into arguments args(1), ..., args(nargs) based on
    ! the delimiters contained in the string 'delims'. Preceding a delimiter in
    ! 'str' by a backslash (\) makes this particular instance not a delimiter.
    ! The integer output variable nargs contains the number of arguments found.

    character(len=*) :: str,delims
    character(len=len_trim(str)) :: strsav
    character(len=*),dimension(:) :: args
    integer,intent(out) ::nargs
    integer :: i,k,na,lenstr


    strsav=str
    call compact(str)
    na=size(args)
    do i=1,na
       args(i)=' '
    end do
    nargs=0
    lenstr=len_trim(str)
    if(lenstr==0) return
    k=0

    do
       if(len_trim(str) == 0) exit
       nargs=nargs+1
       call split(str,delims,args(nargs))
       call removebksl(args(nargs))
    end do
    str=strsav

  end subroutine parse



  subroutine compact(str)

    ! Converts multiple spaces and tabs to single spaces; deletes control characters;
    ! removes initial spaces.

    character(len=*):: str
    character(len=1):: ch
    character(len=len_trim(str)):: outstr
    integer :: i,k,ich,isp,lenstr

    str=adjustl(str)
    lenstr=len_trim(str)
    outstr=' '
    isp=0
    k=0

    do i=1,lenstr
       ch=str(i:i)
       ich=iachar(ch)

       select case(ich)

       case(9,32)     ! space or tab character
          if(isp==0) then
             k=k+1
             outstr(k:k)=' '
          end if
          isp=1

       case(33:)      ! not a space, quote, or control character
          k=k+1
          outstr(k:k)=ch
          isp=0

       end select

    end do

    str=adjustl(outstr)

  end subroutine compact


  subroutine split(str,delims,before,sep)

    ! Routine finds the first instance of a character from 'delims' in the
    ! the string 'str'. The characters before the found delimiter are
    ! output in 'before'. The characters after the found delimiter are
    ! output in 'str'. The optional output character 'sep' contains the 
    ! found delimiter. A delimiter in 'str' is treated like an ordinary 
    ! character if it is preceded by a backslash (\). If the backslash 
    ! character is desired in 'str', then precede it with another backslash.

    character(len=*) :: str,delims,before
    character,optional :: sep
    logical :: pres
    character :: ch,cha
    integer:: i,k,lenstr,ibsl,ipos,iposa

    pres=present(sep)
    str=adjustl(str)
    call compact(str)
    lenstr=len_trim(str)
    if(lenstr == 0) return        ! string str is empty
    k=0
    ibsl=0                        ! backslash initially inactive
    before=' '
    do i=1,lenstr
       ch=str(i:i)
       if(ibsl == 1) then          ! backslash active
          k=k+1
          before(k:k)=ch
          ibsl=0
          cycle
       end if
       if(ch == '\') then          ! backslash with backslash inactive
          k=k+1
          before(k:k)=ch
          ibsl=1
          cycle
       end if
       ipos=index(delims,ch)         
       if(ipos == 0) then          ! character is not a delimiter
          k=k+1
          before(k:k)=ch
          cycle
       end if
       if(ch /= ' ') then          ! character is a delimiter that is not a space
          str=str(i+1:)
          if(pres) sep=ch
          exit
       end if
       cha=str(i+1:i+1)            ! character is a space delimiter
       iposa=index(delims,cha)
       if(iposa > 0) then          ! next character is a delimiter
          str=str(i+2:)
          if(pres) sep=cha
          exit
       else
          str=str(i+1:)
          if(pres) sep=ch
          exit
       end if
    end do
    if(i >= lenstr) str=''
    str=adjustl(str)              ! remove initial spaces
    return

  end subroutine split

  !**********************************************************************

  subroutine removebksl(str)

    ! Removes backslash (\) characters. Double backslashes (\\) are replaced
    ! by a single backslash.

    character(len=*):: str
    character(len=1):: ch
    character(len=len_trim(str))::outstr
    integer :: i,k,ibsl,lenstr

    str=adjustl(str)
    lenstr=len_trim(str)
    outstr=' '
    k=0
    ibsl=0                        ! backslash initially inactive

    do i=1,lenstr
       ch=str(i:i)
       if(ibsl == 1) then          ! backslash active
          k=k+1
          outstr(k:k)=ch
          ibsl=0
          cycle
       end if
  if(ch == '\') then          ! backslash with backslash inactive
   ibsl=1
   cycle
  end if
  k=k+1
  outstr(k:k)=ch              ! non-backslash with backslash inactive
end do

str=adjustl(outstr)

end subroutine removebksl

!**********************************************************************

       

END MODULE routines
