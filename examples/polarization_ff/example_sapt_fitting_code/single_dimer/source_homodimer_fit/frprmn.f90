!*********************************************
! frprmn is the main conjugate gradient fitting subroutine
! used in the fitting program
! 
! the rest of the subroutines in this file are called by frprmn
! 
! all of these subroutines are adopted from numerical recipes
!*********************************************

SUBROUTINE frprmn(p,ftol,iter,fret)
  USE nrtype; USE nrutil, ONLY : nrerror
  USE nr, ONLY : linmin
  IMPLICIT NONE
  INTEGER(I4B), INTENT(OUT) :: iter
  REAL(SP), INTENT(IN) :: ftol
  REAL(SP), INTENT(OUT) :: fret
  real*8,dimension(:), INTENT(INOUT) :: p
  INTERFACE
     FUNCTION kaisq(p)
       use variables
       IMPLICIT NONE
       real*8,dimension(:), INTENT(IN) :: p
       REAL*8 :: kaisq
     END FUNCTION kaisq
     !BL
     FUNCTION dkaisq(p)
       use variables
       IMPLICIT NONE
       real*8,dimension(:), INTENT(IN) :: p
       REAL*8, DIMENSION(size(p)) :: dkaisq
     END FUNCTION dkaisq
  END INTERFACE
  INTEGER(I4B), PARAMETER :: ITMAX=10000
  REAL(SP), PARAMETER :: EPS=1.0e-10_sp
  INTEGER(I4B) :: its
  REAL(SP) :: dgg,fp,gam,gg
  REAL(SP), DIMENSION(size(p)) :: g,h,xi
  fp=kaisq(p)
  xi=dkaisq(p)
!!$        write(*,*) "kaisq",fp
!!$        write(*,*) "dkaisq"
  g=-xi
  h=g
  xi=h
  do its=1,ITMAX
     iter=its
     call linmin(p,xi,fret)
     if (2.0_sp*abs(fret-fp) <= ftol*(abs(fret)+abs(fp)+EPS)) RETURN
     fp=fret
     xi=dkaisq(p)
     gg=dot_product(g,g)
     !		dgg=dot_product(xi,xi)
     dgg=dot_product(xi+g,xi)
     if (gg == 0.0) RETURN
     gam=dgg/gg
     g=-xi
     h=g+gam*h
     xi=h
  end do
  call nrerror('frprmn: maximum iterations exceeded')
END SUBROUTINE frprmn


MODULE f1dim_mod
  USE nrtype
  INTEGER(I4B) :: ncom
  real*8,dimension(:),pointer::pcom
  REAL(SP), DIMENSION(:), POINTER :: xicom
CONTAINS
  !BL
  FUNCTION f1dim(x)
    IMPLICIT NONE
    REAL(SP), INTENT(IN) :: x
    REAL(SP) :: f1dim
    integer::i,j
    INTERFACE
       FUNCTION kaisq(x)
         use variables
         use interface
         real*8,dimension(:), INTENT(IN) :: x
         REAL*8 :: kaisq
       END FUNCTION kaisq
    END INTERFACE
    real*8,dimension(size(pcom))::xt
    xt=pcom+(x*xicom)
    f1dim=kaisq(xt)
    write(*,*) "param", xt
    write(*,*) f1dim
  END FUNCTION f1dim
END MODULE f1dim_mod


SUBROUTINE linmin(p,xi,fret)
  USE nrtype; USE nrutil, ONLY : assert_eq
  USE nr, ONLY : mnbrak,brent
  USE f1dim_mod
  IMPLICIT NONE
  REAL(SP), INTENT(OUT) :: fret
  real*8,dimension(:),target,intent(inout)::p
  REAL(SP), DIMENSION(:), TARGET, INTENT(INOUT) :: xi
  REAL(SP), PARAMETER :: TOL=1.0e-4_sp
  REAL(SP) :: ax,bx,fa,fb,fx,xmin,xx
  integer::i
  ncom=size(xi)
  pcom=>p
  xicom=>xi
  ax=0.0
  xx=1.0
  call mnbrak(ax,xx,bx,fa,fx,fb,f1dim)
  fret=brent(ax,xx,bx,f1dim,TOL,xmin)
  xi=xmin*xi
  p=p+xi
END SUBROUTINE linmin


SUBROUTINE mnbrak(ax,bx,cx,fa,fb,fc,func)
  USE nrtype; USE nrutil, ONLY : swap
  IMPLICIT NONE
  REAL(SP), INTENT(INOUT) :: ax,bx
  REAL(SP), INTENT(OUT) :: cx,fa,fb,fc
  INTERFACE
     FUNCTION func(x)
       USE nrtype
       IMPLICIT NONE
       REAL(SP), INTENT(IN) :: x
       REAL(SP) :: func
     END FUNCTION func
  END INTERFACE
  REAL(SP), PARAMETER :: GOLD=1.618034_sp,GLIMIT=100.0_sp,TINY=1.0e-20_sp
  REAL(SP) :: fu,q,r,u,ulim
  fa=func(ax)
  fb=func(bx)
  if (fb > fa) then
     call swap(ax,bx)
     call swap(fa,fb)
  end if
  cx=bx+GOLD*(bx-ax)
  fc=func(cx)
  do
     if (fb < fc) RETURN
     r=(bx-ax)*(fb-fc)
     q=(bx-cx)*(fb-fa)
     u=bx-((bx-cx)*q-(bx-ax)*r)/(2.0_sp*sign(max(abs(q-r),TINY),q-r))
     ulim=bx+GLIMIT*(cx-bx)
     if ((bx-u)*(u-cx) > 0.0) then
        fu=func(u)
        if (fu < fc) then
           ax=bx
           fa=fb
           bx=u
           fb=fu
           RETURN
        else if (fu > fb) then
           cx=u
           fc=fu
           RETURN
        end if
        u=cx+GOLD*(cx-bx)
        fu=func(u)
     else if ((cx-u)*(u-ulim) > 0.0) then
        fu=func(u)
        if (fu < fc) then
           bx=cx
           cx=u
           u=cx+GOLD*(cx-bx)
           call shft(fb,fc,fu,func(u))
        end if
     else if ((u-ulim)*(ulim-cx) >= 0.0) then
        u=ulim
        fu=func(u)
     else
        u=cx+GOLD*(cx-bx)
        fu=func(u)
     end if
     call shft(ax,bx,cx,u)
     call shft(fa,fb,fc,fu)
  end do
CONTAINS
  !BL
  SUBROUTINE shft(a,b,c,d)
    REAL(SP), INTENT(OUT) :: a
    REAL(SP), INTENT(INOUT) :: b,c
    REAL(SP), INTENT(IN) :: d
    a=b
    b=c
    c=d
  END SUBROUTINE shft
END SUBROUTINE mnbrak


FUNCTION brent(ax,bx,cx,func,tol,xmin)
  USE nrtype; USE nrutil, ONLY : nrerror
  IMPLICIT NONE
  REAL(SP), INTENT(IN) :: ax,bx,cx,tol
  REAL(SP), INTENT(OUT) :: xmin
  REAL(SP) :: brent
  INTERFACE
     FUNCTION func(x)
       USE nrtype
       IMPLICIT NONE
       REAL(SP), INTENT(IN) :: x
       REAL(SP) :: func
     END FUNCTION func
  END INTERFACE
  INTEGER(I4B), PARAMETER :: ITMAX=100
  REAL(SP), PARAMETER :: CGOLD=0.3819660_sp,ZEPS=1.0e-3_sp*epsilon(ax)
  INTEGER(I4B) :: iter
  REAL(SP) :: a,b,d,e,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm
  a=min(ax,cx)
  b=max(ax,cx)
  v=bx
  w=v
  x=v
  e=0.0
  fx=func(x)
  fv=fx
  fw=fx
  do iter=1,ITMAX
     xm=0.5_sp*(a+b)
     tol1=tol*abs(x)+ZEPS
     tol2=2.0_sp*tol1
     if (abs(x-xm) <= (tol2-0.5_sp*(b-a))) then
        xmin=x
        brent=fx
        RETURN
     end if
     if (abs(e) > tol1) then
        r=(x-w)*(fx-fv)
        q=(x-v)*(fx-fw)
        p=(x-v)*q-(x-w)*r
        q=2.0_sp*(q-r)
        if (q > 0.0) p=-p
        q=abs(q)
        etemp=e
        e=d
        if (abs(p) >= abs(0.5_sp*q*etemp) .or. &
             p <= q*(a-x) .or. p >= q*(b-x)) then
           e=merge(a-x,b-x, x >= xm )
           d=CGOLD*e
        else
           d=p/q
           u=x+d
           if (u-a < tol2 .or. b-u < tol2) d=sign(tol1,xm-x)
        end if
     else
        e=merge(a-x,b-x, x >= xm )
        d=CGOLD*e
     end if
     u=merge(x+d,x+sign(tol1,d), abs(d) >= tol1 )
     fu=func(u)
     if (fu <= fx) then
        if (u >= x) then
           a=x
        else
           b=x
        end if
        call shft(v,w,x,u)
        call shft(fv,fw,fx,fu)
     else
        if (u < x) then
           a=u
        else
           b=u
        end if
        if (fu <= fw .or. w == x) then
           v=w
           fv=fw
           w=u
           fw=fu
        else if (fu <= fv .or. v == x .or. v == w) then
           v=u
           fv=fu
        end if
     end if
  end do
  call nrerror('brent: exceed maximum iterations')
CONTAINS
  !BL
  SUBROUTINE shft(a,b,c,d)
    REAL(SP), INTENT(OUT) :: a
    REAL(SP), INTENT(INOUT) :: b,c
    REAL(SP), INTENT(IN) :: d
    a=b
    b=c
    c=d
  END SUBROUTINE shft
END FUNCTION brent
