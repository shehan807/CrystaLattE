!**************************************
! this is just an interface module that we use
!**************************************

module interface

interface
   function Inducenergy(n)
   use variables
   real*8::Inducenergy
   integer,intent(in)::n
   end function Inducenergy
end interface
interface
  FUNCTION elecenergy(n)
  USE variables
  integer,INTENT(in)::n
  REAL*8::elecenergy
  end function elecenergy
end interface
interface
   function dispenergy(n)
   use variables
   real*8::dispenergy
   integer,intent(in)::n
   end function dispenergy
end interface
interface
   function exchenergy(n)
   use variables
   real*8::exchenergy
   integer,intent(in)::n
   end function exchenergy
end interface
interface
   function dhfenergy(n)
   use variables
   real*8::dhfenergy
   integer,intent(in)::n
   end function dhfenergy
end interface
interface
function dexchenergy(atom1,atom2,n)
  use variables
  real*8::dexchenergy
  integer,intent(in)::atom1,atom2,n
end function dexchenergy
end interface
interface
function delecenergy(atom1,atom2,n)
  use variables
  real*8::delecenergy
  integer,intent(in)::atom1,atom2,n
end function delecenergy
end interface
interface
 function dinducenergy(atom1,atom2,n)
   use variables
   real*8::dinducenergy
   integer,intent(in)::atom1,atom2,n
end function dinducenergy
end interface
interface
 function ddhf(atom1,atom2,n)
   use variables
   real*8::ddhf
   integer,intent(in)::atom1,atom2,n
end function ddhf
end interface


end module interface
