SUBROUTINE kirchhoff(gamma,ms,Rload,volt,pot,mtot,mx,my)
!!
!!-----------------------------------------------------------------------
!!  solve the Kirchhoff law for a resistor network
!!
!!  The linear problem G*V = I is solved. G is a conductance matrix
!!  of the network, and I is the source current from the leads.
!!-----------------------------------------------------------------------
!!

!  USE 'mkl_lapack.fi'
  IMPLICIT NONE
  INCLUDE '/opt/intel/compilers_and_libraries_2020.1.216/mac/mkl/include/mkl_dss.fi'
!  TYPE (MKL_DSS_HANDLE) handle
  INTEGER, INTENT(IN) ::  mx,my,mtot
  REAL*8, INTENT(IN) :: gamma, Rload, volt
  REAL*8 , INTENT(IN), DIMENSION(mx,my) :: ms
  REAL*8 , INTENT(OUT) :: pot(mtot+1)
  REAL*8 :: resist(mx,my), gmat(mtot+1,mtot+1), iext(mtot+1)
  INTEGER i,j,k,k1,k2,i1,info
  REAL*8 gij !, gmat2(mtot+1,mtot+1)
!  REAL*8 VL
  INTEGER ipiv(mtot),ja(8*mtot),ia(8*mtot),idum(8*mtot)
  REAL*8 gcsr(8*mtot)

!  REAL time2(2),time3
  INTEGER job(8),error,nNonZeros
  INTEGER*8 handle

  iext = 0.0    ! source current vector
  gmat = 0.0    ! conductance matrix
  pot = 0.0

  do j=1,my
  do i=1,mx
    resist(i,j) = gamma*dmax1(1.d0,(ms(i,j)/gamma)**2)
!    resist(i,j) = gamma*dmax1(1.d0,(ms(i,j)/gamma))/(coef)
  enddo
  enddo

  do i=1,mx

    do j=1,my
      i1 = mod(i,mx)+1
      k1 = mx*(j-1)+i
      k2 = mx*(j-1)+i1
      gij = 2.0/(resist(i,j)+resist(i1,j))
      gmat(k2,k1) = gmat(k2,k1) - gij
      gmat(k1,k2) = gmat(k1,k2) - gij
      gmat(k1,k1) = gmat(k1,k1) + gij
      gmat(k2,k2) = gmat(k2,k2) + gij
    enddo

    do j=1,my-1
      k1 = mx*(j-1)+i
      k2 = mx*j+i
      gij = 2.0/(resist(i,j)+resist(i,j+1))
      gmat(k2,k1) = gmat(k2,k1) - gij
      gmat(k1,k2) = gmat(k1,k2) - gij
      gmat(k1,k1) = gmat(k1,k1) + gij
      gmat(k2,k2) = gmat(k2,k2) + gij
    enddo

    k = i    ! top row
    gij = 1.0/resist(i,1)
    gmat(k,k) = gmat(k,k) + gij
    iext(k) = iext(k) + volt*gij

    k = mx*(my-1)+i    ! bottom row
    gij = 1.0/resist(i,my)
    gmat(k,k) = gmat(k,k) + gij
    gmat(mtot+1,mtot+1) = gmat(mtot+1,mtot+1) + gij
    gmat(k,mtot+1) = gmat(k,mtot+1) - gij
    gmat(mtot+1,k) = gmat(mtot+1,k) - gij

  enddo

  gij = 1.0/Rload
  gmat(mtot+1,mtot+1) = gmat(mtot+1,mtot+1) + gij
  

  !call dgetrf(mtot,mtot,gmat,mtot,ipiv,info)
  !call dgetri(mtot,gmat,mtot,ipiv,work,3*mtot,info)
!  do i=1,mtot
!    pot(i) = 0.0
!    do j=1,mtot
!      pot(i) = pot(i) + gmat(i,j)*iext(j)
!    enddo
!!    write(6,*) i,pot(i)
!  enddo

!!
!!---------------------------------------------------------------------------
!!  LAPACK Linear Solver from Dense Matrix Form
!!---------------------------------------------------------------------------
!!
!  call dcopy(mtot+1,iext,1,pot,1)
!  call dposv('U',mtot+1,1,gmat,mtot+1,pot,mtot+1,info)
!!!
!!!---------------------------------------------------------------------------
!!!  MKL DSS Linear Solver from Sparse CSR Form
!!!---------------------------------------------------------------------------
!!!
!!!  nNonZeros = 3*mtot-(mx+my)+1+mx
        nNonZeros = 3*mtot+1
        job(1) = 0
        job(2) = 1
        job(3) = 1
        job(4) = 1
        job(5) = nNonZeros
        job(6) = 1
        job(7) = 0
        job(8) = 0
        CALL mkl_ddnscsr(job,mtot+1,mtot+1,gmat,mtot+1,gcsr,ja,ia,info)
!
!!  write(6,*) 'nNonZeros=',nNonZeros
!!  do i=1,mtot*mtot
!!    write(6,*) i,gcsr(i),ia(i),ja(i)
!!  enddo
!!  STOP
!!
!!!---------------------------------------------------------------------------
!!!  Solve via MKL_DSS
!!!---------------------------------------------------------------------------
!!!
        error = dss_create(handle, MKL_DSS_DEFAULTS)
        error = dss_define_structure( handle,MKL_DSS_SYMMETRIC,ia,mtot+1,mtot+1,ja,nNonZeros)
        error = dss_reorder( handle, MKL_DSS_AUTO_ORDER, idum)
        error = dss_factor_real( handle, MKL_DSS_DEFAULTS,gcsr)
        error = dss_solve_real( handle, MKL_DSS_DEFAULTS, iext, 1, pot)
        error = dss_delete( handle, MKL_DSS_DEFAULTS )
!  write(6,*) pot(mtot+1),Rload,pot(mtot+1)/Rload

!  VL = pot(mtot+1)
!  do j=1,my
!    do i=1,mx
!      pot(mx*(j-1)+i) = volt - (volt-VL)/(my+1.0)*(j-0.5)
!    enddo
!  enddo

!  CALL efield(mx,my,dx,dy,pot,Ex,Ey)

  RETURN
END SUBROUTINE
!
!!
!!-----------------------------------------------------------------------
!!	  gradient of the electric potential
!!-----------------------------------------------------------------------
!!
!
!subroutine efield(mx,my,dx,dy,pot,Ex,Ey)
!  IMPLICIT NONE
!  INTEGER,INTENT(IN) :: mx,my
!  INTEGER :: mtot
!  INTEGER :: i,j,k1,k2,i1,i2
!  REAL*8, INTENT(IN) :: dx,dy,pot(*)
!  REAL*8, INTENT(OUT) :: Ex(mx,my),Ey(mx,my)
!
!  do i=1,mx
!  do j=1,my
!    i1 = mod(i-2+mx,mx)+1
!    i2 = mod(i,mx)+1
!    k2 = mx*(j-1)+i2
!    k1 = mx*(j-1)+i1
!    Ex(i,j) = 0.5*(pot(k2)-pot(k1))/dx
!    if (j.eq.1) then
!      k2 = mx*j+i
!      k1 = mx*(j-1)+i
!      Ey(i,j) = (pot(k2)-pot(k1))/dy
!    else if (j.eq.my) then
!      k2 = mx*(j-1)+i
!      k1 = mx*(j-2)+i
!      Ey(i,j) = (pot(k2)-pot(k1))/dy
!    else
!      k2 = mx*j+i
!      k1 = mx*(j-2)+i
!      Ey(i,j) = 0.5*(pot(k2)-pot(k1))/dy
!    endif
!  enddo
!  enddo
!  return
!end
