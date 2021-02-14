!!
!!-----------------------------------------------------------------------
!!  thermalize through nsteps
!!-----------------------------------------------------------------------
!!
SUBROUTINE hb_loop(phi,Teff,f,ms,mfphi,icount,rate,nstep,dx,dy,idummy,Tc,Tbath,dphi,r0,v0,gamma,g2,g4,g6,mx,my)
!! Note : comment !f2py intent(in,out) allows python to read inout parameters
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nstep,mx,my
  REAL*8, INTENT(IN) :: r0,dphi,dx,dy,Tbath,Tc,v0(mx,my),gamma,g2,g4,g6
  LOGICAL, INTENT(IN) :: mfphi
  REAL*8, INTENT(INOUT) :: ms(mx,my)
  !f2py intent(in,out), DIMENSION(mx,my) :: ms
  INTEGER, INTENT(INOUT) :: icount(mx,my)
  !f2py intent(in,out), DIMENSION(mx,my) :: icount
  REAL*8, INTENT(OUT) :: rate
  REAL*8, INTENT(INOUT), DIMENSION(mx,my) :: phi
  !f2py intent(in,out), DIMENSION(mx,my) :: phi
  REAL*8, INTENT(INOUT), DIMENSION(mx,my) :: Teff
  !f2py intent(in,out), DIMENSION(mx,my) :: Teff
  REAL*8, DIMENSION(mx,my) :: phit, Teft
  REAL*8, INTENT(INOUT) :: f
  !f2py intent(in,out) :: f
  INTEGER :: iflip, ncount
  REAL*8  :: ran2,dfree,Teff1,phi1,df,T2!, r1,r2
  INTEGER :: n,i,j,idummy
!  CHARACTER(LEN = 20) :: phidis
  phit = 0.0
  Teft = 0.0
  ncount = 0
  iflip = 0
  DO n=1,nstep
  DO i = 1,mx,1
  DO j = 1,my,1
    !!!!! ---- suggest an update and compute the free-energy change
!    i = INT(mx*ran2(idummy))+1
!    j = INT(my*ran2(idummy))+1

    phi1 = phi(i,j) + dphi*2.0*(ran2(idummy)-0.5)
    IF (mfphi) THEN
      Teff1 = Teff(i,j) ! mean field approx to effective temperature
    ELSE
      T2 = (gamma**2+phi(i,j)**2)/(gamma**2+phi1**2)*(Teff(i,j)**2-Tbath**2)
      Teff1 = sqrt(T2+Tbath**2)
    ENDIF
    !!!! ---- compute difference in free energy
    df = dfree(mx,my,Tc,r0,g2,g4,g6,i,j,phi,phi1,Teff,Teff1,v0,dx,dy)

    icount(i,j) = icount(i,j) + 1
    ncount = ncount + 1
    !!!! ---- test if we accept the update
    IF(ran2(idummy) .LT. exp(-df/Tbath)) THEN
      phit(i,j) = phi1
      Teft(i,j) = Teff1
!      f = f + df
      iflip = iflip + 1
    ELSE
      phit(i,j) = phi(i,j)
      Teft(i,j) = Teff(i,j)
    END IF
    ms(i,j) = ms(i,j) + phi(i,j)
  END DO
  END DO
  
  DO i = 1,mx,1
  DO j = 1,my,1
    phi(i,j) = phit(i,j)
    Teff(i,j) = Teft(i,j)
  END DO
  END DO
  
  END DO 
  rate = float(iflip)/float(ncount)
END

FUNCTION ran2(idum)
  INTEGER idum,IM1,IM2,IMM1,IA1,IA2,IQ1,IQ2,IR1,IR2,NTAB,NDIV
  REAL*8 ran2,AM,EPS,RNMX
  PARAMETER (IM1=2147483563,IM2=2147483399,AM=1./IM1,IMM1=IM1-1, &
           &   IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,IR2=3791,NTAB=32,NDIV=1+IMM1/NTAB,EPS=1.2e-7,RNMX=1.-EPS)
  INTEGER idum2,j,k,iv(NTAB),iy
  SAVE iv,iy,idum2
  DATA idum2/123456789/, iv/NTAB*0/, iy/0/
  IF (idum.le.0) then
        idum=max(-idum,1)
        idum2=idum
        DO j=NTAB+8,1,-1
           k=idum/IQ1
           idum=IA1*(idum-k*IQ1)-k*IR1
           IF (idum.LT.0) idum=idum+IM1
           IF (j.LE.NTAB) iv(j)=idum
        ENDDO
        iy=iv(1)
  ENDIF
    k=idum/IQ1
    idum=IA1*(idum-k*IQ1)-k*IR1
    if (idum.lt.0) idum=idum+IM1
    k=idum2/IQ2
    idum2=IA2*(idum2-k*IQ2)-k*IR2
    if (idum2.lt.0) idum2=idum2+IM2
    j=1+iy/NDIV
    iy=iv(j)-idum2
    iv(j)=idum
    if (iy.lt.1) iy=iy+IMM1
    ran2=min(AM*iy,RNMX)
  RETURN
END FUNCTION

!!
!!-----------------------------------------------------------------------
!!  calculate the local free-energy change
!!-----------------------------------------------------------------------
!!

REAL*8 FUNCTION dfree(mx,my,Tc,r0,g2,g4,g6,i,j,phi,phi1,Teff,Teff1,v0,dx,dy)
  IMPLICIT NONE
  INTEGER mx,my,i,j
  REAL*8 r0,phi(mx,my),v0(mx,my),dgrad2,Tc,g2,g4,g6,dgrad2_cbc
  REAL*8 Teff(mx,my),phi1,df,dx,dy,Teff1
!  df = -0.5*g2*(1.0-v0(i,j))*(phi1**2-phi(i,j)**2) + 0.5*g2*(Teff1/Tc*phi1**2 - Teff(i,j)/Tc*phi(i,j)**2)
  df = -(1.0d0/2.0d0)*g2*(phi1**2 - phi(i,j)**2) + (1.0d0/2.0d0)*g2*(Teff1/Tc*phi1**2 - Teff(i,j)/Tc*phi(i,j)**2)
!  df = df + (1.0d0/2.0d0)*r0*dgrad2(mx,my,dx,dy,i,j,phi,phi1)
  df = df + (1.0d0/2.0d0)*r0*dgrad2_cbc(mx,my,dx,dy,i,j,phi,phi1)
  df = df + (1.0d0/4.0d0)*g4*(phi1**4 - phi(i,j)**4)
  df = df + (1.0d0/6.0d0)*g6*(phi1**6 - phi(i,j)**6)
  dfree = df*dx*dy
  RETURN
END FUNCTION


!!
!!-----------------------------------------------------------------------
!!    gradient of the order parameter
!!-----------------------------------------------------------------------
!!
REAL*8 FUNCTION dgrad2(mx,my,dx,dy,i,j,phi,phi1)
  IMPLICIT NONE
  INTEGER mx,my,i,j,i1,i2
  REAL*8 dx,dy,phi(mx,my),dg2,dg1,phi1

  i1 = mod(i-2+mx,mx)+1
  i2 = mod(i,mx)+1

  dg1 = 0.0
  dg2 = 0.0

  dg1 = dg1 + ((phi(i2,j)-phi(i,j))/dx)**2
  dg1 = dg1 + ((phi(i1,j)-phi(i,j))/dx)**2
  dg2 = dg2 + ((phi(i2,j)-phi1)/dx)**2
  dg2 = dg2 + ((phi(i1,j)-phi1)/dx)**2

  IF (j.GT.1) THEN
    dg1 = dg1 + ((phi(i,j-1)-phi(i,j))/dy)**2
    dg2 = dg2 + ((phi(i,j-1)-phi1)/dy)**2
  END IF
  IF (j.LT.my) THEN
    dg1 = dg1 + ((phi(i,j+1)-phi(i,j))/dy)**2
    dg2 = dg2 + ((phi(i,j+1)-phi1)/dy)**2
  END IF

  dgrad2 = dg2-dg1
  RETURN
END FUNCTION

REAL*8 FUNCTION dgrad2_cbc(mx,my,dx,dy,i,j,phi,phi1)
  IMPLICIT NONE
  INTEGER mx,my,i,j
  REAL*8 dx,dy,phi(mx,my),dg2,dg1,phi1

!  i1 = mod(i-2+mx,mx)+1
!  i2 = mod(i,mx)+1
  dg1 = 0.0
  dg2 = 0.0
!  dg1 = dg1 + ((phi(i2,j)-phi(i,j))/dx)**2
!  dg1 = dg1 + ((phi(i1,j)-phi(i,j))/dx)**2
!  dg2 = dg2 + ((phi(i2,j)-phi1)/dx)**2
!  dg2 = dg2 + ((phi(i1,j)-phi1)/dx)**2
  IF (i.GT.1) THEN
    dg1 = dg1 + ((phi(i-1,j) - phi(i,j))/dx)**2
    dg2 = dg2 + ((phi(i-1,j) - phi1)/dx)**2
  END IF
  IF (i.LT.mx) THEN
    dg1 = dg1 + ((phi(i+1,j) - phi(i,j))/dx)**2
    dg2 = dg2 + ((phi(i+1,j) - phi1)/dx)**2
  END IF
  IF (j.GT.1) THEN
    dg1 = dg1 + ((phi(i,j-1) - phi(i,j))/dy)**2
    dg2 = dg2 + ((phi(i,j-1) - phi1)/dy)**2
  END IF
  IF (j.LT.my) THEN
    dg1 = dg1 + ((phi(i,j+1) - phi(i,j))/dy)**2
    dg2 = dg2 + ((phi(i,j+1) - phi1)/dy)**2
  END IF

  dgrad2_cbc = dg2-dg1
  RETURN
END FUNCTION


!	do n=1,nskip
!c
!c-----------------------------------------------------------------------
!c	  suggest an update and compute the free-energy change
!c-----------------------------------------------------------------------
!c
!	  i = INT(mx*ran2(idummy))+1
!	  j = INT(my*ran2(idummy))+1
!	  phi1 = phi(i,j) + dphi*2.0*(ran2(idummy)-0.5)
!	  if (mfphi.eq.1) then
!	    Teff1 = Teff(i,j)
!	  else
!	    T2 = (gamma**2+phi(i,j)**2)/(gamma**2+phi1**2)
!     .		*(Teff(i,j)**2-Tbath**2)
!	    Teff1 = sqrt(T2+Tbath**2)
!	  endif
!	  df = dfree(mx,my,Tc,r0,i,j,phi,phi1,Teff,Teff1,v0,dx,dy)
!c
!c	  test if we accept the update
!c
!          icount(i,j) = icount(i,j) + 1
!          ncount = ncount + 1
!	  if (ran2(idummy).lt.exp(-df/Tbath)) then	! accept
!	    phi(i,j) = phi1
!	    Teff(i,j) = Teff1
!            f = f + df
!            iflip = iflip + 1
!	  endif
!
!	  ms(i,j) = ms(i,j) + phi(i,j)
!c	  ms = ms + phi
!
!	enddo
