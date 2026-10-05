C=======================================================================
C  tepsim_driver.f : callable form of the main program of temain_mod.f
C
C  Source of the dynamics: teprob.f (Downs and Vogel 1993) and the
C  closed-loop controllers CONTRL1..CONTRL20, INTGTR and CONSHAND of
C  temain_mod.f (Russell, Chiang and Braatz 2000).  Both files are used
C  verbatim; build.py copies them from vendor/ without edits.  The
C  blocks inserted at the four markers below are copied verbatim from
C  temain_mod.f and teprob.f; the modified text of temain_mod.f is
C  Copyright 1998-2002 by The Board of Trustees of the University of
C  Illinois (full notice at the top of temain_mod_subs.f).
C
C  This file is a template.  build.py replaces the four marker lines
C  (MAIN_DECLARATIONS, TEPROC_DECLARATIONS, CONTROLLER_SETUP and
C  CONTROL_CALLS, each prefixed by C and two at signs) by blocks copied
C  verbatim from the original files: the COMMON declarations of the
C  main program, the COMMON/TEPROC/ declarations of TEFUNC, the set
C  points, gains, reset times and initial XMV, and the discrete
C  controller calls inside the simulation loop.
C
C  Differences from the main program of temain_mod.f:
C    1. the random generator state G (COMMON/RANDSD/) is set to GSEED
C       right after TEINIT.  TEINIT draws no random numbers, so this is
C       the same as editing the line G=4651207995.D0 of teprob.f, which
C       is how the original instructions ask users to change the seed;
C    2. the single switch  IF (I.GE.SSPTS) IDV(12)=1  is replaced by a
C       schedule: IDV(J)=1 exactly for the integration steps I with
C       IDVON(J) <= I < IDVOFF(J), and IDV(J)=0 otherwise;
C    3. SUBROUTINE OUTPUT (files written with format E13.5) is replaced
C       by the array XOUT, filled at the same point of the loop
C       (I multiple of 180, before INTGTR) at full double precision;
C    4. the shutdown test of TEFUNC is re-evaluated after each step on
C       the same COMMON/TEPROC/ values; the first step with a shutdown
C       is returned in ISDOUT and, if STOPSD is nonzero, the run stops.
C    5. GWSEED initialises the per-walk generator states GW(1..12) of
C       COMMON/RANDSW/.  They are read only by the "split" build
C       (teprob_split.f); the "original" build ignores them.
C
C  Arguments
C    NSAMP   number of 3-minute samples; NPTS = 180*NSAMP steps of 1 s
C    GSEED   initial value of G
C    GWSEED  initial values of GW(1..12)
C    IDVON   first step at which IDV(J) is on (0 = from the start)
C    IDVOFF  first step at which IDV(J) is off again
C    STOPSD  0 = keep integrating after a shutdown (original behaviour)
C    XOUT    XOUT(1..41,K) = XMEAS(1..41), XOUT(42..52,K) = XMV(1..11)
C            of sample K, written at step I = 180*K
C    NWRIT   number of samples written
C    ISDOUT  first step whose TEFUNC call raised the shutdown flag, or 0
C=======================================================================
      SUBROUTINE TEPSIM(NSAMP, GSEED, GWSEED, IDVON, IDVOFF, STOPSD,
     .                  XOUT, NWRIT, ISDOUT)
      INTEGER NSAMP, IDVON(20), IDVOFF(20), STOPSD, NWRIT, ISDOUT
      DOUBLE PRECISION GSEED, GWSEED(12), XOUT(52,NSAMP)
C
C@@MAIN_DECLARATIONS
C
      DOUBLE PRECISION G
      COMMON/RANDSD/ G
      DOUBLE PRECISION GW
      INTEGER ISTRM
      COMMON/RANDSW/ GW(12), ISTRM
C
C@@TEPROC_DECLARATIONS
C
      INTEGER I, J, K, NN, NPTS, TEST, TEST1, ISD
      DOUBLE PRECISION TIME, YY(50), YP(50), XM7
C
      NN = 50
      NPTS = 180 * NSAMP
C
C  Integrator Step Size:  1 Second Converted to Hours (as in temain_mod.f)
C
      DELTAT = 1. / 3600.
C
C  COMMON/FLAG6/ is zero in a freshly loaded program; make it explicit
C
      FLAG = 0
      ISTRM = 0
C
      CALL TEINIT(NN,TIME,YY,YP)
C
C  Random seed: replaces the value assigned to G inside TEINIT
C
      G = GSEED
      DO 50 J = 1, 12
         GW(J) = GWSEED(J)
 50   CONTINUE
C
C@@CONTROLLER_SETUP
C
      DO 100 I = 1, 20
          IDV(I) = 0
 100  CONTINUE
      NWRIT = 0
      ISDOUT = 0
C
C  Simulation Loop
C
      DO 1000 I = 1, NPTS
C
C  Disturbance schedule (replaces IF (I.GE.SSPTS) IDV(12)=1)
C
        DO 110 J = 1, 20
          IF (I.GE.IDVON(J) .AND. I.LT.IDVOFF(J)) THEN
            IDV(J) = 1
          ELSE
            IDV(J) = 0
          ENDIF
 110    CONTINUE
C
C@@CONTROL_CALLS
C
C  Sampling every 180 s (replaces CALL OUTPUT)
C
        IF (MOD(I,180).EQ.0) THEN
          K = I / 180
          DO 120 J = 1, 41
            XOUT(J,K) = XMEAS(J)
 120      CONTINUE
          DO 130 J = 1, 11
            XOUT(41+J,K) = XMV(J)
 130      CONTINUE
          NWRIT = K
        ENDIF
C
        CALL INTGTR(NN,TIME,DELTAT,YY,YP)
C
C  Shutdown test of TEFUNC, same expressions on the same COMMON values
C
        XM7=(PTR-760.0)/760.0*101.325
        ISD=0
        IF(XM7.GT.3000.0)ISD=1
        IF(VLR/35.3145.GT.24.0)ISD=1
        IF(VLR/35.3145.LT.2.0)ISD=1
        IF(TCR.GT.175.0)ISD=1
        IF(VLS/35.3145.GT.12.0)ISD=1
        IF(VLS/35.3145.LT.1.0)ISD=1
        IF(VLC/35.3145.GT.8.0)ISD=1
        IF(VLC/35.3145.LT.1.0)ISD=1
        IF (ISD.NE.0 .AND. ISDOUT.EQ.0) ISDOUT = I
        IF (ISD.NE.0 .AND. STOPSD.NE.0) GOTO 1001
C
        CALL CONSHAND
C
 1000 CONTINUE
 1001 CONTINUE
      RETURN
      END
