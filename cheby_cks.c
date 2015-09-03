
#include <petsc/private/kspimpl.h>                    /*I "petscksp.h" I*/
#include <../src/ksp/ksp/impls/cheby/chebyshevimpl.h>

#undef __FUNCT__
#define __FUNCT__ "KSPReset_Chebyshev"
static PetscErrorCode KSPReset_Chebyshev(KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset(cheb->kspest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_Chebyshev"
static PetscErrorCode KSPSetUp_Chebyshev(KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInt		nwork = 11;		/* Add predefined vectors C1,C2,C3, checksum1(A) CKSAmat1, checksum2(A) CKSAmat2, checksum3(A) CKSAmat3 and checkpoint vectors CKPpk,CKPpkm1; */
  //PetscInt	nwork = 3;
  ierr = KSPSetWorkVecs(ksp,nwork);CHKERRQ(ierr);
  if ((cheb->emin == 0. || cheb->emax == 0.) && !cheb->kspest) { /* We need to estimate eigenvalues */
    ierr = KSPChebyshevEstEigSet(ksp,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevSetEigenvalues_Chebyshev"
static PetscErrorCode KSPChebyshevSetEigenvalues_Chebyshev(KSP ksp,PetscReal emax,PetscReal emin)
{
  KSP_Chebyshev  *chebyshevP = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (emax <= emin) SETERRQ2(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"Maximum eigenvalue must be larger than minimum: max %g min %g",(double)emax,(double)emin);
  if (emax*emin <= 0.0) SETERRQ2(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"Both eigenvalues must be of the same sign: max %g min %g",(double)emax,(double)emin);
  chebyshevP->emax = emax;
  chebyshevP->emin = emin;

  ierr = KSPChebyshevEstEigSet(ksp,0.,0.,0.,0.);CHKERRQ(ierr); /* Destroy any estimation setup */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevEstEigSet_Chebyshev"
static PetscErrorCode KSPChebyshevEstEigSet_Chebyshev(KSP ksp,PetscReal a,PetscReal b,PetscReal c,PetscReal d)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a != 0.0 || b != 0.0 || c != 0.0 || d != 0.0) {
    if (!cheb->kspest) { /* should this block of code be moved to KSPSetUp_Chebyshev()? */
      PetscBool nonzero;

      ierr = KSPCreate(PetscObjectComm((PetscObject)ksp),&cheb->kspest);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)cheb->kspest,(PetscObject)ksp,1);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(cheb->kspest,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(cheb->kspest,"esteig_");CHKERRQ(ierr);
      ierr = KSPSetSkipPCSetFromOptions(cheb->kspest,PETSC_TRUE);CHKERRQ(ierr);

      ierr = KSPSetPC(cheb->kspest,ksp->pc);CHKERRQ(ierr);

      ierr = KSPGetInitialGuessNonzero(ksp,&nonzero);CHKERRQ(ierr);
      ierr = KSPSetInitialGuessNonzero(cheb->kspest,nonzero);CHKERRQ(ierr);
      ierr = KSPSetComputeEigenvalues(cheb->kspest,PETSC_TRUE);CHKERRQ(ierr);

      /* Estimate with a fixed number of iterations */
      ierr = KSPSetConvergenceTest(cheb->kspest,KSPConvergedSkip,0,0);CHKERRQ(ierr);
      ierr = KSPSetNormType(cheb->kspest,KSP_NORM_NONE);CHKERRQ(ierr);
      ierr = KSPSetTolerances(cheb->kspest,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,cheb->eststeps);CHKERRQ(ierr);
    }
    if (a >= 0) cheb->tform[0] = a;
    if (b >= 0) cheb->tform[1] = b;
    if (c >= 0) cheb->tform[2] = c;
    if (d >= 0) cheb->tform[3] = d;
    cheb->amatid    = 0;
    cheb->pmatid    = 0;
    cheb->amatstate = -1;
    cheb->pmatstate = -1;
  } else {
    ierr = KSPDestroy(&cheb->kspest);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevEstEigSetRandom_Chebyshev"
static PetscErrorCode KSPChebyshevEstEigSetRandom_Chebyshev(KSP ksp,PetscRandom random)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (random) {ierr = PetscObjectReference((PetscObject)random);CHKERRQ(ierr);}
  ierr = PetscRandomDestroy(&cheb->random);CHKERRQ(ierr);

  cheb->random = random;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevSetEigenvalues"
/*@
   KSPChebyshevSetEigenvalues - Sets estimates for the extreme eigenvalues
   of the preconditioned problem.

   Logically Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
-  emax, emin - the eigenvalue estimates

  Options Database:
.  -ksp_chebyshev_eigenvalues emin,emax

   Note: If you run with the Krylov method of KSPCG with the option -ksp_monitor_singular_value it will
    for that given matrix and preconditioner an estimate of the extreme eigenvalues.

   Level: intermediate

.keywords: KSP, Chebyshev, set, eigenvalues
@*/
PetscErrorCode  KSPChebyshevSetEigenvalues(KSP ksp,PetscReal emax,PetscReal emin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,emax,2);
  PetscValidLogicalCollectiveReal(ksp,emin,3);
  ierr = PetscTryMethod(ksp,"KSPChebyshevSetEigenvalues_C",(KSP,PetscReal,PetscReal),(ksp,emax,emin));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevEstEigSet"
/*@
   KSPChebyshevEstEigSet - Automatically estimate the eigenvalues to use for Chebyshev

   Logically Collective on KSP

   Input Parameters:
+  ksp - the Krylov space context
.  a - multiple of min eigenvalue estimate to use for min Chebyshev bound (or PETSC_DECIDE)
.  b - multiple of max eigenvalue estimate to use for min Chebyshev bound (or PETSC_DECIDE)
.  c - multiple of min eigenvalue estimate to use for max Chebyshev bound (or PETSC_DECIDE)
-  d - multiple of max eigenvalue estimate to use for max Chebyshev bound (or PETSC_DECIDE)

  Options Database:
.  -ksp_chebyshev_esteig a,b,c,d

   Notes:
   The Chebyshev bounds are estimated using
.vb
   minbound = a*minest + b*maxest
   maxbound = c*minest + d*maxest
.ve
   The default configuration targets the upper part of the spectrum for use as a multigrid smoother, so only the maximum eigenvalue estimate is used.
   The minimum eigenvalue estimate obtained by Krylov iteration is typically not accurate until the method has converged.

   If 0.0 is passed for all transform arguments (a,b,c,d), eigenvalue estimation is disabled.

   The default transform is (0,0.1; 0,1.1) which targets the "upper" part of the spectrum, as desirable for use with multigrid.

   Level: intermediate

.keywords: KSP, Chebyshev, set, eigenvalues, PCMG
@*/
PetscErrorCode KSPChebyshevEstEigSet(KSP ksp,PetscReal a,PetscReal b,PetscReal c,PetscReal d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveReal(ksp,a,2);
  PetscValidLogicalCollectiveReal(ksp,b,3);
  PetscValidLogicalCollectiveReal(ksp,c,4);
  PetscValidLogicalCollectiveReal(ksp,d,5);
  ierr = PetscTryMethod(ksp,"KSPChebyshevEstEigSet_C",(KSP,PetscReal,PetscReal,PetscReal,PetscReal),(ksp,a,b,c,d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevEstEigSetRandom"
/*@
   KSPChebyshevEstEigSetRandom - set random context for estimating eigenvalues

   Logically Collective

   Input Arguments:
+  ksp - linear solver context
-  random - random number context or NULL to disable randomized RHS

   Options Database:
.  -ksp_chebyshev_esteig_random

  Level: intermediate

.seealso: KSPChebyshevEstEigSet(), PetscRandomCreate()
@*/
PetscErrorCode KSPChebyshevEstEigSetRandom(KSP ksp,PetscRandom random)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (random) PetscValidHeaderSpecific(random,PETSC_RANDOM_CLASSID,2);
  ierr = PetscTryMethod(ksp,"KSPChebyshevEstEigSetRandom_C",(KSP,PetscRandom),(ksp,random));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevEstEigGetKSP"
/*@
  KSPChebyshevEstEigGetKSP - Get the Krylov method context used to estimate eigenvalues for the Chebyshev method.  If
  a Krylov method is not being used for this purpose, NULL is returned.  The reference count of the returned KSP is
  not incremented: it should not be destroyed by the user.

  Input Parameters:
. ksp - the Krylov space context

  Output Parameters:
. kspest the eigenvalue estimation Krylov space context

  Level: intermediate

.seealso: KSPChebyshevEstEigSet()
@*/
PetscErrorCode KSPChebyshevEstEigGetKSP(KSP ksp, KSP *kspest)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *kspest = NULL;
  ierr = PetscTryMethod(ksp,"KSPChebyshevEstEigGetKSP_C",(KSP,KSP*),(ksp,kspest));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevEstEigGetKSP_Chebyshev"
static PetscErrorCode KSPChebyshevEstEigGetKSP_Chebyshev(KSP ksp, KSP *kspest)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev*)ksp->data;

  PetscFunctionBegin;
  *kspest = cheb->kspest;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_Chebyshev"
static PetscErrorCode KSPSetFromOptions_Chebyshev(PetscOptions *PetscOptionsObject,KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       neigarg = 2, nestarg = 4;
  PetscReal      eminmax[2] = {0., 0.};
  PetscReal      tform[4] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
  PetscBool      flgeig, flgest;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP Chebyshev Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_chebyshev_esteig_steps","Number of est steps in Chebyshev","",cheb->eststeps,&cheb->eststeps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-ksp_chebyshev_eigenvalues","extreme eigenvalues","KSPChebyshevSetEigenvalues",eminmax,&neigarg,&flgeig);CHKERRQ(ierr);
  if (flgeig) {
    if (neigarg != 2) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"-ksp_chebyshev_eigenvalues: must specify 2 parameters, min and max eigenvalues");
    ierr = KSPChebyshevSetEigenvalues(ksp, eminmax[1], eminmax[0]);CHKERRQ(ierr);
  }
  ierr = PetscOptionsRealArray("-ksp_chebyshev_esteig","estimate eigenvalues using a Krylov method, then use this transform for Chebyshev eigenvalue bounds","KSPChebyshevEstEigSet",tform,&nestarg,&flgest);CHKERRQ(ierr);
  if (flgest) {
    switch (nestarg) {
    case 0:
      ierr = KSPChebyshevEstEigSet(ksp,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
      break;
    case 2:                     /* Base everything on the max eigenvalues */
      ierr = KSPChebyshevEstEigSet(ksp,PETSC_DECIDE,tform[0],PETSC_DECIDE,tform[1]);CHKERRQ(ierr);
      break;
    case 4:                     /* Use the full 2x2 linear transformation */
      ierr = KSPChebyshevEstEigSet(ksp,tform[0],tform[1],tform[2],tform[3]);CHKERRQ(ierr);
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"Must specify either 0, 2, or 4 parameters for eigenvalue estimation");
    }
  }

  /* We need to estimate eigenvalues; need to set this here so that KSPSetFromOptions() is called on the estimator */
  if ((cheb->emin == 0. || cheb->emax == 0.) && !cheb->kspest) {
   ierr = KSPChebyshevEstEigSet(ksp,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  }

  if (cheb->kspest) {
    PetscBool estrand = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_chebyshev_esteig_random","Use Random right hand side for eigenvalue estimation","KSPChebyshevEstEigSetRandom",estrand,&estrand,NULL);CHKERRQ(ierr);
    if (estrand) {
      PetscRandom random;
      ierr = PetscRandomCreate(PetscObjectComm((PetscObject)ksp),&random);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)random,((PetscObject)ksp)->prefix);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)random,"ksp_chebyshev_esteig_");CHKERRQ(ierr);
      ierr = PetscRandomSetFromOptions(random);CHKERRQ(ierr);
      ierr = KSPChebyshevEstEigSetRandom(ksp,random);CHKERRQ(ierr);
      ierr = PetscRandomDestroy(&random);CHKERRQ(ierr);
    }
  }

  if (cheb->kspest) {
    ierr = KSPSetFromOptions(cheb->kspest);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPChebyshevComputeExtremeEigenvalues_Private"
/*
 * Must be passed a KSP solver that has "converged", with KSPSetComputeEigenvalues() called before the solve
 */
static PetscErrorCode KSPChebyshevComputeExtremeEigenvalues_Private(KSP kspest,PetscReal *emin,PetscReal *emax)
{
  PetscErrorCode ierr;
  PetscInt       n,neig;
  PetscReal      *re,*im,min,max;

  PetscFunctionBegin;
  ierr = KSPGetIterationNumber(kspest,&n);CHKERRQ(ierr);
  ierr = PetscMalloc2(n,&re,n,&im);CHKERRQ(ierr);
  ierr = KSPComputeEigenvalues(kspest,n,re,im,&neig);CHKERRQ(ierr);
  min  = PETSC_MAX_REAL;
  max  = PETSC_MIN_REAL;
  for (n=0; n<neig; n++) {
    min = PetscMin(min,re[n]);
    max = PetscMax(max,re[n]);
  }
  ierr  = PetscFree2(re,im);CHKERRQ(ierr);
  *emax = max;
  *emin = min;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_Chebyshev"
static PetscErrorCode KSPSolve_Chebyshev(KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       k,kp1,km1,maxit,ktmp,i;
  PetscScalar    alpha,omegaprod,mu,omega,Gamma,c[3],scale;
  PetscReal      rnorm = 0.0;
  Vec            sol_orig,b,p[3],r;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;
  
  /* Dingwen */
  PetscInt		itv_d, itv_c;
  PetscScalar	CKSr1,CKSr2,CKSr3;
  PetscScalar	CKSp1[3],CKSp2[3],CKSp3[3];
  PetscScalar	CKSb1,CKSb2,CKSb3;
  Vec			CKSAmat1,CKSAmat2, CKSAmat3;
  Vec			C1,C2,C3;
  PetscScalar	d1,d2,d3;
  PetscScalar	sump1[3],sump2[3],sump3[3];
  PetscScalar	sumr1,sumr2,sumr3;
  Vec			CKPpk, CKPpkm1;
  PetscInt		CKPi,CKPk,CKPkp1,CKPkm1;
  PetscScalar	CKPck,CKPckm1;
  PetscBool		flag1 = PETSC_TRUE,flag2 = PETSC_TRUE,flag3 = PETSC_TRUE,flag4 = PETSC_TRUE;
  PetscInt		pos;
  PetscScalar	v;
  PetscScalar	theta1 = 1.0e-6, theta2 = 1.0e-10;
  /* Dingwen */

  PetscFunctionBegin;
  
  /* Dingwen */
  int rank;									/* Get MPI variables */
  MPI_Comm_rank	(MPI_COMM_WORLD,&rank);
  /* Dingwen */
 
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  if (cheb->kspest) {
    PetscObjectId    amatid,    pmatid;
    PetscObjectState amatstate, pmatstate;

    ierr = PetscObjectGetId((PetscObject)Amat,&amatid);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)Pmat,&pmatid);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)Amat,&amatstate);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)Pmat,&pmatstate);CHKERRQ(ierr);
    if (amatid != cheb->amatid || pmatid != cheb->pmatid || amatstate != cheb->amatstate || pmatstate != cheb->pmatstate) {
      PetscReal max=0.0,min=0.0;
      Vec       X,B;
      X = ksp->work[0];
      if (cheb->random) {
        B    = ksp->work[1];
        ierr = VecSetRandom(B,cheb->random);CHKERRQ(ierr);
      } else {
        B = ksp->vec_rhs;
      }
      ierr = KSPSolve(cheb->kspest,B,X);CHKERRQ(ierr);

      if (ksp->guess_zero) {
        ierr = VecZeroEntries(X);CHKERRQ(ierr);
      }
      ierr = KSPChebyshevComputeExtremeEigenvalues_Private(cheb->kspest,&min,&max);CHKERRQ(ierr);

      cheb->emin = cheb->tform[0]*min + cheb->tform[1]*max;
      cheb->emax = cheb->tform[2]*min + cheb->tform[3]*max;

      cheb->amatid    = amatid;
      cheb->pmatid    = pmatid;
      cheb->amatstate = amatstate;
      cheb->pmatstate = pmatstate;
    }
  }

  ksp->its = 0;
  maxit    = ksp->max_it;

  /* These three point to the three active solutions, we
     rotate these three at each solution update */
  km1      = 0; k = 1; kp1 = 2;
  sol_orig = ksp->vec_sol; /* ksp->vec_sol will be asigned to rotating vector p[k], thus save its address */
  b        = ksp->vec_rhs;
  p[km1]   = sol_orig;
  p[k]     = ksp->work[0];
  p[kp1]   = ksp->work[1];
  r        = ksp->work[2];
  /* Dingwen */
  CKSAmat1		= ksp->work[3];
  CKSAmat2		= ksp->work[4];
  CKSAmat3		= ksp->work[5];
  C1			= ksp->work[6];
  C2			= ksp->work[7];
  C3			= ksp->work[8];
  CKPpk			= ksp->work[9];
  CKPpkm1		= ksp->work[10];
  /* Dingwen */


  /* use scale*B as our preconditioner */
  scale = 2.0/(cheb->emax + cheb->emin);

  /*   -alpha <=  scale*lambda(B^{-1}A) <= alpha   */
  alpha     = 1.0 - scale*(cheb->emin);
  Gamma     = 1.0;
  mu        = 1.0/alpha;
  omegaprod = 2.0/alpha;

  c[km1] = 1.0;
  c[k]   = mu;

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,p[km1],r);CHKERRQ(ierr);     /*  r = b - A*p[km1] */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(b,r);CHKERRQ(ierr);
  }

  ierr = KSP_PCApply(ksp,r,p[k]);CHKERRQ(ierr);  /* p[k] = scale B^{-1}r + p[km1] */
  ierr = VecAYPX(p[k],scale,p[km1]);CHKERRQ(ierr);
  
  /* Dingwen */	
  /* checksum coefficients initialization */
  PetscInt size;
  PetscInt *index;
  PetscScalar *v1,*v2,*v3;
  ierr = VecGetSize(b,&size);
  v1 	= (PetscScalar *)malloc(size*sizeof(PetscScalar));
  v2 	= (PetscScalar *)malloc(size*sizeof(PetscScalar));
  v3 	= (PetscScalar *)malloc(size*sizeof(PetscScalar));
  index	= (PetscInt *)malloc(size*sizeof(PetscInt));
  for (i=0; i<size; i++)
  {
	  index[i] = i;
	  v1[i] = 1.0;
	  v2[i] = i+1.0;
	  v3[i] = 1/(i+1.0);
  }
  ierr	= VecSetValues(C1,size,index,v1,INSERT_VALUES);CHKERRQ(ierr);	
  ierr 	= VecSetValues(C2,size,index,v2,INSERT_VALUES);CHKERRQ(ierr);
  ierr	= VecSetValues(C3,size,index,v3,INSERT_VALUES);CHKERRQ(ierr);	

  d1 = 1.0;
  d2 = 2.0;
  d3 = 3.0;
  ierr = KSP_MatMultTranspose(ksp,Amat,C1,CKSAmat1);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat1,-d1,C1);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat1,-d2,C2);CHKERRQ(ierr); 
  ierr = VecAXPY(CKSAmat1,-d3,C3);CHKERRQ(ierr);					/* Compute the initial checksum1(A) */ 
  ierr = KSP_MatMultTranspose(ksp,Amat,C2,CKSAmat2);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat2,-d2,C1);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat2,-d3,C2);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat2,-d1,C3);CHKERRQ(ierr);					/* Compute the initial checksum2(A) */ 
  ierr = KSP_MatMultTranspose(ksp,Amat,C3,CKSAmat3);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat3,-d3,C1);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat3,-d1,C2);CHKERRQ(ierr);
  ierr = VecAXPY(CKSAmat3,-d2,C3);CHKERRQ(ierr);					/* Compute the initial checksum3(A) */  
  /* Checksum Initialization */
  ierr = VecDot(C1,b,&CKSb1);CHKERRQ(ierr);						/* Compute the initial checksum1(b) */
  ierr = VecDot(C2,b,&CKSb2);CHKERRQ(ierr);						/* Compute the initial checksum2(b) */
  ierr = VecDot(C3,b,&CKSb3);CHKERRQ(ierr);						/* Compute the initial checksum3(b) */
  ierr = VecDot(C1,r,&CKSr1);CHKERRQ(ierr);						/* Compute the initial checksum1(r) */
  ierr = VecDot(C2,r,&CKSr2);CHKERRQ(ierr);						/* Compute the initial checksum2(r) */
  ierr = VecDot(C3,r,&CKSr3);CHKERRQ(ierr);						/* Compute the initial checksum3(r) */
  for (i=0;i<3;i++)
  {
	  ierr = VecDot(C1,p[i],&CKSp1[i]);CHKERRQ(ierr);				/* Compute the initial checksum1(p[i]) */
	  ierr = VecDot(C2,p[i],&CKSp2[i]);CHKERRQ(ierr);				/* Compute the initial checksum2(p[i]) */
	  ierr = VecDot(C3,p[i],&CKSp3[i]);CHKERRQ(ierr);				/* Compute the initial checksum3(p[i]) */
  }
  itv_c = 2;
  itv_d = 10;
  /* Dingwen */

  for (i=0; i<maxit; i++) {
    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
	
	/* Dingwen */
	/* Checkpoint and Rollback */
	  if ((i>0) && (i%itv_d == 0))
	  {
		  ierr = VecDot(C1,p[km1],&sump1[km1]);CHKERRQ(ierr);
		  if (PetscAbsScalar(sump1[km1]-CKSp1[km1]) > theta1)
		  {
			  /* Rollback to recovery */
			  if (rank==0) printf ("Recovery start...\n");
			  if (rank==0) printf ("Rollback from iteration-%d to iteration-%d\n",i,CKPi);
			  i 	= CKPi;
			  k 	= CKPk;
			  kp1	= CKPkp1;
			  km1	= CKPkm1;
			  c[k]	= CKPck;
			  c[km1]= CKPckm1;
			  ierr	= VecCopy(CKPpk,p[k]);CHKERRQ(ierr);
			  ierr	= VecCopy(CKPpkm1,p[km1]);CHKERRQ(ierr);
			  ierr = VecDot(C1,p[k],&CKSp1[k]);CHKERRQ(ierr);
			  ierr = VecDot(C2,p[k],&CKSp2[k]);CHKERRQ(ierr);
			  ierr = VecDot(C3,p[k],&CKSp3[k]);CHKERRQ(ierr);
			  ierr = VecDot(C1,p[km1],&CKSp1[km1]);CHKERRQ(ierr);
			  ierr = VecDot(C2,p[km1],&CKSp2[km1]);CHKERRQ(ierr);
			  ierr = VecDot(C3,p[km1],&CKSp3[km1]);CHKERRQ(ierr);
			  if (rank==0) printf ("Recovery end.\n");
		}
		else if (i%(itv_c*itv_d) == 0)
		{
			if (rank==0) printf ("Checkpoint iteration-%d\n",i);
			CKPi	= i;
			CKPk	= k;
			CKPkp1	= kp1;
			CKPkm1	= km1;
			CKPck	= c[k];
			CKPckm1	= c[km1];  
			ierr	= VecCopy(p[k],CKPpk);CHKERRQ(ierr);
			ierr	= VecCopy(p[km1],CKPpkm1);CHKERRQ(ierr);
		}
	}
	/* Dingwen */
	
    ksp->its++;
    ierr   = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    c[kp1] = 2.0*mu*c[k] - c[km1];
    omega  = omegaprod*c[k]/c[kp1];
	
	ierr = KSP_MatMult(ksp,Amat,p[k],r);CHKERRQ(ierr);          /*  r = b - Ap[k]    */
    ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
	
	/* Dingwen */
	/* Inject an error to simulate cache errors */
	if((i==90)&&(flag4))
	  {
		  pos 	= 200;
		  v		= 1;
		  ierr	= VecSetValue(r,pos,v,INSERT_VALUES);CHKERRQ(ierr);
		  VecAssemblyBegin(r);
		  VecAssemblyEnd(r);
		  if (rank==0) printf ("Inject an error between MVM and Checksum update to simulate cache error at iteration-%d\n", i);
		  flag4	= PETSC_FALSE;
	  }

	/* Checksum update of MVM + VLO */
	ierr = VecDot(CKSAmat1, p[k], &CKSr1);CHKERRQ(ierr);
	CKSr1 = CKSb1 - (CKSr1 + d1*CKSp1[k] + d2*CKSp2[k] + d3*CKSp3[k]);				/* Update checksum1(r) = checksum1(b) - (checksum1(A)p[k] + d1*checksum1(p[k]) + d2*checksum2(p[k]) + d3*checksum3(p[k]); */ 
	ierr = VecDot(CKSAmat2, p[k], &CKSr2);CHKERRQ(ierr);
	CKSr2 = CKSb2 - (CKSr2 + d2*CKSp1[k] + d3*CKSp2[k] + d1*CKSp3[k]);				/* Update checksum2(r) = checksum2(b) - (checksum2(A)p[k] + d2*checksum1(p[k]) + d3*checksum2(p[k]) + d1*checksum3(p[k]); */
	ierr = VecDot(CKSAmat3, p[k], &CKSr3);CHKERRQ(ierr);
	CKSr3 = CKSb3 - (CKSr3 + d3*CKSp1[k] + d1*CKSp2[k] + d2*CKSp3[k]);				/* Update checksum3(r) = checksum3(b) - (checksum3(A)p[k] + d3*checksum1(p[k]) + d1*checksum2(p[k]) + d2*checksum3(p[k]); */

	/* Inject an error */
	if((i==30)&&(flag2))
	{
		pos 	= 100;
		v		= 1000;
		ierr	= VecSetValue(r,pos,v,INSERT_VALUES);CHKERRQ(ierr);
		VecAssemblyBegin(r);
		VecAssemblyEnd(r);
		if (rank==0) printf ("Inject an error after MVM at iteration-%d\n", i);
		flag2	= PETSC_FALSE;
	}
	
	/* Inject errors */
	if((i==70)&&(flag3))
	{
		pos 	= 100;
		v		= 10000;
		ierr	= VecSetValue(r,pos,v,INSERT_VALUES);CHKERRQ(ierr);
		pos 	= 150;
		v		= 200;
		ierr	= VecSetValue(r,pos,v,INSERT_VALUES);CHKERRQ(ierr);
		VecAssemblyBegin(r);
		VecAssemblyEnd(r);
		if (rank==0) printf ("Inject errors after MVM at iteration-%d\n", i);
		flag3	= PETSC_FALSE;
	}

	/* Inner Protection */
	PetscScalar delta1,delta2,delta3;			  
	ierr = VecDot(C1,r,&sumr1);CHKERRQ(ierr);
	ierr = VecDot(C2,r,&sumr2);CHKERRQ(ierr);
	ierr = VecDot(C3,r,&sumr3);CHKERRQ(ierr);
	delta1 = sumr1 - CKSr1;
	delta2 = sumr2 - CKSr2;
	delta3 = sumr3 - CKSr3;
	if (PetscAbsScalar(delta1) > theta1)
	{
		ierr = VecDot(C1,p[k],&sump1[k]);CHKERRQ(ierr);
		if (PetscAbsScalar(CKSp1[k]-sump1[k]) > theta1)
		{
			if (rank==0) printf ("Errors occur before MVM\n");
			if (rank==0) printf ("Recovery start...\n");
			if (rank==0) printf ("Rollback from iteration-%d to iteration-%d\n",i,CKPi);
			i 	= CKPi;
			k 	= CKPk;
			kp1	= CKPkp1;
			km1	= CKPkm1;
			c[k]	= CKPck;
			c[km1]= CKPckm1;
			ierr	= VecCopy(CKPpk,p[k]);CHKERRQ(ierr);
			ierr	= VecCopy(CKPpkm1,p[km1]);CHKERRQ(ierr);
			ierr = VecDot(C1,p[k],&CKSp1[k]);CHKERRQ(ierr);
			ierr = VecDot(C2,p[k],&CKSp2[k]);CHKERRQ(ierr);
			ierr = VecDot(C3,p[k],&CKSp3[k]);CHKERRQ(ierr);
			ierr = VecDot(C1,p[km1],&CKSp1[km1]);CHKERRQ(ierr);
			ierr = VecDot(C2,p[km1],&CKSp2[km1]);CHKERRQ(ierr);
			ierr = VecDot(C3,p[km1],&CKSp3[km1]);CHKERRQ(ierr);
			if (rank==0) printf ("Recovery end.\n");
			
			c[kp1] = 2.0*mu*c[k] - c[km1];
			omega  = omegaprod*c[k]/c[kp1];
			ierr = KSP_MatMult(ksp,Amat,p[k],r);CHKERRQ(ierr);          /*  r = b - Ap[k]    */
			ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
			/* Dingwen */	/* MVM + VLO */
			ierr = VecDot(CKSAmat1, p[k], &CKSr1);CHKERRQ(ierr);
			CKSr1 = CKSb1 - (CKSr1 + d1*CKSp1[k] + d2*CKSp2[k] + d3*CKSp3[k]);				/* Update checksum1(r) = checksum1(b) - (checksum1(A)p[k] + d1*checksum1(p[k]) + d2*checksum2(p[k]) + d3*checksum3(p[k]); */ 
			ierr = VecDot(CKSAmat2, p[k], &CKSr2);CHKERRQ(ierr);
			CKSr2 = CKSb2 - (CKSr2 + d2*CKSp1[k] + d3*CKSp2[k] + d1*CKSp3[k]);				/* Update checksum2(r) = checksum2(b) - (checksum2(A)p[k] + d2*checksum1(p[k]) + d3*checksum2(p[k]) + d1*checksum3(p[k]); */
			ierr = VecDot(CKSAmat3, p[k], &CKSr3);CHKERRQ(ierr);
			CKSr3 = CKSb3 - (CKSr3 + d3*CKSp1[k] + d1*CKSp2[k] + d2*CKSp3[k]);				/* Update checksum3(r) = checksum3(b) - (checksum3(A)p[k] + d3*checksum1(p[k]) + d1*checksum2(p[k]) + d2*checksum3(p[k]); */
			}
			else{
			  if (PetscAbsScalar(1.0-(delta2*delta3)/(delta1*delta1)) > theta2)
			  {
			  /* Rollback and Recovery */
			  if (rank==0) printf ("Multiple errors in output vector of MVM\n");
			  if (rank==0) printf ("Recovery start...\n");
			  if (rank==0) printf ("Rollback from iteration-%d to iteration-%d\n",i,CKPi);
			  i 	= CKPi;
			  k 	= CKPk;
			  kp1	= CKPkp1;
			  km1	= CKPkm1;
			  c[k]	= CKPck;
			  c[km1]= CKPckm1;
			  ierr	= VecCopy(CKPpk,p[k]);CHKERRQ(ierr);
			  ierr	= VecCopy(CKPpkm1,p[km1]);CHKERRQ(ierr);
			  ierr = VecDot(C1,p[k],&CKSp1[k]);CHKERRQ(ierr);
			  ierr = VecDot(C2,p[k],&CKSp2[k]);CHKERRQ(ierr);
			  ierr = VecDot(C3,p[k],&CKSp3[k]);CHKERRQ(ierr);
			  ierr = VecDot(C1,p[km1],&CKSp1[km1]);CHKERRQ(ierr);
			  ierr = VecDot(C2,p[km1],&CKSp2[km1]);CHKERRQ(ierr);
			  ierr = VecDot(C3,p[km1],&CKSp3[km1]);CHKERRQ(ierr);
			  if (rank==0) printf ("Recovery end.\n");
			  
			  /* Rollback to beginning of iteration */
			  c[kp1] = 2.0*mu*c[k] - c[km1];
			  omega  = omegaprod*c[k]/c[kp1];

			  ierr = KSP_MatMult(ksp,Amat,p[k],r);CHKERRQ(ierr);          /*  r = b - Ap[k]    */
			  ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
			  	/* Dingwen */	/* MVM + VLO */
				ierr = VecDot(CKSAmat1, p[k], &CKSr1);CHKERRQ(ierr);
				CKSr1 = CKSb1 - (CKSr1 + d1*CKSp1[k] + d2*CKSp2[k] + d3*CKSp3[k]);				/* Update checksum1(r) = checksum1(b) - (checksum1(A)p[k] + d1*checksum1(p[k]) + d2*checksum2(p[k]) + d3*checksum3(p[k]); */ 
				ierr = VecDot(CKSAmat2, p[k], &CKSr2);CHKERRQ(ierr);
				CKSr2 = CKSb2 - (CKSr2 + d2*CKSp1[k] + d3*CKSp2[k] + d1*CKSp3[k]);				/* Update checksum2(r) = checksum2(b) - (checksum2(A)p[k] + d2*checksum1(p[k]) + d3*checksum2(p[k]) + d1*checksum3(p[k]); */
				ierr = VecDot(CKSAmat3, p[k], &CKSr3);CHKERRQ(ierr);
				CKSr3 = CKSb3 - (CKSr3 + d3*CKSp1[k] + d1*CKSp2[k] + d2*CKSp3[k]);				/* Update checksum3(r) = checksum3(b) - (checksum3(A)p[k] + d3*checksum1(p[k]) + d1*checksum2(p[k]) + d2*checksum3(p[k]); */
			  }
			  else{
				  if (rank == 0) printf ("Locate and correct right away\n");
				  VecScatter	ctx;
				  Vec			r_SEQ;
				  PetscScalar	*r_ARR;
				  VecScatterCreateToAll(r,&ctx,&r_SEQ);
				  VecScatterBegin(ctx,r,r_SEQ,INSERT_VALUES,SCATTER_FORWARD);
				  VecScatterEnd(ctx,r,r_SEQ,INSERT_VALUES,SCATTER_FORWARD);
				  VecGetArray(r_SEQ,&r_ARR);
				  pos	= rint(delta2/delta1) - 1;
				  v		= r_ARR[pos];
				  v		= v - delta1;
				  ierr	= VecSetValues(r,1,&pos,&v,INSERT_VALUES);CHKERRQ(ierr);
				  if (rank==0) printf ("Correct an error in output vector of MVM at iteration-%d\n", i);
				  VecDestroy(&r_SEQ);
				  VecScatterDestroy(&ctx);
			  }
		  }
		}
		/* Dingwen */

	ierr = KSP_PCApply(ksp,r,p[kp1]);CHKERRQ(ierr);             /*  p[kp1] = B^{-1}r  */
    ksp->vec_sol = p[k];	
	/* Dingwen */
	ierr = VecDot(C1,p[kp1],&CKSp1[kp1]);CHKERRQ(ierr);				/* Update checksum1(p[kp1]) */
	ierr = VecDot(C2,p[kp1],&CKSp2[kp1]);CHKERRQ(ierr);				/* Update checksum2(p[kp1]) */
	ierr = VecDot(C3,p[kp1],&CKSp3[kp1]);CHKERRQ(ierr);				/* Update checksum3(p[kp1]) */
	/* Dingwen */


    /* calculate residual norm if requested */
    if (ksp->normtype != KSP_NORM_NONE || ksp->numbermonitors) {
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);
      } else {
        ierr = VecNorm(p[kp1],NORM_2,&rnorm);CHKERRQ(ierr);
      }
      ierr         = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->rnorm   = rnorm;
      ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
    }

    /* y^{k+1} = omega(y^{k} - y^{k-1} + Gamma*r^{k}) + y^{k-1} */
    ierr = VecAXPBYPCZ(p[kp1],1.0-omega,omega,omega*Gamma*scale,p[km1],p[k]);CHKERRQ(ierr);
	
	/* Dingwen */
	CKSp1[kp1] = (1.0-omega)*CKSp1[km1] + omega*CKSp1[k] + omega*Gamma*scale*CKSp1[kp1];
	CKSp2[kp1] = (1.0-omega)*CKSp2[km1] + omega*CKSp2[k] + omega*Gamma*scale*CKSp2[kp1];
	CKSp3[kp1] = (1.0-omega)*CKSp3[km1] + omega*CKSp3[k] + omega*Gamma*scale*CKSp3[kp1];
	/* Dingwen */

	/* Dingwen */
	/* Inject error */
	if ((i==50)&&(flag1))
	{
		pos		= 100;
		v	 	= -1;
		ierr	= VecSetValues(p[kp1],1,&pos,&v,INSERT_VALUES);CHKERRQ(ierr);
		ierr	= VecAssemblyBegin(p[kp1]);CHKERRQ(ierr);
		ierr	= VecAssemblyEnd(p[kp1]);CHKERRQ(ierr);  
		flag1	= PETSC_FALSE;
		if (rank==0)printf ("Inject an error at the end of iteration-%d\n", i);
	}
	/* Dingwen */

    ktmp = km1;
    km1  = k;
    k    = kp1;
    kp1  = ktmp;
  }
  if (rank==0)
	  printf ("Number of iterations without rollback = %d\n", i+1);

  if (!ksp->reason) {
    if (ksp->normtype != KSP_NORM_NONE) {
      ierr = KSP_MatMult(ksp,Amat,p[k],r);CHKERRQ(ierr);       /*  r = b - Ap[k]    */
      ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
	  /* Dingwen */	/* MVM + VLO */
	  ierr = VecDot(CKSAmat1, p[k], &CKSr1);CHKERRQ(ierr);
	  CKSr1 = CKSb1 - (CKSr1 + d1*CKSp1[k] + d2*CKSp2[k] + d3*CKSp3[k]);				/* Update checksum1(r) = checksum1(b) - (checksum1(A)p[k] + d1*checksum1(p[k]) + d2*checksum2(p[k]) + d3*checksum3(p[k]); */ 
	  ierr = VecDot(CKSAmat2, p[k], &CKSr2);CHKERRQ(ierr);
	  CKSr2 = CKSb2 - (CKSr2 + d2*CKSp1[k] + d3*CKSp2[k] + d1*CKSp3[k]);				/* Update checksum2(r) = checksum2(b) - (checksum2(A)p[k] + d2*checksum1(p[k]) + d3*checksum2(p[k]) + d1*checksum3(p[k]); */
	  ierr = VecDot(CKSAmat3, p[k], &CKSr3);CHKERRQ(ierr);
	  CKSr3 = CKSb3 - (CKSr3 + d3*CKSp1[k] + d1*CKSp2[k] + d2*CKSp3[k]);				/* Update checksum3(r) = checksum3(b) - (checksum3(A)p[k] + d3*checksum1(p[k]) + d1*checksum2(p[k]) + d2*checksum3(p[k]); */
	  /* Dingwen */

      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);
      } else {
        ierr = KSP_PCApply(ksp,r,p[kp1]);CHKERRQ(ierr); /* p[kp1] = B^{-1}r */
        ierr = VecNorm(p[kp1],NORM_2,&rnorm);CHKERRQ(ierr);
		/* Dingwen */
		ierr = VecDot(C1,p[kp1],&CKSp1[kp1]);CHKERRQ(ierr);				/* Update checksum1(p[kp1]) */
		ierr = VecDot(C2,p[kp1],&CKSp2[kp1]);CHKERRQ(ierr);				/* Update checksum2(p[kp1]) */
		ierr = VecDot(C3,p[kp1],&CKSp3[kp1]);CHKERRQ(ierr);				/* Update checksum3(p[kp1]) */
		/* Dingwen */
      }
      ierr         = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->rnorm   = rnorm;
      ierr         = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->vec_sol = p[k];
      ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i,rnorm);CHKERRQ(ierr);
    }
    if (ksp->its >= ksp->max_it) {
      if (ksp->normtype != KSP_NORM_NONE) {
        ierr = (*ksp->converged)(ksp,i,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
        if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      } else ksp->reason = KSP_CONVERGED_ITS;
    }
  }

  /* make sure solution is in vector x */
  ksp->vec_sol = sol_orig;
  if (k) {
    ierr = VecCopy(p[k],sol_orig);CHKERRQ(ierr);
  }
  
  // /* Dingwen */
  // for (i=0;i<3;i++)
  // {
	// ierr = VecDot(C1,p[i],&sump1[i]);CHKERRQ(ierr);
    // ierr = VecDot(C2,p[i],&sump2[i]);CHKERRQ(ierr);
    // ierr = VecDot(C3,p[i],&sump3[i]);CHKERRQ(ierr);
    
  // }
  // ierr = VecDot(C1,r,&sumr1);CHKERRQ(ierr);
  // ierr = VecDot(C2,r,&sumr2);CHKERRQ(ierr);
  // ierr = VecDot(C3,r,&sumr3);CHKERRQ(ierr);
  // if (rank==0)
  // {
	  // for (i=0;i<3;i++)
	  // {
		  // printf ("sum1 of p[%d] = %f\n", i, sump1[i]);
		  // printf ("checksum1(p[%d]) = %f\n", i, CKSp1[i]);
		  // printf ("sum2 of p[%d] = %f\n", i, sump2[i]);
		  // printf ("checksum2(p[%d]) = %f\n", i, CKSp2[i]);
		  // printf ("sum3 of p[%d] = %f\n", i, sump3[i]);
		  // printf ("checksum3(p[%d]) = %f\n", i, CKSp3[i]);
	  // }
	  // printf ("sum1 of r = %f\n", sumr1);
	  // printf ("checksum1(r) = %f\n", CKSr1);
	  // printf ("sum2 of r = %f\n", sumr2);
	  // printf ("checksum2(r) = %f\n", CKSr2);
	  // printf ("sum3 of r = %f\n", sumr3);
	  // printf ("checksum3(r) = %f\n", CKSr3);
  // }
  // /* Dingwen */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPView_Chebyshev"
static  PetscErrorCode KSPView_Chebyshev(KSP ksp,PetscViewer viewer)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Chebyshev: eigenvalue estimates:  min = %g, max = %g\n",(double)cheb->emin,(double)cheb->emax);CHKERRQ(ierr);
    if (cheb->kspest) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Chebyshev: eigenvalues estimated using %s with translations  [%g %g; %g %g]\n",((PetscObject) cheb->kspest)->type_name,(double)cheb->tform[0],(double)cheb->tform[1],(double)cheb->tform[2],(double)cheb->tform[3]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(cheb->kspest,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      if (cheb->random) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Chebyshev: estimating eigenvalues using random right hand side\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = PetscRandomView(cheb->random,viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_Chebyshev"
static PetscErrorCode KSPDestroy_Chebyshev(KSP ksp)
{
  KSP_Chebyshev  *cheb = (KSP_Chebyshev*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy(&cheb->kspest);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&cheb->random);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevSetEigenvalues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigSet_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigSetRandom_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigGetKSP_C",NULL);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPCHEBYSHEV - The preconditioned Chebyshev iterative method

   Options Database Keys:
+   -ksp_chebyshev_eigenvalues <emin,emax> - set approximations to the smallest and largest eigenvalues
                  of the preconditioned operator. If these are accurate you will get much faster convergence.
.   -ksp_chebyshev_esteig <a,b,c,d> - estimate eigenvalues using a Krylov method, then use this
                         transform for Chebyshev eigenvalue bounds (KSPChebyshevEstEigSet())
.   -ksp_chebyshev_esteig_steps - number of estimation steps 
+   -ksp_chebyshev_esteig_random - use random right hand side for eigenvalue estimation (KSPChebyshevEstEigSetRandom())


   Level: beginner

   Notes: The Chebyshev method requires both the matrix and preconditioner to
          be symmetric positive (semi) definite.
          Only support for left preconditioning.

          Chebyshev is configured as a smoother by default, targetting the "upper" part of the spectrum.
          The user should call KSPChebyshevSetEigenvalues() if they have eigenvalue estimates.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPChebyshevSetEigenvalues(), KSPChebyshevEstEigSet(), KSPChebyshevEstEigSetRandom(), KSPRICHARDSON, KSPCG, PCMG

M*/

#undef __FUNCT__
#define __FUNCT__ "KSPCreate_Chebyshev"
PETSC_EXTERN PetscErrorCode KSPCreate_Chebyshev(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_Chebyshev  *chebyshevP;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&chebyshevP);CHKERRQ(ierr);

  ksp->data = (void*)chebyshevP;
  ierr      = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr      = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);

  chebyshevP->emin = 0.;
  chebyshevP->emax = 0.;

  chebyshevP->tform[0] = 0.0;
  chebyshevP->tform[1] = 0.1;
  chebyshevP->tform[2] = 0;
  chebyshevP->tform[3] = 1.1;
  chebyshevP->eststeps = 10;
  
  ksp->ops->setup          = KSPSetUp_Chebyshev;
  ksp->ops->solve          = KSPSolve_Chebyshev;
  ksp->ops->destroy        = KSPDestroy_Chebyshev;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_Chebyshev;
  ksp->ops->view           = KSPView_Chebyshev;
  ksp->ops->reset          = KSPReset_Chebyshev;

  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevSetEigenvalues_C",KSPChebyshevSetEigenvalues_Chebyshev);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigSet_C",KSPChebyshevEstEigSet_Chebyshev);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigSetRandom_C",KSPChebyshevEstEigSetRandom_Chebyshev);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPChebyshevEstEigGetKSP_C",KSPChebyshevEstEigGetKSP_Chebyshev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
