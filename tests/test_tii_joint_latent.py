import numpy as np
from vioexplain.tii.joint_latent import system_matrices,solve_group_correction,calibrate_tau

def test_group_solver_matches_closed_form_for_isotropic_prior():
    rng=np.random.default_rng(71);G=rng.normal(size=(3,16,4));tau=1.2
    actual,info=solve_group_correction(np.eye(4)*2,G,tau)
    norm=np.linalg.norm(G,axis=1,keepdims=True)
    expected=G/2*np.maximum(0,1-tau/norm)
    np.testing.assert_allclose(actual,expected,atol=1e-9)
    assert info['converged']

def test_normal_calibration_controls_zero_solution_at_max_threshold():
    rng=np.random.default_rng(23);G=rng.normal(size=(10,24,5))
    tau=calibrate_tau(G,1.)
    delta,info=solve_group_correction(np.eye(5),G,tau)
    np.testing.assert_array_equal(delta,np.zeros_like(delta))

def test_joint_gradient_matches_finite_difference_with_intercept():
    rng=np.random.default_rng(13);Y=rng.normal(size=(7,3));mu=rng.normal(size=(7,3))
    A=rng.normal(size=(3,3));b=rng.normal(size=3);wf=np.array([.8,1.2,1.5]);wc=np.array([.9,.7,1.1])
    H,G=system_matrices(Y,mu,A,b,wf,wc);delta=rng.normal(size=Y.shape)*.1
    analytic=delta@H-G
    def loss(d):
        Z=Y-d
        return .5*np.sum(((Z-mu)*wf)**2)+.5*np.sum(((Z@A.T-b)*wc)**2)
    numeric=np.zeros_like(delta);eps=1e-6
    for ij in np.ndindex(delta.shape):
        shift=np.zeros_like(delta);shift[ij]=eps
        numeric[ij]=(loss(delta+shift)-loss(delta-shift))/(2*eps)
    np.testing.assert_allclose(analytic,numeric,atol=1e-7)
