import numpy as np
import random
from pyrosetta import *
from utils_trX2dy.top_prob import *

def gen_rst(npz, tmpdir, params):
    if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
        dist, omega, theta, phi = npz['dist'], npz['omega'], npz['theta'], npz['phi']
        # dictionary to store Rosetta restraints
        rst = {'dist': [], 'omega': [], 'theta': [], 'phi': [], 'rep': []}
    else:
        dist = npz['dist']
        rst = {'dist': []}

    ########################################################
    # assign parameters
    ########################################################
    PCUT = 0.05  # params['PCUT']
    PCUT1 = params['PCUT1']
    EBASE = params['EBASE']
    EREP = params['EREP']
    DREP = params['DREP']
    PREP = params['PREP']
    SIGD = params['SIGD']
    SIGM = params['SIGM']
    MEFF = params['MEFF']
    DCUT = params['DCUT']
    ALPHA = params['ALPHA']

    DSTEP = params['DSTEP']
    ASTEP = np.deg2rad(params['ASTEP'])

    seq = params['seq']

    ########################################################
    # repultion restraints
    ########################################################
    # cbs = ['CA' if a=='G' else 'CB' for a in params['seq']]
    '''
    prob = np.sum(dist[:,:,5:], axis=-1)
    i,j = np.where(prob<PREP)
    prob = prob[i,j]
    for a,b,p in zip(i,j,prob):
        if b>a:
            name=tmpdir.name+"/%d.%d_rep.txt"%(a+1,b+1)
            rst_line = 'AtomPair %s %d %s %d SCALARWEIGHTEDFUNC %.2f SUMFUNC 2 CONSTANTFUNC 0.5 SIGMOID %.3f %.3f\n'%('CB',a+1,'CB',b+1,-0.5,SIGD,SIGM)
            rst['rep'].append([a,b,p,rst_line])
    print("rep restraints:   %d"%(len(rst['rep'])))
    '''

    ########################################################
    # dist: 0..20A
    ########################################################
    nres = dist.shape[0]
    bins = np.array([4.25 + DSTEP * i for i in range(32)])
    prob = np.sum(dist[:, :, 5:], axis=-1)
    bkgr = np.array((bins / DCUT) ** ALPHA)
    attr = -np.log((dist[:, :, 5:] + MEFF) / (dist[:, :, -1][:, :, None] * bkgr[None, None, :]+1e-6)) + EBASE
    repul = np.maximum(attr[:, :, 0], np.zeros((nres, nres)))[:, :, None] + np.array(EREP)[None, None, :]
    dist = np.concatenate([repul, attr], axis=-1)
    bins = np.concatenate([DREP, bins])
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    nbins = 35
    step = 0.5
    for a, b, p in zip(i, j, prob):
        if b > a:
            name = tmpdir.name + "/%d.%d.txt" % (a + 1, b + 1)
            with open(name, "w") as f:
                f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                f.write('y_axis' + '\t%.3f' * nbins % tuple(dist[a, b]) + '\n')
                f.close()
            rst_line = 'AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f' % ('CB', a + 1, 'CB', b + 1, name, 1.0, step)
            rst['dist'].append([a, b, p, rst_line])
    print("dist restraints:  %d" % (len(rst['dist'])))

    if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
        ########################################################
        # omega: -pi..pi
        ########################################################
        nbins = omega.shape[2] - 1 + 4
        bins = np.linspace(-np.pi - 1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
        prob = np.sum(omega[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        omega = -np.log((omega + MEFF) / (omega[:, :, -1] + MEFF)[:, :, None])
        omega = np.concatenate([omega[:, :, -2:], omega[:, :, 1:], omega[:, :, 1:3]], axis=-1)
        for a, b, p in zip(i, j, prob):
            if b > a:
                name = tmpdir.name + "/%d.%d_omega.txt" % (a + 1, b + 1)
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.5f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.5f' * nbins % tuple(omega[a, b]) + '\n')
                    f.close()
                rst_line = 'Dihedral CA %d CB %d CB %d CA %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, b + 1, b + 1, name, 1.0, ASTEP)
                rst['omega'].append([a, b, p, rst_line])
        print("omega restraints: %d" % (len(rst['omega'])))

        ########################################################
        # theta: -pi..pi
        ########################################################
        prob = np.sum(theta[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        theta = -np.log((theta + MEFF) / (theta[:, :, -1] + MEFF)[:, :, None])
        theta = np.concatenate([theta[:, :, -2:], theta[:, :, 1:], theta[:, :, 1:3]], axis=-1)
        for a, b, p in zip(i, j, prob):
            if b != a:
                name = tmpdir.name + "/%d.%d_theta.txt" % (a + 1, b + 1)
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.3f' * nbins % tuple(theta[a, b]) + '\n')
                    f.close()
                rst_line = 'Dihedral N %d CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, a + 1, b + 1, name, 1.0, ASTEP)
                rst['theta'].append([a, b, p, rst_line])
                # if a==0 and b==9:
                #    with open(name,'r') as f:
                #        print(f.read())
        print("theta restraints: %d" % (len(rst['theta'])))

        ########################################################
        # phi: 0..pi
        ########################################################
        nbins = phi.shape[2] - 1 + 4
        bins = np.linspace(-1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
        prob = np.sum(phi[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        phi = -np.log((phi + MEFF) / (phi[:, :, -1] + MEFF)[:, :, None])
        phi = np.concatenate([np.flip(phi[:, :, 1:3], axis=-1), phi[:, :, 1:], np.flip(phi[:, :, -2:], axis=-1)], axis=-1)
        for a, b, p in zip(i, j, prob):
            if b != a:
                name = tmpdir.name + "/%d.%d_phi.txt" % (a + 1, b + 1)
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.3f' * nbins % tuple(phi[a, b]) + '\n')
                    f.close()
                rst_line = 'Angle CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, b + 1, name, 1.0, ASTEP)
                rst['phi'].append([a, b, p, rst_line])
                # if a==0 and b==9:
                #    with open(name,'r') as f:
                #        print(f.read())

        print("phi restraints:   %d" % (len(rst['phi'])))

    return rst

def gen_rst_af2(npz, tmpdir, params):
    if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
        raise RuntimeError("AF2 Not support ")
    else:
        dist = npz['dist']
        af_bins = npz['bins']
        rst = {'dist': []}
    ########################################################
    # assign parameters
    ########################################################
    PCUT = 0.0025 # params['PCUT']
    EBASE = params['EBASE']#-0.5
    EREP = params['EREP']#[10.0,3.0,0.5]
    MEFF = params['MEFF']#0.0001
    DCUT = params['DCUT']#0.05
    ALPHA = params['ALPHA']

    ########################################################
    # dist: 0..21.6875A
    ########################################################
    nres = dist.shape[0]
    bins = af_bins[5:-1]
    prob = np.sum(dist[:, :, 6:-1], axis=-1)
    bkgr = np.array((bins / DCUT) ** ALPHA)
    attr = -np.log((dist[:, :, 6:-1] + MEFF) / (dist[:, :, -2][:, :, None] * bkgr[None, None, -1]+1e-6)) + EBASE
    repul = np.maximum(attr[:, :, 0], np.zeros((nres, nres)))[:, :, None] + np.array(EREP)[None, None, :]
    dist = np.concatenate([repul, attr], axis=-1)
    DREP = [0.0,2.325,3.575]
    bins = np.concatenate([DREP, bins])
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    nbins = 60
    step = 0.3125
    for a, b, p in zip(i, j, prob):
        if b > a:
            name = tmpdir.name + "/%d.%d.txt" % (a + 1, b + 1)
            with open(name, "w") as f:
                f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                f.write('y_axis' + '\t%.3f' * nbins % tuple(dist[a, b]) + '\n')
                f.close()
            rst_line = 'AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f' % ('CA', a + 1, 'CA', b + 1, name, 1.0, step)
            rst['dist'].append([a, b, p, rst_line])
    print("dist restraints:  %d" % (len(rst['dist'])))

    if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
        raise RuntimeError("AF2 Not support ")
    return rst

def gen_idp_rst(npz, tmpdir, params):
    if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
        dist, omega, theta, phi, idr = npz['dist'], npz['omega'], npz['theta'], npz['phi'],npz['idr']
        # native_dist, native_omega, native_theta, native_phi = native_npz['dist'], native_npz['omega'], native_npz['theta'], native_npz['phi']
        # dictionary to store Rosetta restraints
        rst = {'dist': [], 'omega': [], 'theta': [], 'phi': [], 'rep': []}
    else:
        dist = npz['dist']
        idr = npz['idr']
        # native_dist = native_npz['dist']
        rst = {'dist': []}

    ########################################################
    # assign parameters
    ########################################################
    PCUT = 0.05  # params['PCUT']
    PCUT1 = params['PCUT1']
    EBASE = params['EBASE']
    EREP = params['EREP']
    DREP = params['DREP']
    PREP = params['PREP']
    SIGD = params['SIGD']
    SIGM = params['SIGM']
    MEFF = params['MEFF']
    DCUT = params['DCUT']
    ALPHA = params['ALPHA']

    DSTEP = params['DSTEP']
    ASTEP = np.deg2rad(params['ASTEP'])

    seq = params['seq']

    ########################################################
    # repultion restraints
    ########################################################
    # cbs = ['CA' if a=='G' else 'CB' for a in params['seq']]
    '''
    prob = np.sum(dist[:,:,5:], axis=-1)
    i,j = np.where(prob<PREP)
    prob = prob[i,j]
    for a,b,p in zip(i,j,prob):
        if b>a:
            name=tmpdir.name+"/%d.%d_rep.txt"%(a+1,b+1)
            rst_line = 'AtomPair %s %d %s %d SCALARWEIGHTEDFUNC %.2f SUMFUNC 2 CONSTANTFUNC 0.5 SIGMOID %.3f %.3f\n'%('CB',a+1,'CB',b+1,-0.5,SIGD,SIGM)
            rst['rep'].append([a,b,p,rst_line])
    print("rep restraints:   %d"%(len(rst['rep'])))
    '''

    ########################################################
    # dist: 0..20A
    ########################################################
    nres = dist.shape[0]
    bins = np.array([4.25 + DSTEP * i for i in range(32)])
    prob = np.sum(dist[:, :, 5:], axis=-1)
    idr_bkgr = (bins[None,None,:]/bins[np.argmax(dist[:,:,5:],axis=-1)][:,:,None]) ** ALPHA
    idr_attr = -np.log((dist[:, :, 5:] + MEFF) / (np.max(dist[:,:,5:],axis=-1)[:, :, None] * idr_bkgr+1e-6)) + EBASE
    bkgr = np.array((bins / DCUT) ** ALPHA)
    attr = -np.log((dist[:, :, 5:] + MEFF) / (dist[:, :, -1][:, :, None] * bkgr[None, None, :] + 1e-6)) + EBASE

    repul = np.maximum(attr[:, :, 0], np.zeros((nres, nres)))[:, :, None] + np.array(EREP)[None, None, :]
    dist = np.concatenate([repul, attr], axis=-1)
    idr_dist = np.concatenate([repul, idr_attr], axis=-1)
    bins = np.concatenate([DREP, bins])
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    nbins = 35
    step = 0.5
    for a, b, p in zip(i, j, prob):
        if b > a:
            name = tmpdir.name + "/%d.%d.txt" % (a + 1, b + 1)
            if idr[a,b]:
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.3f' * nbins % tuple(idr_dist[a, b]) + '\n')
                    f.close()
            else:
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.3f' * nbins % tuple(dist[a, b]) + '\n')
                    f.close()
            rst_line = 'AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f' % ('CB', a + 1, 'CB', b + 1, name, 1.0, step)
            rst['dist'].append([a, b, p, rst_line])
    print("dist restraints:  %d" % (len(rst['dist'])))

    if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
        ########################################################
        # omega: -pi..pi
        ########################################################
        nbins = omega.shape[2] - 1 + 4
        bins = np.linspace(-np.pi - 1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
        prob = np.sum(omega[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        idr_omega = -np.log((omega + MEFF) / (np.max(omega,axis=-1) + MEFF)[:, :, None])
        idr_omega = np.concatenate([idr_omega[:, :, -2:], idr_omega[:, :, 1:], idr_omega[:, :, 1:3]], axis=-1)
        omega = -np.log((omega + MEFF) / (omega[:, :, -1] + MEFF)[:, :, None])
        omega = np.concatenate([omega[:, :, -2:], omega[:, :, 1:], omega[:, :, 1:3]], axis=-1)
        for a, b, p in zip(i, j, prob):
            if b > a:
                name = tmpdir.name + "/%d.%d_omega.txt" % (a + 1, b + 1)
                if idr[a,b]:
                    with open(name, "w") as f:
                        f.write('x_axis' + '\t%.5f' * nbins % tuple(bins) + '\n')
                        f.write('y_axis' + '\t%.5f' * nbins % tuple(idr_omega[a, b]) + '\n')
                        f.close()
                else:
                    with open(name, "w") as f:
                        f.write('x_axis' + '\t%.5f' * nbins % tuple(bins) + '\n')
                        f.write('y_axis' + '\t%.5f' * nbins % tuple(omega[a, b]) + '\n')
                        f.close()
                rst_line = 'Dihedral CA %d CB %d CB %d CA %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, b + 1, b + 1, name, 1.0, ASTEP)
                rst['omega'].append([a, b, p, rst_line])
        print("omega restraints: %d" % (len(rst['omega'])))

        ########################################################
        # theta: -pi..pi
        ########################################################
        prob = np.sum(theta[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        idr_theta = -np.log((theta + MEFF) / (np.max(theta,axis=-1) + MEFF)[:, :, None])
        idr_theta = np.concatenate([idr_theta[:, :, -2:], idr_theta[:, :, 1:], idr_theta[:, :, 1:3]], axis=-1)
        theta = -np.log((theta + MEFF) / (theta[:, :, -1] + MEFF)[:, :, None])
        theta = np.concatenate([theta[:, :, -2:], theta[:, :, 1:], theta[:, :, 1:3]], axis=-1)
        for a, b, p in zip(i, j, prob):
            if b != a:
                name = tmpdir.name + "/%d.%d_theta.txt" % (a + 1, b + 1)
                if idr[a,b]:
                    with open(name, "w") as f:
                        f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                        f.write('y_axis' + '\t%.3f' * nbins % tuple(idr_theta[a, b]) + '\n')
                        f.close()
                else:
                    with open(name, "w") as f:
                        f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                        f.write('y_axis' + '\t%.3f' * nbins % tuple(theta[a, b]) + '\n')
                        f.close()
                rst_line = 'Dihedral N %d CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, a + 1, b + 1, name, 1.0, ASTEP)
                rst['theta'].append([a, b, p, rst_line])
                # if a==0 and b==9:
                #    with open(name,'r') as f:
                #        print(f.read())
        print("theta restraints: %d" % (len(rst['theta'])))

        ########################################################
        # phi: 0..pi
        ########################################################
        nbins = phi.shape[2] - 1 + 4
        bins = np.linspace(-1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
        prob = np.sum(phi[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        idr_phi = -np.log((phi + MEFF) / (np.max(phi,axis=-1) + MEFF)[:, :, None])
        idr_phi = np.concatenate([np.flip(idr_phi[:, :, 1:3], axis=-1), idr_phi[:, :, 1:], np.flip(idr_phi[:, :, -2:], axis=-1)], axis=-1)
        phi = -np.log((phi + MEFF) / (phi[:, :, -1] + MEFF)[:, :, None])
        phi = np.concatenate([np.flip(phi[:, :, 1:3], axis=-1), phi[:, :, 1:], np.flip(phi[:, :, -2:], axis=-1)],axis=-1)
        for a, b, p in zip(i, j, prob):
            if b != a:
                name = tmpdir.name + "/%d.%d_phi.txt" % (a + 1, b + 1)
                if idr[a,b]:
                    with open(name, "w") as f:
                        f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                        f.write('y_axis' + '\t%.3f' * nbins % tuple(idr_phi[a, b]) + '\n')
                        f.close()
                else:
                    with open(name, "w") as f:
                        f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                        f.write('y_axis' + '\t%.3f' * nbins % tuple(phi[a, b]) + '\n')
                        f.close()
                rst_line = 'Angle CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, b + 1, name, 1.0, ASTEP)
                rst['phi'].append([a, b, p, rst_line])
                # if a==0 and b==9:
                #    with open(name,'r') as f:
                #        print(f.read())

        print("phi restraints:   %d" % (len(rst['phi'])))

    return rst

def ling_sumlt(test,b,bin,bool,rg=5):
    '''
    test.shape=W,H,N  array,float32
    b.shape = W,H,N  array,float32
    bool.shape = W,H  array,bool
    bin.shape = N  array,int
    '''
    t = test.copy()
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if bool[i,j]:
                low_index = np.argsort(b[i,j])[:rg].min()-1
                heigh_index = np.argsort(b[i,j])[:rg].max()+1
                if low_index<0:
                    low_index += 1
                if heigh_index>=len(bin):
                    heigh_index -= 1

                t[i,j][np.argsort(b[i,j])[:rg]] = (bin[np.argsort(b[i,j])[:rg]]-bin[heigh_index])/(bin[low_index]-bin[heigh_index])*(t[i,j][low_index]-t[i,j][heigh_index])+t[i,j][heigh_index]
    return t
def pros(Dist,Omega=None,Theta_asym=None,Phi_asym=None,angle=False):
    Sdist,Somega,Stheta,Sphi = [],[],[],[]
    SSdist,SSomega,SStheta,SSphi = [],[],[],[]
    for i in range(len(Dist)):
        dist = Dist[i]
        Tdist = dist.reshape(-1, 1)
        Adist = np.array([np.arange(2, 20.5, 0.5).tolist()])
        Jdist = np.array([(Adist < Tdist).sum(axis=1)]).reshape(dist.shape[0], dist.shape[1])
        Jdist = np.where(Jdist == 0, 0, Jdist)
        Jdist = np.where(Jdist >= 37, 0, Jdist)
        sdist = np.eye(37)[Jdist]  # 用NumPy实现one-hot编码
        Sdist.append(dist)
        SSdist.append(sdist)
        if angle:
            omega = Omega[i]
            Tomega = omega.reshape(-1, 1)
            Aomega = np.array([np.arange(-np.pi, np.pi, np.pi / 12).tolist()])
            Jomega = np.array([(Aomega < Tomega).sum(axis=1)]).reshape(omega.shape[0], omega.shape[1])
            Jomega = np.where(Jdist == 0, 0, Jomega)
            Jomega = np.where(Jdist >= 37, 0, Jomega)
            somega = np.eye(25)[Jomega]
            Somega.append(omega)
            SSomega.append(somega)

            theta_asym = Theta_asym[i]
            Ttheta_asym = theta_asym.reshape(-1, 1)
            Atheta_asym = np.array([np.arange(-np.pi, np.pi, np.pi / 12).tolist()])
            Jtheta_asym = np.array([(Atheta_asym < Ttheta_asym).sum(axis=1)]).reshape(theta_asym.shape[0],theta_asym.shape[1])
            Jtheta_asym = np.where(Jdist == 0, 0, Jtheta_asym)
            Jtheta_asym = np.where(Jdist >= 37, 0, Jtheta_asym)
            stheta_asym = np.eye(25)[Jtheta_asym]
            Stheta.append(theta_asym)
            SStheta.append(stheta_asym)

            phi_asym = Phi_asym[i]
            Tphi_asym = theta_asym.reshape(-1, 1)
            Aphi_asym = np.array([np.arange(0, np.pi, np.pi / 12).tolist()])
            Jphi_asym = np.array([(Aphi_asym < Tphi_asym).sum(axis=1)]).reshape(phi_asym.shape[0], phi_asym.shape[1])
            Jphi_asym = np.where(Jdist == 0, 0, Jphi_asym)
            Jphi_asym = np.where(Jdist >= 37, 0, Jphi_asym)
            sphi_asym = np.eye(13)[Jphi_asym]
            Sphi.append(phi_asym)
            SSphi.append(sphi_asym)
    l = len(Sdist)
    Sdist = [p.reshape(1, p.shape[0], p.shape[1]) for p in Sdist]
    SSdist = [p.reshape(1, p.shape[0], p.shape[1], p.shape[2]) for p in SSdist]
    if angle:
        Somega = [p.reshape(1, p.shape[0], p.shape[1]) for p in Somega]
        SSomega = [p.reshape(1, p.shape[0], p.shape[1], p.shape[2]) for p in SSomega]

        Stheta = [p.reshape(1, p.shape[0], p.shape[1]) for p in Stheta]
        SStheta = [p.reshape(1, p.shape[0], p.shape[1], p.shape[2]) for p in SStheta]

        Sphi = [p.reshape(1, p.shape[0], p.shape[1]) for p in Sphi]
        SSphi = [p.reshape(1, p.shape[0], p.shape[1], p.shape[2]) for p in SSphi]
        return np.array(SSdist),np.array(SStheta),np.array(SSomega),np.array(SSphi)
    else:
        return np.array(SSdist)
def get_normal_distribution_probabilities(C, mean, std):
    x = np.arange(C)
    normal_probabilities = (1 / (np.sqrt(2 * np.pi * std ** 2)) *
                            np.exp(-((x - mean) ** 2) / (2 * std ** 2)))
    return normal_probabilities
def get_sample(SSdist):  # Sdist为[20,1,H,W,37]的onehot矩阵
    # 将Sdist维度调整为[20,H,W,37]，去掉单通道维度
    Sdist = SSdist[:, 0, :, :, :]

    # 计算所有分布的总和
    all_dist = np.sum(Sdist, axis=0)
    num = Sdist.shape[0]
    H, W, C = Sdist.shape[1], Sdist.shape[2], Sdist.shape[3]

    # 初始化概率矩阵
    all_prob = np.zeros((H, W, C), dtype=float)

    for i in range(H):
        for j in range(W):
            n_list = np.where(all_dist[i, j] != 0)[0]
            for k in n_list:
                if all_dist[i, j, k] < num / 3:
                    std = 1.5
                elif all_dist[i, j, k] > 2 * num / 3:
                    std = 0.5
                else:
                    std = 1.0

                normal_probabilities = get_normal_distribution_probabilities(C, k, std)
                for idx in range(num):
                    if Sdist[idx, i, j, k] == 1:
                        all_prob[i, j, :] += normal_probabilities

    return all_prob / num
def gen_gpcr_rst(npz,known_npz, tmpdir, params):
    if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
        dist, omega, theta, phi, idr = npz['dist'], npz['omega'], npz['theta'], npz['phi'],npz['idr']
        known_dist, known_omega, known_theta, known_phi = known_npz['dist'], known_npz['omega'], known_npz['theta_asym'], known_npz['phi_asym']

        pros_native = pros(known_dist, known_omega, known_theta, known_phi, angle=True)
        cate_dist,cate_theta,cate_omega,cate_phi = (
            get_sample(pros_native[0]),get_sample(pros_native[1]),get_sample(pros_native[2]),get_sample(pros_native[3]))
        # dictionary to store Rosetta restraints
        rst = {'dist': [], 'omega': [], 'theta': [], 'phi': [], 'rep': []}
    else:
        dist = npz['dist']
        idr = npz['idr']
        known_dist = known_npz['dist']
        pros_native = pros(known_dist,angle=False)
        cate_dist = get_sample(pros_native)
        rst = {'dist': []}

    ########################################################
    # assign parameters
    ########################################################
    PCUT = 0.05  # params['PCUT']
    PCUT1 = params['PCUT1']
    EBASE = params['EBASE']
    EREP = params['EREP']
    DREP = params['DREP']
    PREP = params['PREP']
    SIGD = params['SIGD']
    SIGM = params['SIGM']
    MEFF = params['MEFF']
    DCUT = params['DCUT']
    ALPHA = params['ALPHA']

    DSTEP = params['DSTEP']
    ASTEP = np.deg2rad(params['ASTEP'])

    seq = params['seq']

    ########################################################
    # repultion restraints
    ########################################################
    # cbs = ['CA' if a=='G' else 'CB' for a in params['seq']]
    '''
    prob = np.sum(dist[:,:,5:], axis=-1)
    i,j = np.where(prob<PREP)
    prob = prob[i,j]
    for a,b,p in zip(i,j,prob):
        if b>a:
            name=tmpdir.name+"/%d.%d_rep.txt"%(a+1,b+1)
            rst_line = 'AtomPair %s %d %s %d SCALARWEIGHTEDFUNC %.2f SUMFUNC 2 CONSTANTFUNC 0.5 SIGMOID %.3f %.3f\n'%('CB',a+1,'CB',b+1,-0.5,SIGD,SIGM)
            rst['rep'].append([a,b,p,rst_line])
    print("rep restraints:   %d"%(len(rst['rep'])))
    '''

    ########################################################
    # dist: 0..20A
    ########################################################
    nres = dist.shape[0]
    bins = np.array([4.25 + DSTEP * i for i in range(32)])
    prob = np.sum(dist[:, :, 5:], axis=-1)
    bkgr = np.array((bins / DCUT) ** ALPHA)
    attr = -np.log((dist[:, :, 5:] + MEFF) / (dist[:, :, -1][:, :, None] * bkgr[None, None, :]+1e-6)) + EBASE
    repul = np.maximum(attr[:, :, 0], np.zeros((nres, nres)))[:, :, None] + np.array(EREP)[None, None, :]
    dist = np.concatenate([repul, attr], axis=-1)
    bins = np.concatenate([DREP, bins])
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]

    attr = -np.log((cate_dist[:, :, 5:] + MEFF) / (cate_dist[:, :, -1][:, :, None] * bkgr[None, None, :] + 1e-6)) + EBASE
    repul = np.maximum(attr[:, :, 0], np.zeros((nres, nres)))[:, :, None] + np.array(EREP)[None, None, :]
    cate_dist = np.concatenate([repul, attr], axis=-1)
    dist = ling_sumlt(dist,cate_dist,bins,idr)

    nbins = 35
    step = 0.5
    for a, b, p in zip(i, j, prob):
        if b > a:
            name = tmpdir.name + "/%d.%d.txt" % (a + 1, b + 1)
            with open(name, "w") as f:
                f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                f.write('y_axis' + '\t%.3f' * nbins % tuple(dist[a, b]) + '\n')
                f.close()
            rst_line = 'AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f' % ('CB', a + 1, 'CB', b + 1, name, 1.0, step)
            rst['dist'].append([a, b, p, rst_line])
    print("dist restraints:  %d" % (len(rst['dist'])))

    if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
        ########################################################
        # omega: -pi..pi
        ########################################################
        nbins = omega.shape[2] - 1 + 4
        bins = np.linspace(-np.pi - 1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
        prob = np.sum(omega[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        omega = -np.log((omega + MEFF) / (omega[:, :, -1] + MEFF)[:, :, None])
        omega = np.concatenate([omega[:, :, -2:], omega[:, :, 1:], omega[:, :, 1:3]], axis=-1)

        cate_omega = -np.log((cate_omega + MEFF) / (cate_omega[:, :, -1] + MEFF)[:, :, None])
        cate_omega = np.concatenate([cate_omega[:, :, -2:], cate_omega[:, :, 1:], cate_omega[:, :, 1:3]], axis=-1)
        
        omega = ling_sumlt(omega,cate_omega,bins,idr)
        for a, b, p in zip(i, j, prob):
            if b > a:
                name = tmpdir.name + "/%d.%d_omega.txt" % (a + 1, b + 1)
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.5f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.5f' * nbins % tuple(omega[a, b]) + '\n')
                    f.close()
                rst_line = 'Dihedral CA %d CB %d CB %d CA %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, b + 1, b + 1, name, 1.0, ASTEP)
                rst['omega'].append([a, b, p, rst_line])
        print("omega restraints: %d" % (len(rst['omega'])))

        ########################################################
        # theta: -pi..pi
        ########################################################
        prob = np.sum(theta[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        theta = -np.log((theta + MEFF) / (theta[:, :, -1] + MEFF)[:, :, None])
        theta = np.concatenate([theta[:, :, -2:], theta[:, :, 1:], theta[:, :, 1:3]], axis=-1)

        cate_theta = -np.log((cate_theta + MEFF) / (cate_theta[:, :, -1] + MEFF)[:, :, None])
        cate_theta = np.concatenate([cate_theta[:, :, -2:], cate_theta[:, :, 1:], cate_theta[:, :, 1:3]], axis=-1)
        
        theta = ling_sumlt(theta,cate_theta,bins,idr)
        for a, b, p in zip(i, j, prob):
            if b != a:
                name = tmpdir.name + "/%d.%d_theta.txt" % (a + 1, b + 1)
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.3f' * nbins % tuple(theta[a, b]) + '\n')
                    f.close()
                rst_line = 'Dihedral N %d CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, a + 1, b + 1, name, 1.0, ASTEP)
                rst['theta'].append([a, b, p, rst_line])
                # if a==0 and b==9:
                #    with open(name,'r') as f:
                #        print(f.read())
        print("theta restraints: %d" % (len(rst['theta'])))

        ########################################################
        # phi: 0..pi
        ########################################################
        nbins = phi.shape[2] - 1 + 4
        bins = np.linspace(-1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
        prob = np.sum(phi[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        phi = -np.log((phi + MEFF) / (phi[:, :, -1] + MEFF)[:, :, None])
        phi = np.concatenate([np.flip(phi[:, :, 1:3], axis=-1), phi[:, :, 1:], np.flip(phi[:, :, -2:], axis=-1)],axis=-1)

        cate_phi = -np.log((cate_phi + MEFF) / (cate_phi[:, :, -1] + MEFF)[:, :, None])
        cate_phi = np.concatenate([np.flip(cate_phi[:, :, 1:3], axis=-1), cate_phi[:, :, 1:], np.flip(cate_phi[:, :, -2:], axis=-1)],axis=-1)

        phi = ling_sumlt(phi,cate_phi,bins,idr)
        for a, b, p in zip(i, j, prob):
            if b != a:
                name = tmpdir.name + "/%d.%d_phi.txt" % (a + 1, b + 1)
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.3f' * nbins % tuple(phi[a, b]) + '\n')
                    f.close()
                rst_line = 'Angle CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, b + 1, name, 1.0, ASTEP)
                rst['phi'].append([a, b, p, rst_line])
                # if a==0 and b==9:
                #    with open(name,'r') as f:
                #        print(f.read())

        print("phi restraints:   %d" % (len(rst['phi'])))

    return rst

def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres):
        phi, psi = random_dihedral()
        pose.set_phi(i, phi)
        pose.set_psi(i, psi)
        pose.set_omega(i, 180)

    return (pose)


# pick phi/psi randomly from:
# -140  153 180 0.135 B
# -72  145 180 0.155 B
# -122  117 180 0.073 B
# -82  -14 180 0.122 A
# -61  -41 180 0.497 A
#  57   39 180 0.018 L
def random_dihedral():
    phi = 0
    psi = 0
    r = random.random()
    if (r <= 0.135):
        phi = -140
        psi = 153
    elif (r > 0.135 and r <= 0.29):
        phi = -72
        psi = 145
    elif (r > 0.29 and r <= 0.363):
        phi = -122
        psi = 117
    elif (r > 0.363 and r <= 0.485):
        phi = -82
        psi = -14
    elif (r > 0.485 and r <= 0.982):
        phi = -61
        psi = -41
    else:
        phi = 57
        psi = 39
    return (phi, psi)


def remove_clash(scorefxn, mover, pose):
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)


def add_rst(pose, rst, sep1, sep2, params, nogly=False):
    pcut = params['PCUT']
    seq = params['seq']

    array = []

    if nogly == True:
        array += [line for a, b, p, line in rst['dist'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != 'G' and seq[b] != 'G' and p >= pcut]
        if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
            array += [line for a, b, p, line in rst['omega'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != 'G' and seq[b] != 'G' and p >= pcut + 0.5]  # 0.5
            array += [line for a, b, p, line in rst['theta'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != 'G' and seq[b] != 'G' and p >= pcut + 0.5]  # 0.5
            array += [line for a, b, p, line in rst['phi'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != 'G' and seq[b] != 'G' and p >= pcut + 0.6]  # 0.6
    else:
        array += [line for a, b, p, line in rst['dist'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut]
        if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
            array += [line for a, b, p, line in rst['omega'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.5]
            array += [line for a, b, p, line in rst['theta'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.5]
            array += [line for a, b, p, line in rst['phi'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.6]  # 0.6

    if len(array) < 1:
        return

    random.shuffle(array)

    # save to file
    tmpname = params['TDIR'] + '/minimize.cst'
    with open(tmpname, 'w') as f:
        for line in array:
            f.write(line + '\n')
        f.close()

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(tmpname)
    constraints.add_constraints(True)
    constraints.apply(pose)

    os.remove(tmpname)

def add_idr_rst(pose, rst, idr, params, nogly=False):
    pcut = params['PCUT']
    seq = params['seq']

    array = []

    if nogly == True:
        array += [line for a, b, p, line in rst['dist'] if idr[a,b] and seq[a] != 'G' and seq[b] != 'G' and p >= pcut]
        if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
            array += [line for a, b, p, line in rst['omega'] if idr[a,b] and seq[a] != 'G' and seq[b] != 'G' and p >= pcut + 0.5]  # 0.5
            array += [line for a, b, p, line in rst['theta'] if idr[a,b] and seq[a] != 'G' and seq[b] != 'G' and p >= pcut + 0.5]  # 0.5
            array += [line for a, b, p, line in rst['phi'] if idr[a,b] and seq[a] != 'G' and seq[b] != 'G' and p >= pcut + 0.6]  # 0.6
    else:
        array += [line for a, b, p, line in rst['dist'] if idr[a,b] and p >= pcut]
        if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
            array += [line for a, b, p, line in rst['omega'] if idr[a,b] and p >= pcut + 0.5]
            array += [line for a, b, p, line in rst['theta'] if idr[a,b] and p >= pcut + 0.5]
            array += [line for a, b, p, line in rst['phi'] if idr[a,b] and p >= pcut + 0.6]  # 0.6

    if len(array) < 1:
        return

    random.shuffle(array)

    # save to file
    tmpname = params['TDIR'] + '/minimize.cst'
    with open(tmpname, 'w') as f:
        for line in array:
            f.write(line + '\n')
        f.close()

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(tmpname)
    constraints.add_constraints(True)
    constraints.apply(pose)

    os.remove(tmpname)

def cal_cscore(npz):
    prob, sep = top_dist(npz['dist'], 12)
    if os.path.exists("good_temp.txt"):
        cscore = 0.9342 * prob + 0.2333 * sep + 0.0957
    else:
        cscore = 1.158 * prob + 0.1878 * sep + 0.0318
    if cscore > 1:
        cscore = 1.0
    elif cscore < 0.1:
        cscore = 0.1
    with open("cscore.txt", 'w+') as f1:
        print("Estimated TM-score of the top-predicted model: ", round(cscore, 2), file=f1)
    return cscore



def get_conda_pth(name='tRS'):
    for line in os.popen(f'conda info --envs|grep {name}'):
        if line.split()[0] == name:
            return line.split()[1]