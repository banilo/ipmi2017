"""
HCP: sparse region-network estimator

"One of the greatest challenges left for systems
neuroscience is to understand the normal and dysfunctional
operations of the cerebral cortex by relating local and global
patterns of activity [...]."

Buzsaki 2007 Nature
"""
print __doc__

import spams

import os
import os.path as op
import numpy as np
import glob
from scipy.linalg import norm
import nibabel as nib
from sklearn.grid_search import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from matplotlib import pylab as plt
from nilearn.image import concat_imgs, resample_img
import joblib
import time
import scipy.sparse as ssp
from nilearn.image import index_img
from nilearn import datasets
from scipy.stats import zscore

FORCE_TWO_CLASSES = False
# REG_PEN = 'l1'
# REG_PEN = 'l2'
# REG_PEN = 'sparse-group-lasso-l2'
# REG_PEN = 'sparse-group-lasso-linf'
# REG_PEN = 'tree-l0'
# REG_PEN = 'tree-l2'
# REG_PEN = 'trace-norm'
MY_MAX_IT = 100
MY_DATA_RATIO = 100
N_JOBS = 5

# 4 -> 51 regions
# 9 -> 100 regions
# 14 -> 148 regions
# 19 -> 196 regions
# 25 -> 250 regions
# 30 -> 320 regions
# 31 -> 357 regions
# 32 -> 397 regions
# 33 -> 440 regions
# 35 -> 508 regions
reg_atlas_index = 19

# REGNET_RATIO_GRID = [1000, 100, 50, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.02, 0.01, 0.001]
REGNET_RATIO_GRID = [1]

LAMBDA_GRID = np.logspace(-2, 1, 9)
# LAMBDA_GRID = np.array([  0.01      ,   0.02371374,   0.05623413,   0.13335214,
#          0.31622777,   0.74989421,   1.77827941,   4.21696503,  10.        ])
# LAMBDA_GRID = np.array([ 0.01      ,   0.02371374,   0.05623413, 0.13335214 ])
# LAMBDA_GRID = np.array([ 0.01      ])
# LAMBDA_GRID = np.array([ 0.01      ,   0.02371374 ])

RES_NAME = 'srne_benchmark_dataratio100_loggrid_zerobrainlevel_weighted'

if FORCE_TWO_CLASSES:
    RES_NAME += '_2cl'
WRITE_DIR = op.join(os.getcwd(), RES_NAME)
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)

REGS = ['tree-l2']

# REGS = ['tree-l0', 'tree-l2',
#     'sparse-group-lasso-linf', 'sparse-group-lasso-l2',
#     'l1', 'l2',
#     'trace-norm']

on_server = op.exists(r'/storage')

##############################################################################
# load+preprocess data
##############################################################################

# load the mask
mask_img = 'grey10_icbm_3mm_bin.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
mask_nvox = nifti_masker.mask_img_.get_data().sum()
print('Mask shape is %i/%i/%i' % nifti_masker.mask_img_.get_data().shape)
print('Mask voxels: %i' % mask_nvox)

# load atlas rois
# atlas_path = 'resources/aal.nii'  # HACK !
crad = datasets.fetch_craddock_2012_atlas()
# 4 -> 51 regions
# 9 -> 100 regions
# 14 -> 148 regions
# 19 -> 196 regions
# 25 -> 250 regions
# 30 -> 320 regions
# 31 -> 357 regions
# 32 -> 397 regions
# 33 -> 440 regions
# 35 -> 508 regions
atlas_nii = index_img(crad['scorr_mean'], reg_atlas_index) #42)

r_atlas_nii = resample_img(
    img=atlas_nii,
    target_affine=nifti_masker.mask_img_.get_affine(),
    target_shape=nifti_masker.mask_img_.shape,
    interpolation='nearest'
)
r_atlas_nii.to_filename('debug_ratlas.nii.gz')
atlas_labels = nifti_masker.transform(r_atlas_nii)[0, :]
atlas_labels += 1

# impose continuity and 1-based indexing in label numbers (for SPAMs)
atlas2 = np.zeros_like(atlas_labels)
for i_label, label in enumerate(np.unique(atlas_labels)):
    atlas2[np.where(atlas_labels == label)[0]] = i_label + 1  # one-based indexing
atlas_labels = atlas2

n_regions = len(np.unique(atlas_labels))

print('Current number of regions in atlas: %i' % n_regions)

# load HCP task data
print('Loading data...')
if on_server:
    X_task, labels = joblib.load('/storage/workspace/danilo/prni2015/preload_HT_3mm')
else:
    X_task, labels = joblib.load('/git/prni2015/preload_HT_3mm')

labels = np.array(labels)
if FORCE_TWO_CLASSES:
    inds1 = labels == 0  # TOM
    inds2 = labels == 1
    # inds1 = labels == 2  # TOM
    # inds2 = labels == 3
    # inds1 = labels == 4  # object grasp/orientation
    # inds2 = labels == 5
    inds = np.logical_or(inds1, inds2)

    X_task = X_task[inds]
    Y = labels[inds].astype(np.float64)
    Y[Y==2] = -1
    Y[Y==3] = 1
else:
    Y = np.float64(labels)

print('Done (%.2f MB)!' % (X_task.nbytes / 1e6))

# prepare Smith2009 ICA components
from nilearn.image import resample_img
rsn_nii4d = nib.load(datasets.fetch_smith_2009()['rsn20'])

my_rsns = index_img(
    rsn_nii4d,
    [5, 15, 9, 2, 1, 13, 10, 11, 12, 14, 7, 8])
my_rsns_data = nifti_masker.transform(my_rsns)

class StructuredEstimator(BaseEstimator):
    def __init__(self, regul, lambda1, reg_data=None, net_data=None,
                 group_labels=None,
                 max_it=1000, n_threads=1, reg_net_ratio=1, verbose=True):
        """
        Wrapper to enable access to SPAMS/Python-interface by
        an sklearn-typical estimator class.
        
        Parameters
        ----------
        lambda1 : float
            Set importance of penalty term
            
        regul : string
            Choice of regularization

        Valid values for the regularization parameter (regul) are:
          "l1", "l2",
          "sparse-group-lasso-l2", "sparse-group-lasso-linf",
          "tree-l0", "tree-l2", "trace-norm"
        """
        self.lambda1 = lambda1
        self.regul = regul
        self.reg_data = reg_data
        self.net_data = net_data
        self.group_labels = group_labels
        self.verbose = verbose
        self.max_it = max_it
        self.n_threads = n_threads
        self.reg_net_ratio = reg_net_ratio
        
        self.net_reg_map = None

    def fit(self, X, y):
        if self.verbose:
            print(self)
            # print('Unique Y: ')
            # print(np.unique(y))
        
        # SPAMS expects -1/+1 labelling
        y[y == 0] = -1

        Y = np.asfortranarray(np.float64(y[:, np.newaxis]))
        W0 = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float64, order="FORTRAN")
        W0 = np.asfortranarray(np.float64(W0))

        if 'tree' in self.regul:
            X_task_tree = np.asfortranarray(np.zeros_like(X))
            if self.net_reg_map is None:
                # define tree structure for SPAMS
                reg_labels = np.unique(self.reg_data)[1:]
                # self.net_data = zscore(self.net_data, axis=1)  # no values from an RSN dominate
                combo_map = self.net_data.argmax(axis=0) + 1

                self.net_reg_map = np.zeros(self.net_data.shape)
                for reg_label in reg_labels:
                    reg_inds = np.where(self.reg_data == reg_label)[0]
                    rsn_assigns = np.argmax(self.net_data[:, reg_inds], axis=0)
                    bins = np.bincount(rsn_assigns)
                    i_assigned_rsn = np.argmax(bins)
                    print('Region %i has %i voxels -> assigned to RSN index %i' %
                          (reg_label, len(reg_inds), i_assigned_rsn))
                    print(bins)
                    
                    # write the current region to the assigned RSN
                    self.net_reg_map[i_assigned_rsn, reg_inds] = reg_label

                net_reg_map_summed = np.sum(self.net_reg_map, axis=0)
                assert len(np.unique(net_reg_map_summed)) == n_regions # each region has been assigned to a RSN?
                
                self.N_own_variables = []
                self.own_variables = []
                self.eta_g = np.array(np.ones(13 + n_regions - 1), dtype=np.float32)
                
                self.groups = np.asfortranarray(np.zeros((13 + n_regions - 1, 13 + n_regions - 1)), dtype=np.bool)

                # add net info
                self.N_own_variables += list(np.zeros((13), dtype=np.int32))  # for root group + net groups
                self.own_variables = [np.int32(0)]

                self.groups[1:13, 0] = True  # each of the 12 nets belongs to root group

                cur_ind = 0
                for i_net in range(self.net_data.shape[0]):
                    n_net_vox = np.count_nonzero(self.net_reg_map[i_net, :])
                    self.own_variables += [np.int32(cur_ind)]
                    cur_ind += n_net_vox  # move behind size of current net

                self.N_own_variables[0] = np.int32((net_reg_map_summed == 0).sum())  # zero entries in network label map belong to g1
                # all network sizes add up to the number of non-zero voxels in network label map?
                assert cur_ind == len(net_reg_map_summed) - (net_reg_map_summed == 0).sum()
                assert len(self.own_variables) == 13

                # add reg info
                cur_ind = 0
                i_gr = 13  # first group is root, then 12 net groups = 13 = [0..12]
                for i_net in range(self.net_data.shape[0]):
                    regs_in_label = np.unique(self.net_reg_map[i_net, :])[1:]
                    for reg_label in regs_in_label:
                        reg_inds = np.where(self.net_reg_map[i_net, :] == reg_label)[0]
                        n_reg_vox = len(reg_inds)
                        self.own_variables += [np.int32(cur_ind)]
                        self.N_own_variables += [n_reg_vox]  # no region voxels have decendences
                        
                        self.groups[i_gr, i_net + 1] = True  # cur reg belongs to cur net
                        i_gr += 1
                        
                        X_task_tree[:, cur_ind:(cur_ind + n_reg_vox)] = X[:, reg_inds]

                        cur_ind += n_reg_vox  # move behind size of current net

                # all region sizes add up to the number of non-zero voxels in network label map?
                assert cur_ind == len(net_reg_map_summed) - (net_reg_map_summed == 0).sum()
                assert self.groups.sum() - 12 == n_regions - 1 # one dependence per region
                assert i_gr == 13 + n_regions - 1
                assert len(self.N_own_variables) == 13 + n_regions - 1
            
                self.own_variables =  np.array(self.own_variables, dtype=np.int32)
                self.N_own_variables =  np.array(self.N_own_variables,dtype=np.int32)
                self.groups = np.asfortranarray(self.groups)
                self.groups = ssp.csc_matrix(self.groups, dtype=np.bool)
                
                nifti_masker.inverse_transform(self.net_reg_map).to_filename(
                    'dbg_rsn_12_assigned_craddock_nostd.nii.gz')
            
            cur_ind = 0
            for i_net in range(self.net_data.shape[0]):
                regs_in_label = np.unique(self.net_reg_map[i_net, :])[1:]
                for reg_label in regs_in_label:
                    reg_inds = np.where(self.net_reg_map[i_net, :] == reg_label)[0]
                    n_reg_vox = len(reg_inds)
                    
                    X_task_tree[:, cur_ind:(cur_ind + n_reg_vox)] = X[:, reg_inds]

                    cur_ind += n_reg_vox  # move behind size of current net
            X = X_task_tree
            
            # group weighting (cf. Yuan 2007)
            self.eta_g[0] = 0
            for i_n, net_vect in enumerate(self.net_reg_map):  # net weights
                p_j = len(np.where(net_vect != 0)[0])
                weight = 1. / np.sqrt(p_j)
                self.eta_g[i_n + 1] = weight * (1 / self.reg_net_ratio)
                print('Weighting net group %i (n_vox=%i) to %.4f' % (i_n + 1, p_j, weight))
            for i_r, n_reg_vox in enumerate(self.N_own_variables[13:]):  # reg weights
                p_j = n_reg_vox
                weight = 1. / np.sqrt(p_j)
                self.eta_g[i_r + 12 + 1] = weight * self.reg_net_ratio

            # run SPAMS
            X = np.asfortranarray(X)
            param = {'numThreads' : self.n_threads,
                     'verbose' : self.verbose,
                     'lambda1' : float(self.lambda1),
                     'it0' : 1, 'max_it' : self.max_it,
                     'L0' : 0.1, 'tol' : 1e-03,
                     'intercept' : False,
                     'pos' : False}
            tree = {
                'eta_g': np.float64(self.eta_g),
                'groups': self.groups,
                'own_variables': self.own_variables,
                'N_own_variables': self.N_own_variables}
            param['compute_gram'] = True
            param['regul'] = self.regul
            param['loss'] = 'logistic'

            print(param)
            (W, optim_info) = spams.fistaTree(
                np.float64(Y), np.float64(X), # U: double m x n matrix   (input signals) m is the signal size
                W0, tree, True,
                **param)

            # bring weight vector back in original order
            cur_ind = 0
            W_org = np.zeros_like(W)
            for i_net in range(self.net_data.shape[0]):
                regs_in_label = np.unique(self.net_reg_map[i_net, :])[1:]
                for reg_label in regs_in_label:
                    reg_inds = np.where(self.net_reg_map[i_net, :] == reg_label)[0]
                    n_reg_vox = len(reg_inds)
                    
                    W_org[reg_inds, 0] = W[cur_ind:(cur_ind + n_reg_vox), 0]

                    cur_ind += n_reg_vox  # move behind size of current net
            self.W_ = W_org
        # end if 'tree'
        else:
            X = np.asfortranarray(X)

            param = {'L0': 0.1,
             'a': 0.1,
             'b': 1000,
             'compute_gram': False,
             'intercept': False,
             'ista': False,
             'it0': 1,
             'lambda1': float(self.lambda1),
             'loss': 'logistic',
             'max_it': self.max_it,
             'numThreads': self.n_threads,
             'pos': False,
             'regul': self.regul,
             'subgrad': False,
             'tol': 1e-03,
             'verbose': self.verbose}
            if self.group_labels is not None:
                param['groups'] = np.int32(self.group_labels)
            print(param)
            (self.W_, optim_info) = spams.fistaFlat(Y, X, W0, True, **param)            
        
        self.optim_info_ = optim_info

    def decision_function(self, X):
        return np.array(np.dot(X, self.W_), dtype=np.float32)[:, 0]

    def predict(self, X_test):
        y_pred = np.array(np.dot(X_test, self.W_) > 0,
                          dtype=np.float32)[:, 0]
        y_pred[y_pred == 0] = -1
        return y_pred

    def score(self, X, y, return_prfs=False):
        pred_y = np.float32(self.predict(X))
        acc = np.mean(pred_y == y)
        prfs = precision_recall_fscore_support(pred_y, y)
        if return_prfs:
            return acc, prfs
        else:
            return acc

# subsample input data
from sklearn.cross_validation import StratifiedShuffleSplit
if MY_DATA_RATIO != 100:
    ratio = MY_DATA_RATIO / 100.
    print('DATA SUBSELECTION at %.2f!!!' % ratio)
    folder = StratifiedShuffleSplit(Y, n_iter=10, train_size=ratio,
                                    random_state=42)
    inds_train, _ = iter(folder).next()
    X_task = X_task[inds_train]
    Y = Y[inds_train]

# run SPAMs
for REG_PEN in REGS:
    for REGNET_RATIO in REGNET_RATIO_GRID:
        folder = StratifiedShuffleSplit(Y, n_iter=10, test_size=0.1,
                                        random_state=42)
        inds_train, inds_test = iter(folder).next()
        X_train = X_task[inds_train]
        # all estimators will have the same training set due to random_state
        Y_train = Y[inds_train]
        X_test = X_task[inds_test]
        Y_test = Y[inds_test]

        if (REG_PEN == 'l1') or (REG_PEN == 'l2') or (REG_PEN == 'trace-norm'):
            cur_atlas_labels = None
            cur_net_data = None
            cur_group_labels = None
        elif 'tree' in REG_PEN:
            cur_atlas_labels = atlas_labels
            cur_net_data = my_rsns_data
            cur_group_labels = None
        elif 'sparse-group-lasso' in REG_PEN:
            cur_atlas_labels = None
            cur_net_data = None
            cur_group_labels = np.int32(atlas_labels)
        else:
            raise Exception('Unknown penalty term!')

        clf = StructuredEstimator(
            lambda1=0.25,
            regul=REG_PEN,
            reg_data=cur_atlas_labels,
            net_data=cur_net_data,
            max_it=MY_MAX_IT,
            n_threads=1,
            group_labels=cur_group_labels,
            reg_net_ratio=REGNET_RATIO
        )

        from sklearn.grid_search import GridSearchCV
        from sklearn.multiclass import OneVsRestClassifier

        param_grid = {'estimator__lambda1': LAMBDA_GRID}

        # start time
        start_time = time.time()

        clf_ovr = OneVsRestClassifier(clf, n_jobs=1)
        clf_ovr_gs = GridSearchCV(clf_ovr, param_grid, n_jobs=N_JOBS, cv=3)
        clf_ovr_gs.fit(X_train, Y_train)

        train_acc = clf_ovr_gs.score(X_train, Y_train)
        print('Train-Accuracy: %.2f' % train_acc)

        test_acc = clf_ovr_gs.score(X_test, Y_test)
        print('Test-Accuracy: %.2f' % test_acc)
        y_pred = clf_ovr_gs.predict(X_test)
        test_prfs = precision_recall_fscore_support(y_pred, Y_test)

        clf_ovr_gs.train_acc = train_acc
        clf_ovr_gs.test_acc = test_acc
        clf_ovr_gs.test_prfs = test_prfs

        # stop time
        dur = time.time() - start_time
        total_mins = dur / 60
        hs, mins = divmod(total_mins, 60)
        print('-' * 80)
        print("Elapsed time: %i hours and %i minutes" % (hs, mins))

        out_fname = '%s_dataratio%i_maxit%i' % (clf.regul, MY_DATA_RATIO, clf.max_it)
        out_path = op.join(WRITE_DIR, out_fname)
        joblib.dump(clf_ovr_gs, out_path, compress=9)


STOP

# test_acc = gs.best_estimator_.score(X_test, Y_test)
# print('Accuracy: %.2f' % test_acc)
# 
# 
# out_fname = 'test_TOMaudio_vs_video_%s_lambda%.3f.nii.gz' % (
#     clf.regul, clf.lambda1
# )
# nifti_masker.inverse_transform(clf.W_.T).to_filename(out_fname)

contrasts_names = [
'REWARD-PUNISH', 'PUNISH-REWARD', 'SHAPES-FACES', 'FACES-SHAPES',
'RANDOM-TOM', 'TOM-RANDOM',

'MATH-STORY', 'STORY-MATH',
'T-AVG', 'F-H', 'H-F',
'MATCH-REL', 'REL-MATCH',

'BODY-AVG', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG',
'2BK-0BK'
]

def dump_comps(masker, compressor, components, threshold=2, fwhm=None,
               perc=None):
    from scipy.stats import zscore
    from nilearn.plotting import plot_stat_map
    from nilearn.image import smooth_img
    from scipy.stats import scoreatpercentile

    n_comp = len(components)
    if isinstance(compressor, basestring):
        comp_name = compressor
    else:
        comp_name = compressor.__str__().split('(')[0]

    for i_c, comp in enumerate(components):
        path_mask = op.join(WRITE_DIR, '%s_%i-%i' % (comp_name,
                                                     n_comp, i_c + 1))
        nii_raw = masker.inverse_transform(comp)
        nii_raw.to_filename(path_mask + '.nii.gz')
        
        comp_z = zscore(comp)
        
        if perc is not None:
            cur_thresh = scoreatpercentile(np.abs(comp_z), per=perc)
            path_mask += '_perc%.2f' % perc
            print('Applying percentile %.2f (threshold: %.2f)' % (perc, cur_thresh))
        else:
            cur_thresh = threshold
            path_mask += '_thr%.2f' % cur_thresh
            print('Applying threshold: %.2f' % cur_thresh)
            
        path_mask += '_%s' % contrasts_names[i_c]

        nii_z = masker.inverse_transform(comp_z)
        gz_path = path_mask + '_zmap.nii.gz'
        nii_z.to_filename(gz_path)
        plot_stat_map(gz_path, bg_img='colin.nii', threshold=cur_thresh,
                      cut_coords=(0, -2, 0), draw_cross=False,
                      output_file=path_mask + 'zmap.png')
                      
        # optional: do smoothing
        if fwhm is not None:
            nii_z_fwhm = smooth_img(nii_z, fwhm=fwhm)
            plot_stat_map(nii_z_fwhm, bg_img='colin.nii', threshold=cur_thresh,
                          cut_coords=(0, -2, 0), draw_cross=False,
                          output_file=path_mask +
                          ('zmap_%imm.png' % fwhm))

# n_est = len(clf_ovr_gs.best_estimator_.estimators_)
# coef_per_class = [est.W_ for est in clf_ovr_gs.best_estimator_.estimators_]
# coef_per_class = np.squeeze(coef_per_class)

# dump_comps(nifti_masker, 'trace_dataratio%i_maxit100', coef_per_class, threshold=0.0)


# rsync -vza dbzdok@drago:/storage/workspace/danilo/srne/srne_benchmark_rnratio1_zerobrainlevel_weighted/* /git/srne/srne_benchmark_rnratio1_zerobrainlevel_weighted/

import matplotlib
# matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import re
%matplotlib qt

plt.close('all')

LOG_LBD = False
for reg in ['l1', 'l2']:
    anal_str = '%s_dataratio%i_maxit%i' % (reg, MY_DATA_RATIO, MY_MAX_IT)
    tar_dump_file = '%s/%s' % (WRITE_DIR, anal_str)
    if not op.exists(tar_dump_file):
        print('SKIPPED: %s' % tar_dump_file)
        # continue
    clf_ovr_gs = joblib.load(tar_dump_file)
    plt.close('all')
    plt.figure()
    means = []
    stds = []
    lbds = []
    for grid_str in clf_ovr_gs.grid_scores_:
        mean, std, lbd = re.findall("\d+.\d+", str(grid_str))
        mean, std, lbd = np.float(mean), np.float(std), np.float(lbd)
        means.append(mean)
        stds.append(std)
        lbds.append(lbd)

    if LOG_LBD:
        plt.errorbar(np.log10(lbds), y=means, yerr=stds, color='r', linewidth=2)
    else:
        plt.errorbar(lbds, y=means, yerr=stds, color='r', linewidth=2)
    if LOG_LBD:
        plt.xlabel('log_10($\lambda$)')
    else:
        plt.xlabel('$\lambda$')
    plt.ylabel('accuracy (mean)')
    plt.ylim(0.0, 1.0)
    if LOG_LBD:
        plt.xticks(np.log10(lbds))
        x_text = np.mean(np.log10(lbds))
    else:
        plt.xticks(lbds)
        x_text = np.mean(lbds)
    plt.text(x_text, 0.95, 'Final train-set acc: %.2f%%' % (clf_ovr_gs.train_acc * 100),
             fontsize=18)
    plt.text(x_text, 0.90, 'Final test-set acc: %.2f%%' % (clf_ovr_gs.test_acc * 100),
             fontsize=18)
    plt.title('GridSearch: ' + anal_str)
    plt.savefig(tar_dump_file + '_gs.png')

    # PRFS
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(contrasts_names)), clf_ovr_gs.test_prfs[0], label='precision')
    plt.plot(range(len(contrasts_names)), clf_ovr_gs.test_prfs[1], label='recall')
    plt.xticks(range(len(contrasts_names)), contrasts_names, rotation=90)
    plt.ylabel('accuracy')
    plt.title('Class-wise model performance', {'fontsize': 16})
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig(tar_dump_file + '_precrec.png')

for reg in [REGS[0]]:
    anal_str = '%s_dataratio%i_maxit%i' % (reg, MY_DATA_RATIO, MY_MAX_IT)
    tar_dump_file = '%s/%s' % (WRITE_DIR, anal_str)
    if not op.exists(tar_dump_file):
        print('SKIPPED: %s' % tar_dump_file)
        continue
    clf_ovr_gs = joblib.load(tar_dump_file)
    # weights = clf_ovr_gs.best_estimator_.estimators_[0].W_.T

    n_est = len(clf_ovr_gs.best_estimator_.estimators_)
    coef_per_class = [est.W_ for est in clf_ovr_gs.best_estimator_.estimators_]
    coef_per_class = np.squeeze(coef_per_class)
    dump_comps(
        nifti_masker,
        anal_str + '_weights',
        coef_per_class,
        threshold=0.0)

from nilearn import plotting
from scipy.stats import zscore
for i_cont, cont_name in enumerate(contrasts_names):
    out_fname = 'plots/tree-l2_weights_%s' % cont_name
    coef = coef_per_class[i_cont, :]
    weight_nii = nifti_masker.inverse_transform(
        coef)
    plotting.plot_stat_map(weight_nii, cut_coords=(0, 0, 0),
                           title='', bg_img='colin.nii',
                           colorbar=True, draw_cross=False,
                           black_bg=True)
    plt.savefig(out_fname + '_raw.png',
                dpi=200, transparent=True)
    
    coef_z = zscore(coef)
    weight_nii = nifti_masker.inverse_transform(
        coef_z)
    plotting.plot_stat_map(weight_nii, cut_coords=(0, 0, 0),
                           title='', bg_img='colin.nii',
                           colorbar=True, draw_cross=False,
                           black_bg=True)
    plt.savefig(out_fname + '_zmap.png',
                dpi=200, transparent=True)
    

