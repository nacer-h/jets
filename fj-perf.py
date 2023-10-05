def load_fastjet_fjcontrib():
    import os
    import cppyy
    fastjet_prefix = '../fastjet'
    cppyy.add_include_path(os.path.join(fastjet_prefix, 'include'))
    cppyy.add_library_path(os.path.join(fastjet_prefix, 'lib'))
    #
    cppyy.load_library('libfastjettools')
    cppyy.load_library('libfastjet')
    cppyy.load_library('libfastjetcontribfragile')
    #
    cppyy.include('fastjet/ClusterSequenceArea.hh')
    cppyy.include('fastjet/Selector.hh')
    # cppyy.include('fastjet/tools/GridMedianBackgroundEstimator.hh')
    cppyy.include('fastjet/tools/JetMedianBackgroundEstimator.hh')
    #
    cppyy.include('fastjet/contrib/IterativeConstituentSubtractor.hh')
    cppyy.include('fastjet/contrib/ConstituentSubtractor.hh')
    cppyy.include('fastjet/contrib/Nsubjettiness.hh')
    cppyy.include('fastjet/contrib/EnergyCorrelator.hh')
    cppyy.include('fastjet/contrib/SoftDrop.hh')
    #
    cppyy.add_include_path(os.path.join('../fjcontrib-1.048'))
    cppyy.include('ConstituentSubtractor/functions.hh')
    #
    global fastjet
    from cppyy.gbl import fastjet

def sigmoid(x, inverse = False):
    if inverse:
        return -(1. / x - 1.).log()
    step = x.sign()
    return .5 * (1. - step) + step / (1. + (-x.abs()).exp())

def load_unet(filename_sl, filename_ssl):
    from network.unet import UNet
    model_sl = UNet(in_channels = nchannel,
                    out_channels = nchannel)
    model_sl.to(device)
    if device.type != 'cuda':
        model_sl.load_state_dict(torch.load(filename_sl, map_location = torch.device(device)))
    else:
        model_sl.load_state_dict(torch.load(filename_sl))
    model_sl.eval()
    model_ssl = UNet(in_channels = nchannel + 1,
                     out_channels = nchannel)
    model_ssl.to(device)
    if device.type != 'cuda':
        model_ssl.load_state_dict(torch.load(filename_ssl, map_location = torch.device(device)))
    else:
        model_ssl.load_state_dict(torch.load(filename_ssl))
    model_ssl.eval()

    return model_sl, model_ssl

def load_unetplusplus(filename_sl, filename_ssl):
    from network.archs import NestedUNet
    model_sl = NestedUNet(input_channels = nchannel,
                          num_classes = nchannel)
    model_sl.to(device)
    model_sl.load_state_dict(torch.load(filename_sl))
    model_sl.eval()
    model_ssl = NestedUNet(input_channels = nchannel + 1,
                           num_classes = nchannel)
    model_ssl.to(device)
    model_ssl.load_state_dict(torch.load(filename_ssl))
    model_ssl.eval()

    return model_sl, model_ssl

def load_swin_unet(filename_sl, filename_ssl):
    from network.swin_transformer_unet_skip_expand_decoder_sys \
        import SwinTransformerSys

    model_sl = SwinTransformerSys(img_size = img_size,
                                  patch_size = 4,
                                  in_chans = nchannel,
                                  num_classes = nchannel,
                                  embed_dim = 96,
                                  depths = depths,
                                  num_heads = num_heads,
                                  window_size = window_size,
                                  mlp_ratio = 4.,
                                  qkv_bias = True,
                                  qk_scale = None,
                                  drop_rate = 0.,
                                  drop_path_rate = .1,
                                  ape = False,
                                  patch_norm = True,
                                  use_checkpoint = False)
    model_sl.to(device)
    model_sl.load_state_dict(torch.load(filename_sl))
    model_sl.eval()

    model_ssl = SwinTransformerSys(img_size = img_size,
                                   patch_size = 4,
                                   in_chans = nchannel + 1,
                                   num_classes = nchannel,
                                   embed_dim = 96,
                                   depths = depths,
                                   num_heads = num_heads,
                                   window_size = window_size,
                                   mlp_ratio = 4.,
                                   qkv_bias = True,
                                   qk_scale = None,
                                   drop_rate = 0.,
                                   drop_path_rate = .1,
                                   ape = False,
                                   patch_norm = True,
                                   use_checkpoint = False)
    model_ssl.to(device)
    model_ssl.load_state_dict(torch.load(filename_ssl))
    model_ssl.eval()

    return model_sl, model_ssl

import mpi

mpi_type, mpi_name, mpi_rank, mpi_size = mpi.init()

load_fastjet_fjcontrib()

import cppyy

cppyy.cppdef('''
void *image_to_pseudojet(void *ptr, int shape_0, int shape_1, int shape_2, int nazim_pad_unet)
{
    float *image = reinterpret_cast<float *>(ptr);
    const size_t nprap = shape_1;
    const size_t nazim = shape_2 - nazim_pad_unet;
    const float prap_max = nprap * M_PI / nazim;
    const float pixel_size = 2.0 * M_PI / nazim;
    std::vector<fastjet::PseudoJet> *pjet = new std::vector<fastjet::PseudoJet>();
    for (size_t index_perp = 0; index_perp < shape_0; index_perp++) 
    {
        for (size_t index_prap = 0; index_prap < nprap; index_prap++) 
        {
            for (size_t index_azim = 0; index_azim < nazim; index_azim++) 
            {
                const size_t flat_index = (index_perp * shape_1 + index_prap) * shape_2 + index_azim + (nazim_pad_unet >> 1);
                // The minimum pT in FastJet's
                // ClusterSequenceActiveAreaExplicitGhosts::
                // _post_process()
                if (image[flat_index] > 1e-100 / DBL_EPSILON) 
                {
                    const float perp = image[flat_index];
                    const float azim = (index_azim + 0.5) * pixel_size - M_PI;
                    const float prap = (index_prap + 0.5) * pixel_size - prap_max;
                    const float px = perp * std::cos(azim);
                    const float py = perp * std::sin(azim);
                    const float pz = perp * std::sinh(prap);
                    pjet->push_back(fastjet::PseudoJet(px, py, pz, std::sqrt(std::pow(perp, 2) + std::pow(pz, 2))));
                }
            }
        }
    }
    return pjet;
}
''')

def image_to_pseudojet(image):
    import numpy as np
    import sys
    import ctypes

    return cppyy.bind_object(cppyy.gbl.image_to_pseudojet(image.float().cpu().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)), image.shape[0], image.shape[1], image.shape[2], nazim_pad_unet), 'std::vector<fastjet::PseudoJet>')
# truth: ghost, event: real events, inject ghost to real event
cppyy.cppdef('''
void tag_event(std::vector<fastjet::PseudoJet> &event,
               std::vector<fastjet::PseudoJet> truth,
               fastjet::ClusterSequence cluster_sequence_truth,
               float scale_ghost = std::pow(2.0, -10))
{
    for (size_t i = 0; i < truth.size(); i++) 
    {
        for (auto &c : cluster_sequence_truth.constituents(truth[i])) 
        {
            event.push_back(c * scale_ghost);
            // The default user_index = -1, so using non-negative i
            // is safe.
            event.back().set_user_index(i);
        }
    }
}
''')

#### calculate what is z (pT fraction of true jet in the reco jet), and determine what is the index of the highest #participating jet.
#cppyy.cppdef('''
#std::vector<int> match_jet(std::vector<fastjet::PseudoJet> reco,
#                           fastjet::ClusterSequenceArea
#                           cluster_sequence_reco)
#{
#    std::vector<int> index_truth;
#    for (auto &r : reco)
#    {
#        std::unordered_map<int, float> truth_content;
#        for (auto &c : cluster_sequence_reco.constituents(r))
#        {
#            if (c.user_index() >= 0)
#            {
#                truth_content[c.user_index()] += c.perp();
#            }
#        }
#        index_truth.push_back(-1);
#        float truth_content_max = -FLT_MAX;
#        for (auto &t : truth_content)
#        {
#            if (t.second > truth_content_max)
#            {
#                index_truth.back() = t.first;
#                truth_content_max = t.second;
#            }
#        }
#    }
#    return index_truth;
#}
#''')

### calculate what is z (pT fraction of true jet in the reco jet), and determine what is the index of jet fraction > 50% .
cppyy.cppdef('''
std::vector<int> match_jet(std::vector<fastjet::PseudoJet> reco, fastjet::ClusterSequenceArea cluster_sequence_reco)
{
    std::ofstream out_file("output.txt");
    std::vector<int> index_truth;
    std::map<int, double> pt_truth_tot;

    for (auto &r : reco)
    {
        for (auto &c : cluster_sequence_reco.constituents(r))
        {  
            if (c.user_index() >= 0)
            {
                pt_truth_tot[c.user_index()] += c.perp();
            }
        }
    }

    for (auto &r : reco)
    {
        std::unordered_map<int, float> truth_content;

        for (auto &c : cluster_sequence_reco.constituents(r))
        {
            if (c.user_index() >= 0)
            {
                truth_content[c.user_index()] += c.perp();
            }
        }

        index_truth.push_back(-1);
        float truth_content_max = -FLT_MAX;

        for (auto &t : truth_content)
        {
            if (t.second > truth_content_max)
            {
                truth_content_max = t.second;
                if (t.second > 0.5 * pt_truth_tot[t.first]) { index_truth.back() = t.first; }
            }   
        }
    }

    // out_file << "reco.size() = " << reco.size() << " " << std::endl;
    // for (auto &index: index_truth)
    // out_file <<index << ", ";

    // out_file.close();

    return index_truth;
}
''')

cppyy.cppdef('''
fastjet::contrib::SoftDrop::StructureType
structure(fastjet::PseudoJet jet)
{
    return jet.structure_of<fastjet::contrib::SoftDrop>();
}
''')

cppyy.cppdef('''
double lha(fastjet::PseudoJet jet, double distance_0)
{
    static const double kappa = 1.0;
    static const double beta = 0.5;
    double s = 0;
    for (auto &c : jet.constituents()) 
    {
        s += pow(c.perp(), kappa) * pow(c.squared_distance(jet), 0.5 * beta);
    }
    return s / pow(jet.perp(), kappa) * pow(distance_0, beta);
}
''')


#####################################################################
def print_matched(reco, truth, index_truth, bge_rho, bge_rho_truth, algo, subtract_rho, **kwargs):
    import kinematics
    for i in range(len(reco)):
        if not (reco[i].perp() > 2. and index_truth[i] >= 0):
            continue
        # print("------- i, pt = \n", i, reco[i].perp())
        rho_a = 0
        if subtract_rho:
            # if reco[i].perp() - rho_a > 150.:
                # tqdm.write(f'{reco[i].perp()} {bge_rho.rho()} {bge_rho_truth} {reco[i].area()} {np.pi * antikt_d**2}')
            rho_a = bge_rho.rho() * reco[i].area()
        
        n_consti_r = 0
        for p in reco[i].constituents():
            n_consti_r = len([p for p in reco[i].constituents()]) # if p.pt() > 0.05
        
        n_consti_t = 0
        for p in truth[index_truth[i]].constituents():
            n_consti_t = len([p for p in truth[index_truth[i]].constituents()]) # if p.pt() > 0.05

        print(bge_rho.rho(),				                                    # 0
              algo,					                                            # 1
              reco[i].delta_R(truth[index_truth[i]]),	                        # 2
              reco[i].pseudorapidity(),			                                # 3
              truth[index_truth[i]].pseudorapidity(),	                        # 4
              reco[i].m(),				                                        # 5
              truth[index_truth[i]].m(),		                                # 6
              reco[i].perp() - rho_a,			                                # 7
              truth[index_truth[i]].perp(),		                                # 8
              width(reco[i]),				                                    # 9
              width(truth[index_truth[i]]),		                                # 10
              nsub1_beta1(reco[i]),			                                    # 11
              nsub1_beta1(truth[index_truth[i]]),	                            # 12
              nsub2_beta1(reco[i]),			                                    # 13
              nsub2_beta1(truth[index_truth[i]]),	                            # 14
              nsub3_beta1(reco[i]),			                                    # 15
              nsub3_beta1(truth[index_truth[i]]),	                            # 16
              ecf_2_4(reco[i]),				                                    # 17
              ecf_2_4(truth[index_truth[i]]),	                            	# 18
              ecf_3_4(reco[i]),				                                    # 19
              ecf_3_4(truth[index_truth[i]]),		                            # 20
              cppyy.gbl.lha(reco[i], antikt_d),	                            	# 21
              cppyy.gbl.lha(truth[index_truth[i]], antikt_d),	        		# 22
              cppyy.gbl.structure(sd_2_01(reco[i])).delta_R(),		        	# 23
              cppyy.gbl.structure(sd_2_01(truth[index_truth[i]])).delta_R(),	# 24
              cppyy.gbl.structure(sd_2_01(reco[i])).symmetry(),		        	# 25
              cppyy.gbl.structure(sd_2_01(truth[index_truth[i]])).symmetry(),	# 26
              reco[i].rap(),            	                                    # 27
              truth[index_truth[i]].rap(),              	                    # 28
              reco[i].phi(),            	                                    # 29
              truth[index_truth[i]].phi(),	                                    # 30
              n_consti_r,                                                       # 31
              n_consti_t,	                                                    # 32
              len(reco),	                                                    # 33
              len(truth),                                                       # 34
            #   *[c.perp() for c in reco[i].constituents()],                     
              **kwargs)

import sys
import h5py

if len(sys.argv) < 3:
    sys.exit(0)

f = h5py.File(sys.argv[1], 'r')
# print(f['X'].shape)

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# perp = torch.from_numpy(f['X'][1, :, :, :f['X'].shape[-1] - nazim_pad_unet]).float() # .to(device)

# print(perp.shape)

import numpy as np

nchannel = 12
# model_sl = UNet(in_channels = nchannel, out_channels = nchannel)
img_size = (7 * 32, 7 * 64)
num_heads = [3, 6, 12, 24]
depths = [2, 2, 4, 2]
window_size = 7

nazim_pad_unet = 96

prap_max = img_size[0] * np.pi / (img_size[1] - nazim_pad_unet);

# print(prap_max)

alice_tpc = True

# Workaround for non-functioning operator ! on fastjet::Selector in
# Cppyy
cppyy.cppdef('''
fastjet::Selector fastjet_sw_not(fastjet::Selector s)
{
    return !s;
}
''')

if alice_tpc:
    prap_max_detector = .9
else:
    prap_max_detector = prap_max

sel_bge = fastjet.SelectorPtMin(1e-50) * \
    fastjet.SelectorAbsRapMax(prap_max_detector - .5) * \
    cppyy.gbl.fastjet_sw_not(fastjet.SelectorNHardest(2))
jet_def_bge = fastjet.JetDefinition(fastjet.kt_algorithm, .5)
area_def_bge = fastjet.AreaDefinition(fastjet.active_area_explicit_ghosts, fastjet.GhostedAreaSpec(prap_max_detector))
# bge_rho = fastjet.GridMedianBackgroundEstimator(.9, .5)
bge_rho = fastjet.JetMedianBackgroundEstimator(sel_bge, jet_def_bge, area_def_bge)

subtractor = fastjet.contrib.IterativeConstituentSubtractor()
subtractor.set_distance_type(fastjet.contrib.ConstituentSubtractor.deltaR)
sel_max_pt = fastjet.Selector(fastjet.SelectorPtMax(15))
subtractor.set_particle_selector(sel_max_pt)

### define Constituent Subtractor
constituentSubtractor = fastjet.contrib.ConstituentSubtractor()
constituentSubtractor.set_distance_type(fastjet.contrib.ConstituentSubtractor.deltaR)
constituentSubtractor.set_particle_selector(sel_max_pt)

from cppyy.gbl.std import vector

if sys.argv[-1].find('_1') != -1:
    # ICS paper appendix B
    antikt_d = 1.
    max_distances = vector['double']((.2, .35))
    alphas = vector['double']((1, 1))
    ghost_area = .0025
elif sys.argv[-1].find('_04r') != -1:
    # Rey CS adapted to ICS
    antikt_d = .4
    max_distances = vector['double']((.2, .25))
    alphas = vector['double']((1, 1))
    ghost_area = .01
elif sys.argv[-1].find('_04') != -1:
    # ICS paper appendix B
    antikt_d = .4
    max_distances = vector['double']((.2, .1))
    alphas = vector['double']((1, 1))
    ghost_area = .0025
elif sys.argv[-1].find('_02') != -1:
    # ICS paper appendix B
    antikt_d = .2
    max_distances = vector['double']((.2, .1))
    alphas = vector['double']((1, 1))
    ghost_area = .01

# antikt_d = .2

subtractor.set_parameters(max_distances, alphas)
subtractor.set_ghost_removal(True)
subtractor.set_ghost_area(ghost_area)
subtractor.set_max_eta(prap_max_detector)
subtractor.set_background_estimator(bge_rho)

subtractor.initialize()

### define Constituent Subtractor
# constituentSubtractor.set_parameters(max_distances, alphas)
constituentSubtractor.set_max_distance(0.2)
constituentSubtractor.set_alpha(1)
# constituentSubtractor.set_ghost_removal(True)
constituentSubtractor.set_ghost_area(ghost_area)
constituentSubtractor.set_max_eta(prap_max_detector)
constituentSubtractor.set_background_estimator(bge_rho)
constituentSubtractor.initialize()

jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, antikt_d)

from cppyy.gbl import JetWidth

width = JetWidth()

nsub1_beta1 = fastjet.contrib.Nsubjettiness(1, fastjet.contrib.OnePass_WTA_KT_Axes(), fastjet.contrib.UnnormalizedMeasure(1.))
nsub2_beta1 = fastjet.contrib.Nsubjettiness(2, fastjet.contrib.OnePass_WTA_KT_Axes(), fastjet.contrib.UnnormalizedMeasure(1.))
nsub3_beta1 = fastjet.contrib.Nsubjettiness(3, fastjet.contrib.OnePass_WTA_KT_Axes(), fastjet.contrib.UnnormalizedMeasure(1.))

ecf_2_4 = fastjet.contrib.EnergyCorrelator(2, 4.)
ecf_3_4 = fastjet.contrib.EnergyCorrelator(3, 4.)

sd_2_01 = fastjet.contrib.SoftDrop(2., .1)

model_sl, model_ssl = load_unet('save/sl_model.pth', 'save/ssl_model.pth')

from sparse import coo
from num import func

flat_ue = True

# def print_jet(jets):
#     for j in jets[-3:]:
#         print(j.perp(), j.pseudorapidity(), j.phi_std(), nsub3_beta1(j))
#     print()

# area_def = fastjet.VoronoiAreaSpec()
area_def = fastjet.AreaDefinition(fastjet.active_area, fastjet.GhostedAreaSpec(prap_max_detector))

azim = None

if mpi_rank is None:
    f_out = open(sys.argv[-1], 'w')
else:
    f_out = open(f'{sys.argv[-1]}_{mpi_rank:05}', 'w')

### write jet constituents pT into a file
file_consti_cs = open('file_consti_cs.txt', 'w')
file_consti_ics = open('file_consti_ics.txt', 'w')
file_consti_sup = open('file_consti_sup.txt', 'w')

from tqdm.auto import tqdm, trange

nevent = f['event_hard_image_indices'][0,-1]
if mpi_rank is None or mpi_size is None:
    index_event_begin = 0
    index_event_end = nevent
else:
    index_event_begin = mpi_rank * nevent // mpi_size
    index_event_end = (mpi_rank + 1) * nevent // mpi_size

for index_event in trange(index_event_begin, index_event_end):
    # if index_event > 10: #loop over only few events
    #     continue
    # load aonly one even at time, done this way because dealing with sparse tensors
    hard = coo.h5_partial_load(
        (f['event_hard_image_indices'],
         f['event_hard_image_values'],
         (-1, nchannel, img_size[0], img_size[1])),
        (index_event,
         index_event + 1))

    hard_event = image_to_pseudojet(hard.to_dense().to(device)[0])

    clust_seq_hard = fastjet.ClusterSequence(hard_event, jet_def)
    hard_jet = clust_seq_hard.inclusive_jets()

    if len(hard_jet) > 0:
        hardest_jet = fastjet.sorted_by_pt(hard_jet)[0]
    else:
        hardest_jet = None
        continue
    if hardest_jet.pt() < 20:
        continue
    ### loop over hard_jet consti --> overwrite "hard_jet"
    ### loop over hard_jet constituents
    ### compute pT, Eta, Phi for each constituent
    ### convert to numpy.array
    ### convert to pytorch (tensor: image)

    # if index_event % 10 == 0: print(index_event, file = sys.stderr)

    if not max([0] + [j.perp() for j in hard_jet]) >= 20:
        continue
    # if not max([0] + [j.perp() for j in hard_jet]) <= 70:
        # continue

    ue = coo.h5_partial_load(
        (f['event_ue_image_indices'],
         f['event_ue_image_values'],
         (-1, nchannel, img_size[0], img_size[1])),
        (index_event,
         index_event + 1))
    bge_rho_truth = torch.sum(ue.coalesce().values()).item() / (4. * np.pi * prap_max_detector)

    if hard is None or ue is None:
        continue
    full_image = (hard + ue).to_dense().to(device)
    # full_image = ue.to_dense().to(device)

    full_event = image_to_pseudojet(full_image[0])
    # ue_event = image_to_pseudojet(torch.from_numpy(f['Y'][0]))

    # clust_seq_full = fastjet.ClusterSequence(full_event, jet_def)
    # full_jet = clust_seq_full.inclusive_jets()
    # print_jet(full_jet)

    if True:
        bge_rho.set_particles(full_event)
        # ue_event = image_to_pseudojet(ue.to_dense().to(device)[0])
        # tqdm.write(f'{ue_event.size()} {full_event.size()}')
        # bge_rho.set_particles(ue_event)
    else:
        clust_seq_bge = fastjet.ClusterSequenceArea(full_event, jet_def_bge, area_def_bge)
        bge_jet = clust_seq_bge.inclusive_jets()
        bge_rho.set_jets(bge_jet)
        # if bge_rho.rho() < 1e-10: tqdm.write(f'{bge_rho.rho()} {bge_rho_truth} {bge_rho.rho() / bge_rho_truth} {bge_rho.n_jets_used()} {bge_rho.mean_area()}')
        # for i in range(bge_jet.size()): tqdm.write(f'{i} {bge_jet[i].perp()} {bge_jet[i].pseudorapidity()} {bge_jet[i].phi_std()}')

    hard_ics_event = subtractor.subtract_event(full_event)
    cppyy.gbl.tag_event(hard_ics_event, hard_jet, clust_seq_hard)
    
    ### Constituent Subtraction method
    hard_cs_event = constituentSubtractor.subtract_event(full_event)
    cppyy.gbl.tag_event(hard_cs_event, hard_jet, clust_seq_hard)

    cppyy.gbl.tag_event(full_event, hard_jet, clust_seq_hard)

    clust_seq_rho_a = fastjet.ClusterSequenceArea(full_event, jet_def, area_def)
    hard_rho_a_jet = clust_seq_rho_a.inclusive_jets()
    hard_rho_a_jet_index_hard = cppyy.gbl.match_jet(hard_rho_a_jet, clust_seq_rho_a)
    triggered = False
    for i in range(hard_rho_a_jet.size()):
        if hard_rho_a_jet[i].perp() - bge_rho.rho() * hard_rho_a_jet[i].area() < 50.:
            triggered = True
            break
    if False:
        import matplotlib.pyplot as plt
        plt.clf()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.matshow(torch.max(.1 * torch.ones(1, 1), torch.sum(hard.to_dense()[0], axis = 0)).log10().cpu().numpy())
        ax2.matshow(torch.max(.1 * torch.ones(1, 1), torch.sum(ue.to_dense()[0], axis = 0)).log().cpu().numpy())
        fig.savefig('trigger.png')
        plt.close()

    print_matched(hard_rho_a_jet, hard_jet, hard_rho_a_jet_index_hard, bge_rho, bge_rho_truth, 9, True, file = f_out, flush = True)

    clust_seq_corr = fastjet.ClusterSequenceArea(hard_ics_event, jet_def, area_def)
    hard_ics_jet = clust_seq_corr.inclusive_jets()
    hard_ics_jet_index_hard = cppyy.gbl.match_jet(hard_ics_jet, clust_seq_corr)

    ### Constituent Subtraction method
    clust_seq_corr_cs = fastjet.ClusterSequenceArea(hard_cs_event, jet_def, area_def)
    hard_cs_jet = clust_seq_corr_cs.inclusive_jets()
    hard_cs_jet_index_hard = cppyy.gbl.match_jet(hard_cs_jet, clust_seq_corr_cs)

    # print_jet(hard_ics_jet)

    print_matched(hard_ics_jet, hard_jet, hard_ics_jet_index_hard, bge_rho, bge_rho_truth, 0, False, file = f_out, flush = True)

    ### print jet constituents pT into a file
    for i, jet in enumerate(hard_ics_jet):
        if not (jet.perp() > 2. and hard_ics_jet_index_hard[i] >= 0):
            continue
        print(*[c.perp() for c in jet.constituents()], file = file_consti_ics, flush = True) # if c.perp() > 0.05

    ### Constituent Subtraction method
    print_matched(hard_cs_jet, hard_jet, hard_cs_jet_index_hard, bge_rho, bge_rho_truth, 3, False, file = f_out, flush = True)

    ### print jet constituents pT into a file
    for i, jet in enumerate(hard_cs_jet):
        if not (jet.perp() > 2. and hard_cs_jet_index_hard[i] >= 0):
            continue
        print(*[c.perp() for c in jet.constituents()], file = file_consti_cs, flush = True) # if c.perp() > 0.05

    pred = model_sl(full_image)
    hard_est = func.sigmoid(pred) * full_image
    # hard_est = full_image
    ue_est = full_image - hard_est

    # assert(full_image.shape[-1] == 320)
    # assert(ue.shape[-1] == 320)
    # hard_superv_im = 
    # assert(hard_superv_im.shape[-1] == 320)
    hard_superv_event = image_to_pseudojet(hard_est.detach()[0])
    cppyy.gbl.tag_event(hard_superv_event, hard_jet, clust_seq_hard)
    clust_seq_hard_superv = fastjet.ClusterSequenceArea(hard_superv_event, jet_def, area_def)
    hard_superv_jet = clust_seq_hard_superv.inclusive_jets()
    hard_superv_jet_index_hard = cppyy.gbl.match_jet(hard_superv_jet, clust_seq_hard_superv)
    
    # tqdm.write(f'{torch.sum(full_image).item():7.02f}, {torch.sum(hard.to_dense()).item():7.02f}, {torch.sum(hard_est).item():7.02f}')
    # tqdm.write(f'{sum([p.perp() for p in full_event]):7.02f}, {sum([p.perp() for p in hard_event]):7.02f}, {sum([p.perp() for p in hard_superv_event]):7.02f}, {sum([p.perp() for p in hard_ics_event]):7.02f}')

    print_matched(hard_superv_jet, hard_jet, hard_superv_jet_index_hard, bge_rho, bge_rho_truth, 1, False, file = f_out, flush = True)

    ### print jet constituents pT into a file
    for i, jet in enumerate(hard_superv_jet):
        if not (jet.perp() > 2. and hard_superv_jet_index_hard[i] >= 0):
            continue
        print(*[c.perp() for c in jet.constituents()], file = file_consti_sup, flush = True) # if c.perp() > 0.05

    hard_superv_event_2 = image_to_pseudojet(hard_est.detach()[0])
    cppyy.gbl.tag_event(hard_superv_event_2, hard_ics_jet, clust_seq_corr)
    clust_seq_hard_superv = fastjet.ClusterSequenceArea(hard_superv_event_2, jet_def, area_def)
    hard_superv_jet_2 = clust_seq_hard_superv.inclusive_jets()
    hard_superv_jet_2_index_hard = cppyy.gbl.match_jet(hard_superv_jet_2, clust_seq_hard_superv)

    print_matched(hard_superv_jet_2, hard_ics_jet, hard_superv_jet_2_index_hard, bge_rho, bge_rho_truth, 12, False, file = f_out, flush = True)

    # ### Constituent Subtraction method
    # tqdm.write(f'{sum([p.perp() for p in full_event]):7.02f}, {sum([p.perp() for p in hard_event]):7.02f}, {sum([p.perp() for p in hard_superv_event]):7.02f}, {sum([p.perp() for p in hard_cs_event]):7.02f}')
    # hard_superv_event_3 = image_to_pseudojet(hard_est.detach()[0])
    # cppyy.gbl.tag_event(hard_superv_event_3, hard_cs_jet, clust_seq_corr_cs)
    # clust_seq_hard_superv_2 = fastjet.ClusterSequenceArea(hard_superv_event_3, jet_def, area_def)
    # hard_superv_jet_3 = clust_seq_hard_superv_2.inclusive_jets()
    # hard_superv_jet_3_index_hard = cppyy.gbl.match_jet(hard_superv_jet_3, clust_seq_hard_superv_2)
    # print_matched(hard_superv_jet_3, hard_cs_jet, hard_superv_jet_3_index_hard, bge_rho, bge_rho_truth, 13, False, file = f_out, flush = True)

    ue_1 = torch.sum(ue_est, axis = 1, keepdims = True)
    ue_1 = torch.mean(ue_1, (-2, -1), keepdims = True).repeat(1, 1, ue_1.shape[-2], ue_1.shape[-1])
    full_ue = torch.cat((full_image, 1 * ue_1), axis = 1)
    hard_selfsup = func.sigmoid(model_ssl(full_ue)) * full_image

    hard_selfsup_im = hard_selfsup.detach()
    hard_selfsup_event = image_to_pseudojet(hard_selfsup_im[0])
    cppyy.gbl.tag_event(hard_selfsup_event, hard_jet, clust_seq_hard)
    clust_seq_hard_selfsup = fastjet.ClusterSequenceArea(hard_selfsup_event, jet_def, area_def)
    hard_selfsup_jet = clust_seq_hard_selfsup.inclusive_jets()
    hard_selfsup_jet_index_hard = cppyy.gbl.match_jet(hard_selfsup_jet, clust_seq_hard_selfsup)

    print_matched(hard_selfsup_jet, hard_jet, hard_selfsup_jet_index_hard, bge_rho, bge_rho_truth, 2, False, file = f_out, flush = True)

f.close()

# print_jet(hard_selfsup_jet)

# GridMedianBackgroundEstimator
# bge_rho(max_eta,0.5);
