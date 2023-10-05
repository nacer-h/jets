import sys

d_1 = len(sys.argv) > 1 and sys.argv[1].find('1') != -1

if d_1:
    f = open('compare_1.txt', 'r')
else:
    f = open('compare.txt', 'r')
data = f.read().split('\n')
f.close()

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import histplot
import pandas as pd
import argparse
import awkward1 as ak
from io import StringIO
import seaborn as sns
# sns.set()

np.seterr(divide='ignore')

index_start = 0
# count_match = 0
# for i in range(len(data)):
#     if data[i][:4] == '#---':
#         count_match += 1
#         if count_match == 2:
#             index_start = i
#             break
data_np = np.array(list(map(
    lambda s: list(map(float, s.split(' '))),
    data[index_start+1:-1])))

# print(data_np.shape)

perp_truth_range = (20., 200.)

### convert first file data from numpy array to dataFrame
col_names = ['bge_rho','algo','delta_R','eta_r','eta_t','m_r','m_t','pt_r','pt_t','width_r','width_t','nsub1_r','nsub1_t','nsub2_r','nsub2_t','nsub3_r','nsub3_t','ecf_2_4_r','ecf_2_4_t','ecf_3_4_r','ecf_3_4_t','lha_r','lha_t','sd_delta_R_r','sd_delta_R_t','sd_symmetry_r','sd_symmetry_t','y_r','y_t','phi_r','phi_t','nConsti_r','nConsti_t','njet_r','njet_t']
df = pd.DataFrame(data_np, columns = col_names)

## ---------------------- CS

col_names_extra = ['eta_t','pt_t', 'pt_r','njet_r','njet_t', 'nConsti_r','nConsti_t']
df_cs = df.loc[df['algo'] == 3, col_names_extra]
df_cs = df_cs.reset_index(drop=True)
df_cs_2 = pd.read_csv('file_consti_cs.txt', sep=' ', lineterminator='\n', names=list(map(str, range(int(df_cs['nConsti_r'].max()))))) ##, on_bad_lines='skip'
# df_cs_2 = df_cs_2[df_cs_2.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
df_cs_2 = df_cs_2.reset_index(drop=True)

df_cs_3 = df_cs.join(df_cs_2)
# print(df_cs_3.head(20))
df_cs_3 = df_cs_3.loc[(df_cs_3['pt_t'] >= perp_truth_range[0]) & (df_cs_3['pt_t'] <= perp_truth_range[1]) & (abs(df_cs_3['eta_t']) < .5)]
df_cs_3 = df_cs_3.reset_index(drop=True)
df_cs_3_1 = df_cs_3
df_cs_3 = df_cs_3.melt(col_names_extra, df_cs_3.columns[7:-1], value_name='pt_consti_r').dropna()
# df_melt=pd.melt(df_cs_3, id_vars = col_names_extra, value_vars=[‘MinDuration’,’MaxDuration’],var_name=’DurationType’,value_name=’Duration’)
# df_cs_3['pt_consti_r'] = np.round(df_cs_3['pt_consti_r'], decimals = 0)
df_cs_4 = df_cs_3.groupby('pt_consti_r').mean().reset_index()

plt.clf()
plt.tight_layout()
df_cs_4.plot(x='pt_consti_r', y='nConsti_r', marker='.', linestyle='none')
plt.title('CS')
plt.xlabel('$p_T^\\mathrm{constituents}$', ha = 'right', x = 1.)
plt.ylabel('$<N>^\\mathrm{constituents}$', ha = 'right', x = 1.)
plt.savefig('output_jetue/n_vs_pt_consti_cs.pdf')

## ---------------------- ICS

col_names_extra = ['eta_t','pt_t', 'pt_r','njet_r','njet_t', 'nConsti_r','nConsti_t']
df_ics = df.loc[df['algo'] == 0, col_names_extra]
df_ics = df_ics.reset_index(drop=True)
df_ics_2 = pd.read_csv('file_consti_ics.txt', sep=' ', lineterminator='\n', names=list(map(str, range(int(df_ics['nConsti_r'].max()))))) ##, on_bad_lines='skip'
# df_ics_2 = df_ics_2[df_ics_2.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
df_ics_2 = df_ics_2.reset_index(drop=True)

df_ics_3 = df_ics.join(df_ics_2)
# print(df_ics_3.head(20))
df_ics_3 = df_ics_3.loc[(df_ics_3['pt_t'] >= perp_truth_range[0]) & (df_ics_3['pt_t'] <= perp_truth_range[1]) & (abs(df_ics_3['eta_t']) < .5)]
df_ics_3 = df_ics_3.reset_index(drop=True)
df_ics_3_1 = df_ics_3
df_ics_3 = df_ics_3.melt(col_names_extra, df_ics_3.columns[7:-1], value_name='pt_consti_r').dropna()
# df_melt=pd.melt(df_ics_3, id_vars = col_names_extra, value_vars=[‘MinDuration’,’MaxDuration’],var_name=’DurationType’,value_name=’Duration’)
# df_ics_3['pt_consti_r'] = np.round(df_ics_3['pt_consti_r'], decimals = 0)
df_ics_4 = df_ics_3.groupby('pt_consti_r').mean().reset_index()

plt.clf()
plt.tight_layout()
df_ics_4.plot(x='pt_consti_r', y='nConsti_r', marker='.', linestyle='none')
plt.title('ICS')
plt.xlabel('$p_T^\\mathrm{constituents}$', ha = 'right', x = 1.)
plt.ylabel('$<N>^\\mathrm{constituents}$', ha = 'right', x = 1.)
plt.savefig('output_jetue/n_vs_pt_consti_ics.pdf')

## ---------------------- supervised

col_names_extra = ['eta_t','pt_t', 'pt_r','njet_r','njet_t', 'nConsti_r','nConsti_t']
df_sup = df.loc[df['algo'] == 1, col_names_extra]
df_sup = df_sup.reset_index(drop=True)
df_sup_2 = pd.read_csv('file_consti_sup.txt', sep=' ', lineterminator='\n', names=list(map(str, range(int(df_sup['nConsti_r'].max()))))) ##, on_bad_lines='skip'
# df_sup_2 = df_sup_2[df_sup_2.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
df_sup_2 = df_sup_2.reset_index(drop=True)

df_sup_3 = df_sup.join(df_sup_2)
# print(df_sup_3.head(20))
df_sup_3 = df_sup_3.loc[(df_sup_3['pt_t'] >= perp_truth_range[0]) & (df_sup_3['pt_t'] <= perp_truth_range[1]) & (abs(df_sup_3['eta_t']) < .5)]
df_sup_3 = df_sup_3.reset_index(drop=True)
df_sup_3_1 = df_sup_3
df_sup_3 = df_sup_3.melt(col_names_extra, df_sup_3.columns[7:-1], value_name='pt_consti_r').dropna()
# df_melt=pd.melt(df_sup_3, id_vars = col_names_extra, value_vars=[‘MinDuration’,’MaxDuration’],var_name=’DurationType’,value_name=’Duration’)
# df_sup_3['pt_consti_r'] = np.round(df_sup_3['pt_consti_r'], decimals = 0)
df_sup_4 = df_sup_3.groupby('pt_consti_r').mean().reset_index()

plt.clf()
plt.tight_layout()
df_sup_4.plot(x='pt_consti_r', y='nConsti_r', marker='.', linestyle='none')
plt.title('Supervised')
plt.xlabel('$p_T^\\mathrm{constituents}$', ha = 'right', x = 1.)
plt.ylabel('$<N>^\\mathrm{constituents}$', ha = 'right', x = 1.)
plt.savefig('output_jetue/n_vs_pt_consti_sup.pdf')

### plotting in one plot

c_pt_cs = (df_cs_3['pt_consti_r']).to_numpy()
c_pt_ics = (df_ics_3['pt_consti_r']).to_numpy()
c_pt_sup = (df_sup_3['pt_consti_r']).to_numpy()

plt.clf()
plt.hist(df_cs_3['pt_consti_r'], weights=np.ones_like(df_cs_3[df_cs_3.columns[0]]) / len(df_cs_3_1.index), bins=200, range=(0, 4), histtype=u'step', label='CS', color='red')
plt.hist(df_ics_3['pt_consti_r'], weights=np.ones_like(df_ics_3[df_ics_3.columns[0]]) / len(df_ics_3_1.index), bins=200, range=(0, 4), histtype=u'step', label='ICS', color='blue')
plt.hist(df_sup_3['pt_consti_r'], weights=np.ones_like(df_sup_3[df_sup_3.columns[0]]) / len(df_sup_3_1.index), bins=200, range=(0, 4), histtype=u'step', label = 'Supervised', color='green')
plt.legend()
plt.yscale("log")  
plt.xlabel('$p_T^\\mathrm{constituents}$', ha = 'right', x = 1.)
plt.ylabel('Counts Normalized by number of jets', ha = 'center', x = 1.)
plt.savefig('output_jetue/n_vs_pt_consti_all.pdf')

# ## --------------  use ROOT to plot
# import ROOT

# df_cs_3.to_csv('cs.csv', index=False)
# df_ics_3.to_csv('ics.csv', index=False)
# df_sup_3.to_csv('sup.csv', index=False)
# rdf_cs = ROOT.RDF.MakeCsvDataFrame('cs.csv')
# rdf_ics = ROOT.RDF.MakeCsvDataFrame('ics.csv')
# rdf_sup = ROOT.RDF.MakeCsvDataFrame('sup.csv')
# # rdf_cs.Snapshot('tree', 'cs.root')

# h_c_pt_cs = rdf_cs.Histo1D(("h_c_pt_cs", ";p_{T}^{constituents} (GeV/c);Counts normalized", 200, 0, 0.3), "pt_consti_r")
# h_c_pt_ics = rdf_ics.Histo1D(("h_c_pt_ics", ";p_{T}^{constituents} (GeV/c);Counts normalized", 200, 0, 0.3), "pt_consti_r")
# h_c_pt_sup = rdf_sup.Histo1D(("h_c_pt_sup", ";p_{T}^{constituents} (GeV/c);Counts normalized", 200, 0, 0.3), "pt_consti_r")
# # norm_cs = rdf_cs.Count().GetValue()
# # norm_ics = rdf_ics.Count().GetValue()
# # norm_sup = rdf_sup.Count().GetValue()
# norm_cs = len(df_cs_3_1.index)
# norm_ics = len(df_ics_3_1.index)
# norm_sup = len(df_sup_3_1.index)

# print(norm_cs)
# print(norm_ics)
# print(norm_sup)
# canv_c_pt = ROOT.TCanvas()
# # canv_c_pt.SetLogy()
# ROOT.gStyle.SetOptStat(0)
# h_c_pt_cs.SetLineColor(2)
# h_c_pt_ics.SetLineColor(4)
# h_c_pt_sup.SetLineColor(3)
# h_c_pt_cs.Scale(1./norm_cs)
# h_c_pt_ics.Scale(1./norm_ics)
# h_c_pt_sup.Scale(1./norm_sup)
# h_c_pt_sup.Draw('hist')
# h_c_pt_cs.Draw('hist same')
# h_c_pt_ics.Draw('hist same')
# # l_c_pt = ROOT.TLegend(0.67, 0.6, 0.78, 0.87)
# # l_c_pt.SetTextSize(0.05)
# # l_c_pt.SetBorderSize(0)
# # l_c_pt.AddEntry(h_c_pt_cs, 'CS', 'lpf')
# # l_c_pt.AddEntry(h_c_pt_ics, 'ICS', 'lpf')
# # l_c_pt.AddEntry(h_c_pt_sup, 'Supervised', 'lpf')
# # l_c_pt.Draw()
# canv_c_pt.SaveAs("output_jetue/h_c_pt.pdf")

# ## --------------

# c_pt_20_40 = df_cs_3.loc[(df_cs_3['pt_r'] >= 20) & (df_cs_3['pt_r'] <= 40), list(map(str, range(int(df_cs['nConsti_r'].max()))))].to_numpy()
# c_pt_20_40 = df_cs_3.loc[(df_cs_3['pt_r'] >= 20) & (df_cs_3['pt_r'] <= 40), ['pt_consti_r']].to_numpy()
# c_pt_40_60 = df_cs_3.loc[(df_cs_3['pt_r'] >= 40) & (df_cs_3['pt_r'] <= 60), ['pt_consti_r']].to_numpy()
# c_pt_60_80 = df_cs_3.loc[(df_cs_3['pt_r'] >= 60) & (df_cs_3['pt_r'] <= 80), ['pt_consti_r']].to_numpy()
# c_pt_80 = df_cs_3.loc[(df_cs_3['pt_r'] >= 80), ['pt_consti_r']].to_numpy()

# plt.clf()
# histplot.histogram(c_pt_20_40, bins = 30, range = (0., 0.1), label = '20 - 40 GeV/c')
# histplot.histogram(c_pt_40_60, bins = 30, range = (0., 0.1), label = '40 - 60 GeV/c')
# histplot.histogram(c_pt_60_80, bins = 30, range = (0., 0.1), label = '60 - 80 GeV/c')
# histplot.histogram(c_pt_80, bins = 30, range = (0., 0.1), label = '>= 80 GeV/c')
# plt.grid(True, linestyle = 'dashed')
# plt.xlabel('$p_T^\\mathrm{constituents}$', ha = 'right', x = 1.)
# plt.savefig('output_jetue/consti_pt.pdf')

algo_0 = np.logical_and(data_np[:,1] == 0, np.logical_and(data_np[:,8] >= perp_truth_range[0], np.logical_and(data_np[:,8] < perp_truth_range[1], np.abs(data_np[:,4]) < .5)))
algo_1 = np.logical_and(data_np[:,1] == 1, np.logical_and(data_np[:,8] >= perp_truth_range[0], np.logical_and(data_np[:,8] < perp_truth_range[1], np.abs(data_np[:,4]) < .5)))
algo_2 = np.logical_and(data_np[:,1] == 2, np.logical_and(data_np[:,8] >= perp_truth_range[0], np.logical_and(data_np[:,8] < perp_truth_range[1], np.abs(data_np[:,4]) < .5)))
algo_3 = np.logical_and(data_np[:,1] == 3, np.logical_and(data_np[:,8] >= perp_truth_range[0], np.logical_and(data_np[:,8] < perp_truth_range[1], np.abs(data_np[:,4]) < .5)))
algo_9 = np.logical_and(data_np[:,1] == 9, np.logical_and(data_np[:,8] >= perp_truth_range[0], np.logical_and(data_np[:,8] < perp_truth_range[1], np.abs(data_np[:,4]) < .5)))
algo_12 = np.logical_and(data_np[:,1] == 12, np.logical_and(data_np[:,8] >= perp_truth_range[0], np.logical_and(data_np[:,8] < perp_truth_range[1], np.logical_and(data_np[:,7] >= 2., np.abs(data_np[:,4]) < .5))))

# hist_0 = np.histogram(data_np[algo_0,-2] - data_np[algo_0,-1])
# hist_1 = np.histogram(data_np[algo_1,-2] - data_np[algo_1,-1])

histplot.root_like_rc()

if d_1:
    nbin = 30
else:
    nbin = 90

# nbin //= 6
nbin //= 3

# plot_range = (.8, 1.2)
plot_range = (0., 2.)

dpi = 200
figsize = (np.sqrt(32.), np.sqrt(32.))
# adjust_kw = {'top': 1. - 2.**-6, 'bottom': 1.375 * 2.**-3, 'left': 1.375 * 2**-3, 'right': 1. - 1.5 * 2.**-6}
adjust_kw = {'top': 1. - 2.**-6, 'bottom': 1.375 * 2.**-3, 'left': 1.375 * 2**-3, 'right': 1. - 1.5 * 2.**-6}
ylim_ratio = [0, 1.992]

plt.subplots_adjust(**adjust_kw)

if not d_1:
    plt.clf()
    histplot.histogram(data_np[algo_0,2], bins = nbin, range = (0, .05), label = 'ICS')
    histplot.histogram(data_np[algo_3,2], bins = nbin, range = (0, .05), label = 'CS')
    histplot.histogram(data_np[algo_1,2], bins = nbin, range = (0, .05), label = 'Supervised')
    # histplot.histogram(data_np[algo_2,2], bins = nbin, range = (0, .05), label = 'Self-supervised')
    plt.grid(True, linestyle = 'dashed')
    plt.xlabel('$\\Delta R$', ha = 'right', x = 1.)
    plt.legend()
    plt.savefig('output_jetue/dr.pdf')

variable = (
    ('eta', '$\\eta_\\mathrm{pred} / \\eta_\\mathrm{truth}$'),
    ('mass', '$m_\\mathrm{pred} / m_\\mathrm{truth}$'),
    ('perp', '$p_T^\\mathrm{pred} / p_T^\\mathrm{truth}$'),
    ('girth', '$g_\\mathrm{pred} / g_\\mathrm{truth}$'),
    ('nsub1', '$\\tau_{1}^\\mathrm{pred} / \\tau_{1}^\\mathrm{truth}$'),
    ('nsub2', '$\\tau_{2}^\\mathrm{pred} / \\tau_{2}^\\mathrm{truth}$'),
    ('nsub3', '$\\tau_{3}^\\mathrm{pred} / \\tau_{3}^\\mathrm{truth}$'))

for i in range(len(variable)):
    plt.clf()
    # histplot.histogram(np.divide(data_np[algo_0,i*2+3], data_np[algo_0,i*2+4]), bins = nbin, range = plot_range, label = 'ICS')
    histplot.histogram(np.divide(data_np[algo_3,i*2+3], data_np[algo_3,i*2+4]), bins = nbin, range = plot_range, label = 'CS')
    histplot.histogram(np.divide(data_np[algo_1,i*2+3], data_np[algo_1,i*2+4]), bins = nbin, range = plot_range, label = 'Supervised')
    # histplot.histogram(np.divide(data_np[algo_2,i*2+3], data_np[algo_2,i*2+4]), bins = nbin, range = plot_range, label = 'Self-supervised')
    plt.grid(True, linestyle = 'dashed')
    plt.xlabel(variable[i][1], ha = 'right', x = 1.)
    plt.legend()
    if d_1:
        plt.savefig('output_jetue/' + variable[i][0] + '_1.pdf')
    else:
        plt.savefig('output_jetue/' + variable[i][0] + '.pdf')
    # plt.clf()
    # print(data_np[algo_0,i*2+3].shape, data_np[algo_1,i*2+3].shape)
    # np.histogram2d(data_np[algo_0,i*2+3], data_np[algo_1,i*2+3])
    # plt.savefig(variable[i][0] + '_2d.pdf')

plt.clf()
histplot.histogram(data_np[algo_0,7], bins = nbin, range = (0, 200), label = 'ICS', linewidth=2) #, fontsize=24
histplot.histogram(data_np[algo_3,7], bins = nbin, range = (0, 200), label = 'CS', linewidth=2)
histplot.histogram(data_np[algo_1,7], bins = nbin, range = (0, 200), label = 'Supervised', linewidth=2)
# histplot.histogram(data_np[algo_2,7], bins = nbin, range = (0, 200), label = 'Self-supervised')
histplot.histogram(data_np[algo_3,8], bins = nbin, range = (0, 200), label = 'hard jet', linewidth=2)
plt.grid(True, linestyle = 'dashed')
plt.xlabel('$p_T$', ha = 'right', x = 1.)
plt.legend()
plt.savefig('output_jetue/pT.pdf')

range_2d = ((0, 200), (0, 200))

plt.clf()
plt.tight_layout()
plt.hist2d(data_np[algo_0,8], data_np[algo_0,7], bins = nbin, range = range_2d, norm = mpl.colors.LogNorm())
# plt.hist2d(data_np[algo_0,8], data_np[algo_0,7], bins = nbin, range = range_2d, cmap=plt.cm.jet)
plt.xlabel('$p_T^\\mathrm{truth}\\:(\\mathrm{GeV}/c)$', ha = 'right', x = 1.)
plt.ylabel('$p_T^\\mathrm{pred}\\:(\\mathrm{GeV}/c)$', ha = 'right', y = 1.)
plt.title('ICS')
# plt.savefig('perp_ics.png')
plt.savefig('output_jetue/perp_ics.pdf')

plt.clf()
plt.hist2d(data_np[algo_3,8], data_np[algo_3,7], bins = nbin, range = range_2d, norm = mpl.colors.LogNorm())
plt.xlabel('$p_T^\\mathrm{truth}\\:(\\mathrm{GeV}/c)$', ha = 'right', x = 1.)
plt.ylabel('$p_T^\\mathrm{pred}\\:(\\mathrm{GeV}/c)$', ha = 'right', y = 1.)
plt.title('CS')
# plt.savefig('perp_ics.png')
plt.savefig('output_jetue/perp_cs.pdf')

plt.clf()
plt.hist2d(data_np[algo_1,8], data_np[algo_1,7], bins = nbin, range = range_2d, norm = mpl.colors.LogNorm())
plt.xlabel('$p_T^\\mathrm{truth}\\:(\\mathrm{GeV}/c)$', ha = 'right', x = 1.)
plt.ylabel('$p_T^\\mathrm{pred}\\:(\\mathrm{GeV}/c)$', ha = 'right', y = 1.)
plt.title('Supervised')
# plt.savefig('perp_sup.png')
plt.savefig('output_jetue/perp_sup.pdf')

# plt.clf()
# plt.hist2d(data_np[algo_2,8], data_np[algo_2,7], bins = nbin, range = range_2d, norm = mpl.colors.LogNorm())
# plt.xlabel('$p_T^\\mathrm{truth}\\:(\\mathrm{GeV}/c)$', ha = 'right', x = 1.)
# plt.ylabel('$p_T^\\mathrm{pred}\\:(\\mathrm{GeV}/c)$', ha = 'right', y = 1.)
# plt.title('Self-supervised')
# # plt.savefig('perp_ssup.png')
# plt.savefig('output_jetue/perp_ssup.pdf')

plt.clf()
plt.hist2d(data_np[algo_9,8], data_np[algo_9,7], bins = nbin, range = range_2d, norm = mpl.colors.LogNorm())
plt.xlabel('$p_T^\\mathrm{truth}\\:(\\mathrm{GeV}/c)$', ha = 'right', x = 1.)
plt.ylabel('$p_T^\\mathrm{pred}\\:(\\mathrm{GeV}/c)$', ha = 'right', y = 1.)
# plt.savefig('perp_rho_a.png')
plt.savefig('output_jetue/perp_rho_a.pdf')

plt.clf()
plt.hist2d(data_np[algo_12,8], data_np[algo_12,7], bins = nbin, range = range_2d, norm = mpl.colors.LogNorm())
plt.xlabel('$p_T^\\mathrm{truth}\\:(\\mathrm{GeV}/c)$', ha = 'right', x = 1.)
plt.ylabel('$p_T^\\mathrm{pred}\\:(\\mathrm{GeV}/c)$', ha = 'right', y = 1.)
# plt.savefig('perp_ics_vs_sup.png')
plt.savefig('output_jetue/perp_ics_vs_sup.pdf')

plt.clf()
plt.hist2d(data_np[algo_0,-1], data_np[algo_0,0], bins = nbin, range = range_2d, norm = mpl.colors.LogNorm())
plt.xlabel('$\\rho_\\mathrm{truth}\\:(\\mathrm{GeV}/c)$', ha = 'right', x = 1.)
plt.ylabel('$\\rho_\\mathrm{pred}\\:(\\mathrm{GeV}/c)$', ha = 'right', y = 1.)
# plt.savefig('rho_fj_vs_truth.png')
plt.savefig('output_jetue/rho_fj_vs_truth.pdf')

# ### pT jet vs. pT constituents
# plt.clf()
# plt.hist2d(data_np[algo_3,8], data_np[algo_0,0], bins = nbin, range = range_2d, norm = mpl.colors.LogNorm())
# plt.xlabel('$p_T^\\mathrm{jet}(\\mathrm{GeV}/c)$', ha = 'right', x = 1.)
# plt.ylabel('$p_T^\\mathrm{constituents}(\\mathrm{GeV}/c)$', ha = 'right', y = 1.)
# # plt.savefig('rho_fj_vs_truth.png')
# plt.savefig('output_jetue/rho_fj_vs_truth.pdf')

### plot pseudorapidity
plt.clf()
histplot.histogram(data_np[algo_0, 3], bins = nbin, range = (-1, 1), label = 'ICS')
histplot.histogram(data_np[algo_3, 3], bins = nbin, range = (-1, 1), label = 'CS')
histplot.histogram(data_np[algo_1, 3], bins = nbin, range = (-1, 1), label = 'Supervised')
# histplot.histogram(data_np[algo_2, 3], bins = nbin, range = (-1, 1), label = 'Self-supervised')
histplot.histogram(data_np[algo_3, 4], bins = nbin, range = (-1, 1), label = 'hard jet')
plt.grid(True, linestyle = 'dashed')
plt.xlabel('$\\eta$', ha = 'right', x = 1.)
plt.legend()
plt.savefig('output_jetue/prap.pdf')

### plot some variables
variable_1 = (
    ('y', '$y$'),
    ('phi', '$\\phi$'),
    ('N', 'Number of constituents'))

for i in range(len(variable_1)):
    plt.clf()
    histplot.histogram(data_np[algo_0, i*2+27], bins = nbin, range = (-3, 100), label = 'ICS')
    histplot.histogram(data_np[algo_3, i*2+27], bins = nbin, range = (-3, 100), label = 'CS')
    histplot.histogram(data_np[algo_1, i*2+27], bins = nbin, range = (-3, 100), label = 'Supervised')
    # histplot.histogram(data_np[algo_2, i*2+27], bins = nbin, range = (-3, 100), label = 'Self-supervised')
    histplot.histogram(data_np[algo_3, i*2+28], bins = nbin, range = (-3, 100), label = 'hard jet')
    plt.grid(True, linestyle = 'dashed')
    plt.xlabel(variable_1[i][1], ha = 'right', x = 1.)
    plt.legend()
    plt.savefig('output_jetue/' + variable_1[i][0] + '.pdf')

# ### plot 3D eta vs phi vs. pT
# # defining surface and axes
# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# ax.view_init(elev=20, azim=32)
# # ax.plot_trisurf(data_np[algo_3, 8], data_np[algo_3, 4], data_np[algo_3, 30], cmap='hsv', linewidth=0, antialiased=False)
# # ax.scatter(data_np[algo_3, 8], data_np[algo_3, 4], data_np[algo_3, 30], c=data_np[algo_3, 8], cmap='hsv')
# p = ax.scatter(data_np[algo_3, 8], data_np[algo_3, 4], data_np[algo_3, 30], c=data_np[algo_3, 8], label=data_np[algo_3, 8], cmap='magma')
# #plt.tight_layout()
# ax.set_xlabel('$p_T^\\mathrm{truth}$', labelpad=12)
# ax.set_ylabel('$\\eta$', labelpad=12)
# ax.set_zlabel('$\\phi$', labelpad=12)
# fig.colorbar(p, ax=ax)
# plt.show()
# fig.savefig('output_jetue/etaPhiPt.pdf')

### number of constituents per jet
plt.clf()
histplot.histogram(data_np[algo_0, 31], bins = nbin, range = (-1, 1), label = 'ICS')
histplot.histogram(data_np[algo_3, 31], bins = nbin, range = (-1, 1), label = 'CS')
histplot.histogram(data_np[algo_1, 31], bins = nbin, range = (-1, 1), label = 'Supervised')
# histplot.histogram(data_np[algo_2, 3], bins = nbin, range = (-1, 1), label = 'Self-supervised')
histplot.histogram(data_np[algo_3, 32], bins = nbin, range = (-1, 1), label = 'hard jet')
plt.grid(True, linestyle = 'dashed')
plt.xlabel('$\\eta$', ha = 'right', x = 1.)
plt.legend()
plt.savefig('output_jetue/prap.pdf')