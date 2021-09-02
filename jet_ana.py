import sys, os
from ROOT import TH1F, TH2D, TFile, TCanvas, TFile, gROOT, TLegend, gStyle, TGraphErrors, gPad
from math import sqrt
import numpy as np
# from alice_data_analysis import AnalyzeDataFile

# Run ROOT in batch mode to avoid losing focus when plotting
gROOT.SetBatch(True)
# modify title and label sizes
gStyle.SetTitleSize(.06, "xyz")
gStyle.SetTitleOffset(1.3, "y")
gStyle.SetLabelSize(.06, "xyz")
gStyle.SetPadTopMargin(0.12)
gStyle.SetPadBottomMargin(0.16)
gStyle.SetPadRightMargin(0.16) # 2d:0.17 , 1d:0.05
gStyle.SetPadLeftMargin(0.16)  # 2d:0.16, 1d:0.15
gStyle.SetOptStat(0)
gStyle.SetLineWidth(2)
gStyle.SetPadGridX(True)
gStyle.SetPadGridY(True)

# Open the inpput tree
try:
    fin = TFile.Open('/home/nacer/analysis/alice_data_analysis_output.root')
except FileNotFoundError:
    print(f'file {fin} does not exist')
t = fin.tjets

## save plots in ROOT file
fout = '~/analysis/output'
figout = TFile(f'{fout}/jet_ana_output.root','RECREATE')

# plot jet soft drop z
c_sd01_z = TCanvas('c_sd01_z', 'c_sd01_z', 700,500)
c_sd01_z.cd()
h_sd01_z = TH1F('h_sd01_z', ';z_{SD};Counts', 20, 0.1, 0.5)
t.Project('h_sd01_z','j_sd01_z')
h_sd01_z.Draw('e')
h_sd01_z.Write()
c_sd01_z.Print(f'{fout}/c_sd01_z.root', "root")
c_sd01_z.Print(f'{fout}/c_sd01_z.pdf', "pdf")

# plot jet soft drop zg
c_zg = TCanvas('c_zg', 'c_zg', 700,500)
c_zg.cd()
h_zg = TH1F('h_zg', ';#it{z}_{g};Counts', 20, 0.1, 0.5)
t.Project('h_zg','zg')
h_zg.Draw('e')
h_zg.Write()
c_zg.Print(f'{fout}/c_zg.root', "root")
c_zg.Print(f'{fout}/c_zg.pdf', "pdf")