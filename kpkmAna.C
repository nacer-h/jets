#include "TFile.h"
#include "TH1F.h"
#include "TF1.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphMultiErrors.h"
#include "TLatex.h"
#include "TText.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TEllipse.h"
#include "TTree.h"
#include "TDatabasePDG.h"
#include "TDatabasePDG.h"
#include "TLorentzVector.h"
#include <iostream>
#include <fstream>
#include "TFormula.h"
#include "TLatex.h"
#include "TSystem.h"
#include "TString.h"
#include "TRatioPlot.h"

using namespace std;

void kpkmAna()
{
    gStyle->SetPalette(kRainBow); // choose different Palette color scheme
    gStyle->SetOptStat(0); // remove stat box
    gStyle->SetTitleSize(.05, "xyz");
    gStyle->SetTitleOffset(1.4, "y");
    // gStyle->SetLabelOffset(5.0, "y");
    gStyle->SetLabelSize(.05, "xyz");
    gStyle->SetPadTopMargin(0.12);
    gStyle->SetPadBottomMargin(0.15);
    gStyle->SetPadRightMargin(0.17); // 2d:0.17 , 1d:0.05
    gStyle->SetPadLeftMargin(0.16);  // 2d:0.16, 1d:0.15
    gStyle->SetLineWidth(2);
    gStyle->SetPadGridX(true);
    gStyle->SetPadGridY(true);

    TString outDir = "/Users/ahamdi/lbl/upc_loc/Phi0Ana/output_Phi0Ana";
    TFile *f_data = TFile::Open(Form("%s/Phi0Ana_data.root", outDir.Data()));
    TFile *f_mc = TFile::Open(Form("%s/Phi0Ana_mc.root", outDir.Data()));
    if(f_data == nullptr || f_mc == nullptr)
        cout<<"something went wrong! abort!"<<endl;
    
    TFile *outputFile = new TFile(Form("%s/kpkmAna.root",outDir.Data()), "RECREATE");

    FILE *tab_sys = fopen(Form("%s/tab_sys.tex",outDir.Data()), "w");
    fprintf(tab_sys,"\\documentclass[10pt]{extarticle}\n \\usepackage[margin=0.1in]{geometry}\n \\usepackage{tabularx}\n \\usepackage{caption} \n \\usepackage{makecell} \n \\captionsetup{labelformat=empty}\n \\begin{document}\n \\begin{table}[!htbp]\n \\centering\n \\caption{Systematic errors in pair rapidity bins}\n \\begin{tabular}{c|c|c|c|c|c}\n \\hline\n $y$ (rad) & \\thead{$P_{t}$\\\\(0.08, 0.1, 0.12)} & \\thead{Sig/Bkg range\\\\(3.5$\\sigma$, 3$\\sigma$, 2.5$\\sigma$)} & \\thead{Reject $\\pi^\\pm$ \\\\(1.5$\\sigma$, 2$\\sigma$, 2.5$\\sigma$)} & \\thead{Reject $\\mu^\\pm$ \\\\(1.5$\\sigma$, 2$\\sigma$, 2.5$\\sigma$)} & \\thead{Reject $e^\\pm$ \\\\(1.5$\\sigma$, 2$\\sigma$, 2.5$\\sigma$)} \\\\ \n \\hline\n");
    
    // Pair ransverse Momentum 
    auto cPtPair_all = new TCanvas("cPtPair_all", "cPtPair_all", 900, 600);
    cPtPair_all->cd();
    cPtPair_all->SetLogy();
    auto hPtPair_data = (TH1F *)f_data->Get("hPtPair");
    auto hPtPair_mc = (TH1F *)f_mc->Get("hPtPair");
    // hPtPair_data->Rebin(2);
    // hPtPair_mc->Rebin(2);
    hPtPair_data->Scale(1./hPtPair_data->Integral());
    hPtPair_mc->Scale(1./hPtPair_mc->Integral());
    // hPtPair_mc->SetMinimum(1.);
    hPtPair_mc->SetMarkerColor(2);
    hPtPair_mc->SetLineColor(2);
    hPtPair_mc->Draw();
    hPtPair_data->Draw("same");
    auto lPtPair_all = new TLegend(0.75, 0.75, 0.85, 0.88);
    lPtPair_all->SetTextSize(0.03);
    lPtPair_all->SetBorderSize(0);
    lPtPair_all->AddEntry(hPtPair_data, "Data", "l");
    lPtPair_all->AddEntry(hPtPair_mc, "MC", "l");
    lPtPair_all->Draw();
    cPtPair_all->Print(Form("%s/cPtPair_all.root", outDir.Data()),"root");
    cPtPair_all->Print(Form("%s/cPtPair_all.pdf", outDir.Data()),"pdf");

    // Pair Azimuthal angle 
    auto cAzimuthal_all = new TCanvas("cAzimuthal_all", "cAzimuthal_all", 900, 600);
    cAzimuthal_all->cd();
    // padAzimuthal_1->cd();    
    auto hAzimuthal_data = (TH1F *)f_data->Get("hAzimuthal");
    auto hAzimuthal_mc = (TH1F *)f_mc->Get("hAzimuthal");
    hAzimuthal_mc->SetTitle(";#phi (rad);Counts (Normalized)");
    hAzimuthal_data->Rebin(2);
    hAzimuthal_mc->Rebin(2);
    hAzimuthal_data->Scale(1./hAzimuthal_data->Integral());
    hAzimuthal_mc->Scale(1./hAzimuthal_mc->Integral());
    hAzimuthal_data->SetMinimum(0.);
    hAzimuthal_mc->SetMinimum(0.);
    hAzimuthal_mc->SetMarkerColor(2);
    hAzimuthal_mc->SetLineColor(2);
    hAzimuthal_mc->Draw("e");
    hAzimuthal_data->Draw("esame");
    auto lAzimuthal_all = new TLegend(0.75, 0.35, 0.85, 0.5);
    lAzimuthal_all->SetTextSize(0.03);
    lAzimuthal_all->SetBorderSize(0);
    lAzimuthal_all->AddEntry(hAzimuthal_data, "Data", "l");
    lAzimuthal_all->AddEntry(hAzimuthal_mc, "MC", "l");
    lAzimuthal_all->Draw();
    // auto hRatioAzimuthal = new TRatioPlot(hAzimuthal_mc, hAzimuthal_data);
    // hRatioAzimuthal->Draw();
    // hRatioAzimuthal->GetLowerRefYaxis()->SetTitle("ratio");
    // hRatioAzimuthal->GetUpperRefYaxis()->SetTitle("entries");
    cAzimuthal_all->Print(Form("%s/cAzimuthal_all.root", outDir.Data()),"root");
    cAzimuthal_all->Print(Form("%s/cAzimuthal_all.pdf", outDir.Data()),"pdf");

    // Pair VertexZ
    auto cVertexZ_all = new TCanvas("cVertexZ_all", "cVertexZ_all", 900, 600);
    cVertexZ_all->cd();
    auto hVertexZ_data = (TH1F *)f_data->Get("hVertexZ");
    auto hVertexZ_mc = (TH1F *)f_mc->Get("hVertexZ");
    hVertexZ_mc->SetTitle(";Vertex z-axis (cm);Counts (Normalized)");
    hVertexZ_data->Rebin(2);
    hVertexZ_mc->Rebin(2);
    hVertexZ_data->Scale(1./hVertexZ_data->Integral());
    hVertexZ_mc->Scale(1./hVertexZ_mc->Integral());
    hVertexZ_data->SetMinimum(0.);
    hVertexZ_mc->SetMinimum(0.);
    hVertexZ_mc->SetMarkerColor(2);
    hVertexZ_mc->SetLineColor(2);
    hVertexZ_mc->Draw("e");
    hVertexZ_data->Draw("esame");
    // auto hRatioVertexZ = new TRatioPlot(hVertexZ_mc, hVertexZ_data);
    // hRatioVertexZ->Draw();
    // hRatioVertexZ->GetLowerRefYaxis()->SetTitle("ratio");
    // hRatioVertexZ->GetUpperRefYaxis()->SetTitle("entries");
    auto lVertexZ_all = new TLegend(0.75, 0.75, 0.85, 0.88);
    lVertexZ_all->SetTextSize(0.03);
    lVertexZ_all->SetBorderSize(0);
    lVertexZ_all->AddEntry(hVertexZ_data, "Data", "l");
    lVertexZ_all->AddEntry(hVertexZ_mc, "MC", "l");
    lVertexZ_all->Draw();
    cVertexZ_all->Print(Form("%s/cVertexZ_all.root", outDir.Data()),"root");
    cVertexZ_all->Print(Form("%s/cVertexZ_all.pdf", outDir.Data()),"pdf");
    // printf("------------------ No problem up to here ! \n");

    // ------------- TPC dE/dx s.t.d 
    // --- Kaon
    auto cStdDEdxKaon_all = new TCanvas("cStdDEdxKaon_all", "cStdDEdxKaon_all", 900, 600);
    cStdDEdxKaon_all->cd();
    auto hStdDEdxKaon_data = (TH1F *)f_data->Get("hStdDEdxKaon");
    auto hStdDEdxKaon_mc = (TH1F *)f_mc->Get("hStdDEdxKaon");
    hStdDEdxKaon_mc->SetTitle(";#sigma^{K^{#pm}}_{dE/dx};Counts (Normalized)");
    hStdDEdxKaon_data->Scale(1./hStdDEdxKaon_data->Integral());
    hStdDEdxKaon_mc->Scale(1./hStdDEdxKaon_mc->Integral());
    hStdDEdxKaon_data->SetMinimum(0.);
    hStdDEdxKaon_mc->SetMinimum(0.);
    hStdDEdxKaon_mc->SetMarkerColor(2);
    hStdDEdxKaon_mc->SetLineColor(2);
    hStdDEdxKaon_mc->Draw("hist");
    hStdDEdxKaon_data->Draw("hist same");
    auto lStdDEdxKaon_all = new TLegend(0.75, 0.75, 0.85, 0.88);
    lStdDEdxKaon_all->SetTextSize(0.03);
    lStdDEdxKaon_all->SetBorderSize(0);
    lStdDEdxKaon_all->AddEntry(hStdDEdxKaon_data, "Data", "l");
    lStdDEdxKaon_all->AddEntry(hStdDEdxKaon_mc, "MC", "l");
    lStdDEdxKaon_all->Draw();
    cStdDEdxKaon_all->Print(Form("%s/cStdDEdxKaon_all.root", outDir.Data()),"root");
    cStdDEdxKaon_all->Print(Form("%s/cStdDEdxKaon_all.pdf", outDir.Data()),"pdf");
    // --- Pion
    auto cStdDEdxPion_all = new TCanvas("cStdDEdxPion_all", "cStdDEdxPion_all", 900, 600);
    cStdDEdxPion_all->cd();
    auto hStdDEdxPion_data = (TH1F *)f_data->Get("hStdDEdxPion");
    auto hStdDEdxPion_mc = (TH1F *)f_mc->Get("hStdDEdxPion");
    hStdDEdxPion_data->SetTitle(";#sigma^{#pi^{#pm}}_{dE/dx};Counts (Normalized)");
    hStdDEdxPion_data->Scale(1./hStdDEdxPion_data->Integral());
    hStdDEdxPion_mc->Scale(1./hStdDEdxPion_mc->Integral());
    hStdDEdxPion_data->SetMinimum(0.);
    hStdDEdxPion_mc->SetMinimum(0.);
    hStdDEdxPion_mc->SetMarkerColor(2);
    hStdDEdxPion_mc->SetLineColor(2);
    hStdDEdxPion_data->Draw("hist");
    hStdDEdxPion_mc->Draw("hist same");
    auto lStdDEdxPion_all = new TLegend(0.75, 0.75, 0.85, 0.88);
    lStdDEdxPion_all->SetTextSize(0.03);
    lStdDEdxPion_all->SetBorderSize(0);
    lStdDEdxPion_all->AddEntry(hStdDEdxPion_data, "Data", "l");
    lStdDEdxPion_all->AddEntry(hStdDEdxPion_mc, "MC", "l");
    lStdDEdxPion_all->Draw();
    cStdDEdxPion_all->Print(Form("%s/cStdDEdxPion_all.root", outDir.Data()),"root");
    cStdDEdxPion_all->Print(Form("%s/cStdDEdxPion_all.pdf", outDir.Data()),"pdf");
    // --- Muon
    auto cStdDEdxMuon_all = new TCanvas("cStdDEdxMuon_all", "cStdDEdxMuon_all", 900, 600);
    cStdDEdxMuon_all->cd();
    auto hStdDEdxMuon_data = (TH1F *)f_data->Get("hStdDEdxMuon");
    auto hStdDEdxMuon_mc = (TH1F *)f_mc->Get("hStdDEdxMuon");
    hStdDEdxMuon_data->SetTitle(";#sigma^{#mu^{#pm}}_{dE/dx};Counts (Normalized)");
    hStdDEdxMuon_data->Scale(1./hStdDEdxMuon_data->Integral());
    hStdDEdxMuon_mc->Scale(1./hStdDEdxMuon_mc->Integral());
    hStdDEdxMuon_data->SetMinimum(0.);
    hStdDEdxMuon_mc->SetMinimum(0.);
    hStdDEdxMuon_mc->SetMarkerColor(2);
    hStdDEdxMuon_mc->SetLineColor(2);
    hStdDEdxMuon_data->Draw("hist");
    hStdDEdxMuon_mc->Draw("hist same");
    auto lStdDEdxMuon_all = new TLegend(0.75, 0.75, 0.85, 0.88);
    lStdDEdxMuon_all->SetTextSize(0.03);
    lStdDEdxMuon_all->SetBorderSize(0);
    lStdDEdxMuon_all->AddEntry(hStdDEdxMuon_data, "Data", "l");
    lStdDEdxMuon_all->AddEntry(hStdDEdxMuon_mc, "MC", "l");
    lStdDEdxMuon_all->Draw();
    cStdDEdxMuon_all->Print(Form("%s/cStdDEdxMuon_all.root", outDir.Data()),"root");
    cStdDEdxMuon_all->Print(Form("%s/cStdDEdxMuon_all.pdf", outDir.Data()),"pdf");
    // --- Electron
    auto cStdDEdxElectron_all = new TCanvas("cStdDEdxElectron_all", "cStdDEdxElectron_all", 900, 600);
    cStdDEdxElectron_all->cd();
    auto hStdDEdxElectron_data = (TH1F *)f_data->Get("hStdDEdxElectron");
    auto hStdDEdxElectron_mc = (TH1F *)f_mc->Get("hStdDEdxElectron");
    hStdDEdxElectron_data->SetTitle(";#sigma^{e^{#pm}}_{dE/dx};Counts (Normalized)");
    hStdDEdxElectron_data->Scale(1./hStdDEdxElectron_data->Integral());
    hStdDEdxElectron_mc->Scale(1./hStdDEdxElectron_mc->Integral());
    hStdDEdxElectron_data->SetMinimum(0.);
    hStdDEdxElectron_mc->SetMinimum(0.);
    hStdDEdxElectron_mc->SetMarkerColor(2);
    hStdDEdxElectron_mc->SetLineColor(2);
    hStdDEdxElectron_data->Draw("hist");
    hStdDEdxElectron_mc->Draw("hist same");
    auto lStdDEdxElectron_all = new TLegend(0.75, 0.75, 0.85, 0.88);
    lStdDEdxElectron_all->SetTextSize(0.03);
    lStdDEdxElectron_all->SetBorderSize(0);
    lStdDEdxElectron_all->AddEntry(hStdDEdxElectron_data, "Data", "l");
    lStdDEdxElectron_all->AddEntry(hStdDEdxElectron_mc, "MC", "l");
    lStdDEdxElectron_all->Draw();
    cStdDEdxElectron_all->Print(Form("%s/cStdDEdxElectron_all.root", outDir.Data()),"root");
    cStdDEdxElectron_all->Print(Form("%s/cStdDEdxElectron_all.pdf", outDir.Data()),"pdf");

    // // Invarian Mass 
    // auto cMassPairKaonUnLikeCharge_all = new TCanvas("cMassPairKaonUnLikeCharge_all", "cMassPairKaonUnLikeCharge_all", 900, 600);
    // cMassPairKaonUnLikeCharge_all->cd();
    // // cMassPairKaonUnLikeCharge_all->SetLogy();
    // auto hMassPairKaonUnLikeCharge_data = (TH1F *)f_data->Get("hMassPairKaonUnLikeCharge");
    // auto hMassPairKaonUnLikeCharge_mc = (TH1F *)f_mc->Get("hMassPairKaonUnLikeCharge");
    // hMassPairKaonUnLikeCharge_data->Scale(1./hMassPairKaonUnLikeCharge_data->Integral(), "width");
    // hMassPairKaonUnLikeCharge_mc->Scale(1./hMassPairKaonUnLikeCharge_mc->Integral(), "width");
    // hMassPairKaonUnLikeCharge_mc->SetMarkerColor(2);
    // hMassPairKaonUnLikeCharge_mc->SetLineColor(2);
    // hMassPairKaonUnLikeCharge_data->Draw("e");
    // hMassPairKaonUnLikeCharge_mc->Draw("esame");
    // auto lMassPairKaonUnLikeCharge_all = new TLegend(0.75, 0.75, 0.85, 0.88);
    // lMassPairKaonUnLikeCharge_all->SetTextSize(0.03);
    // lMassPairKaonUnLikeCharge_all->SetBorderSize(0);
    // lMassPairKaonUnLikeCharge_all->AddEntry(hMassPairKaonUnLikeCharge_data, "Data", "l");
    // lMassPairKaonUnLikeCharge_all->AddEntry(hMassPairKaonUnLikeCharge_mc, "MC", "l");
    // lMassPairKaonUnLikeCharge_all->Draw();
    // hMassPairKaonUnLikeCharge_data->Write("hMassPairKaonUnLikeCharge_data", TObject::kWriteDelete);
    // hMassPairKaonUnLikeCharge_mc->Write("hMassPairKaonUnLikeCharge_mc", TObject::kWriteDelete);
    // cMassPairKaonUnLikeCharge_all->Print(Form("%s/cMassPairKaonUnLikeCharge_all.root", outDir.Data()),"root");
    // cMassPairKaonUnLikeCharge_all->Print(Form("%s/cMassPairKaonUnLikeCharge_all.pdf", outDir.Data()),"pdf");

    // Invarian Mass 
    auto cMassPairKaon_all = new TCanvas("cMassPairKaon_all", "cMassPairKaon_all", 900, 600);
    cMassPairKaon_all->cd();
    // cMassPairKaon_all->SetLogy();
    auto hMassPairKaon_data = (TH1F *)f_data->Get("hMassPairKaonRejectAllSub");
    auto hMassPairKaon_mc = (TH1F *)f_mc->Get("hMassPairKaon");
    auto hMassPairKaon_UnLikeCharge_mc = (TH1F *)f_mc->Get("hMassPairKaonUnLikeCharge");
    auto hMassPairKaon_dEdxCut_mc = (TH1F *)f_mc->Get("hMassPairKaondEdxCut");
    auto hMassPairKaon_PtCut01_mc = (TH1F *)f_mc->Get("hMassPairKaonPtCut01");
    auto hMassPairKaon_RejectPionSub_mc = (TH1F *)f_mc->Get("hMassPairKaonRejectPionSub");
    auto hMassPairKaon_RejectPionMuonSub_mc = (TH1F *)f_mc->Get("hMassPairKaonRejectPionMuonSub");
    auto hMassPairKaon_RejectAllSub_mc = (TH1F *)f_mc->Get("hMassPairKaonRejectAllSub");
    auto hMassPairKaon_truth = (TH1F *)f_mc->Get("hMassPairKaon_truth");
    hMassPairKaon_data->Rebin(2);
    hMassPairKaon_mc->Rebin(6);
    hMassPairKaon_UnLikeCharge_mc->Rebin(6);
    hMassPairKaon_dEdxCut_mc->Rebin(6);
    hMassPairKaon_PtCut01_mc->Rebin(6);
    hMassPairKaon_RejectPionSub_mc->Rebin(2);
    hMassPairKaon_RejectPionMuonSub_mc->Rebin(2);
    hMassPairKaon_RejectAllSub_mc->Rebin(2);
    hMassPairKaon_truth->Rebin(6);
    hMassPairKaon_RejectAllSub_mc->SetMarkerColor(2);
    hMassPairKaon_RejectAllSub_mc->SetLineColor(2);
    hMassPairKaon_truth->SetMarkerColor(1);
    hMassPairKaon_truth->SetLineColor(1);
    hMassPairKaon_truth->Draw("e");
    hMassPairKaon_RejectAllSub_mc->Draw("esame");
    hMassPairKaon_data->Draw("esame");
    auto lMassPairKaon_all = new TLegend(0.75, 0.75, 0.85, 0.88);
    lMassPairKaon_all->SetTextSize(0.03);
    lMassPairKaon_all->SetBorderSize(0);
    lMassPairKaon_all->AddEntry(hMassPairKaon_data, "Data", "l");
    lMassPairKaon_all->AddEntry(hMassPairKaon_RejectAllSub_mc, "MC", "l");
    lMassPairKaon_all->AddEntry(hMassPairKaon_truth, "Generated", "l");
    lMassPairKaon_all->Draw();
    hMassPairKaon_data->Write("hMassPairKaon_data", TObject::kWriteDelete);
    hMassPairKaon_RejectAllSub_mc->Write("hMassPairKaon_RejectAllSub_mc", TObject::kWriteDelete);
    hMassPairKaon_truth->Write("hMassPairKaon_truth", TObject::kWriteDelete);
    cMassPairKaon_all->Print(Form("%s/cMassPairKaon_all.root", outDir.Data()),"root");
    cMassPairKaon_all->Print(Form("%s/cMassPairKaon_all.pdf", outDir.Data()),"pdf");

    // Number of events per run
    auto hEvents_data = (TH1F *)f_data->Get("hEvents");

    // Efficiency
    TGraphErrors *gEff = new TGraphErrors();
    TGraphErrors *gEff_UnLikeCharge = new TGraphErrors();
    TGraphErrors *gEff_dEdxCut = new TGraphErrors();
    TGraphErrors *gEff_PtCut01 = new TGraphErrors();
    TGraphErrors *gEff_RejectPionSub = new TGraphErrors();
    TGraphErrors *gEff_RejectPionMuonSub = new TGraphErrors();
    TGraphErrors *gEff_RejectAllSub = new TGraphErrors();
    gEff->SetMarkerStyle(kOpenSquare);
    gEff_UnLikeCharge->SetMarkerStyle(20);
    gEff_dEdxCut->SetMarkerStyle(20);
    gEff_PtCut01->SetMarkerStyle(20);
    gEff_RejectPionSub->SetMarkerStyle(20);
    gEff_RejectPionMuonSub->SetMarkerStyle(20);
    gEff_RejectAllSub->SetMarkerStyle(20);
    gEff->SetMarkerColor(kBlack);
    gEff_UnLikeCharge->SetMarkerColor(28);
    gEff_dEdxCut->SetMarkerColor(40);
    gEff_PtCut01->SetMarkerColor(kRed);
    gEff_RejectPionSub->SetMarkerColor(kGreen);
    gEff_RejectPionMuonSub->SetMarkerColor(kMagenta);
    gEff_RejectAllSub->SetMarkerColor(kBlue);
    gEff->SetMinimum(0.0);
    gEff->SetTitle("; m_{K^{+}K^{-}} (GeV/c^{2}); Efficiency (%)");

    // Cross Section
    TGraphErrors *gkpkmXSection = new TGraphErrors();
    gkpkmXSection->SetMarkerStyle(20);
    // gkpkmXSection->SetMarkerSize(1.5);
    gkpkmXSection->SetMarkerColor(kBlue);
    gkpkmXSection->SetMinimum(0.0);
    gkpkmXSection->SetTitle("; m_{K^{+}K^{-}} (GeV/c^{2}); Cross Section (\\mub)");

    // residual
    TGraph *gresidual = new TGraph();
    gresidual->SetMarkerStyle(20);
    gresidual->SetMarkerColor(kBlue);
    gresidual->SetMinimum(0.0);
    gresidual->SetTitle("; m_{K^{+}K^{-}} (GeV/c^{2}); N_{Truth} - N_{Reco} (GeV/c^{2})");

    for (Int_t i = 0; i < hMassPairKaon_RejectAllSub_mc->GetNbinsX(); ++i)
    {
        double eff = 0, deff = 0, eff_UnLikeCharge = 0, deff_UnLikeCharge = 0, eff_dEdxCut = 0, deff_dEdxCut = 0, eff_PtCut01 = 0, deff_PtCut01 = 0, eff_RejectPionSub = 0, deff_RejectPionSub = 0, eff_RejectPionMuonSub = 0, deff_RejectPionMuonSub = 0, eff_RejectAllSub = 0, deff_RejectAllSub = 0, xsec_kpkm = 0, dxsec_kpkm_stat = 0;
        double lumi_kpkm = 0.512; // micro barn^{-1}
        double Nsig_data = hMassPairKaon_data->GetBinContent(i+1);
        double dNsig_data = hMassPairKaon_data->GetBinError(i+1);
        double Nsig_truth = hMassPairKaon_truth->GetBinContent(i+1);
        double Nsig_mc = hMassPairKaon_mc->GetBinContent(i+1);
        double Nsig_UnLikeCharge_mc = hMassPairKaon_UnLikeCharge_mc->GetBinContent(i+1);
        double Nsig_dEdxCut_mc = hMassPairKaon_dEdxCut_mc->GetBinContent(i+1);
        double Nsig_PtCut01_mc = hMassPairKaon_PtCut01_mc->GetBinContent(i+1);
        double Nsig_RejectPionSub_mc = hMassPairKaon_RejectPionSub_mc->GetBinContent(i+1);
        double Nsig_RejectPionMuonSub_mc = hMassPairKaon_RejectPionMuonSub_mc->GetBinContent(i+1);
        double Nsig_RejectAllSub_mc = hMassPairKaon_RejectAllSub_mc->GetBinContent(i+1);
        double dNsig_mc = hMassPairKaon_mc->GetBinError(i+1);
        double dNsig_UnLikeCharge_mc = hMassPairKaon_UnLikeCharge_mc->GetBinError(i+1);
        double dNsig_dEdxCut_mc = hMassPairKaon_dEdxCut_mc->GetBinError(i+1);
        double dNsig_PtCut01_mc = hMassPairKaon_PtCut01_mc->GetBinError(i+1);
        double dNsig_RejectPionSub_mc = hMassPairKaon_RejectPionSub_mc->GetBinError(i+1);
        double dNsig_RejectPionMuonSub_mc = hMassPairKaon_RejectPionMuonSub_mc->GetBinError(i+1);
        double dNsig_RejectAllSub_mc = hMassPairKaon_RejectAllSub_mc->GetBinError(i+1);
        double kpkmMass = hMassPairKaon_RejectAllSub_mc->GetXaxis()->GetBinCenter(i);
        double diffMcTruth = Nsig_truth - Nsig_RejectAllSub_mc;

        if(Nsig_truth != 0)
        {
            if(Nsig_mc != 0)
            {
                eff = Nsig_mc / Nsig_truth; // Efficiency = N_observed/N_generated
                deff = eff * (dNsig_mc / Nsig_mc);
            }
            if(Nsig_UnLikeCharge_mc != 0)
            {
                eff_UnLikeCharge = Nsig_UnLikeCharge_mc / Nsig_truth; // Efficiency = N_observed/N_generated
                deff_UnLikeCharge = eff_UnLikeCharge * (dNsig_UnLikeCharge_mc / Nsig_UnLikeCharge_mc);
            }
            if(Nsig_dEdxCut_mc != 0)
            {
                eff_dEdxCut = Nsig_dEdxCut_mc / Nsig_truth; // Efficiency = N_observed/N_generated
                deff_dEdxCut = eff_dEdxCut * (dNsig_dEdxCut_mc / Nsig_dEdxCut_mc);
            }
            if(Nsig_PtCut01_mc != 0)
            {
                eff_PtCut01 = Nsig_PtCut01_mc / Nsig_truth;
                deff_PtCut01 = eff_PtCut01 * (dNsig_PtCut01_mc / Nsig_PtCut01_mc);
            }
            if (Nsig_RejectPionSub_mc != 0)
            {
                eff_RejectPionSub = Nsig_RejectPionSub_mc / Nsig_truth;
                deff_RejectPionSub = eff_RejectPionSub * (dNsig_RejectPionSub_mc / Nsig_RejectPionSub_mc);
            }
            if (Nsig_RejectPionMuonSub_mc != 0)
            {
                eff_RejectPionMuonSub = Nsig_RejectPionMuonSub_mc / Nsig_truth;
                deff_RejectPionMuonSub = eff_RejectPionMuonSub * (dNsig_RejectPionMuonSub_mc / Nsig_RejectPionMuonSub_mc);
            }
            if (Nsig_RejectAllSub_mc != 0)
            {
                eff_RejectAllSub = Nsig_RejectAllSub_mc / Nsig_truth;
                deff_RejectAllSub = eff_RejectAllSub * (dNsig_RejectAllSub_mc / Nsig_RejectAllSub_mc);
            }

            if(eff_RejectAllSub != 0 && Nsig_data != 0)
            {
                xsec_kpkm = Nsig_data / (eff_RejectAllSub * lumi_kpkm);
                dxsec_kpkm_stat = xsec_kpkm * (dNsig_data / Nsig_data);
            }
            else
            {
                xsec_kpkm = 0;
                dxsec_kpkm_stat = 0;
            }

        }
        // else
        // {
        //     eff_dEdxCut = eff_PtCut01 = eff_RejectPionSub = eff_RejectPionMuonSub = eff_RejectAllSub = 0;
        //     deff_dEdxCut = deff_PtCut01 = deff_RejectPionSub = deff_RejectPionMuonSub = deff_RejectAllSub = 0;
        // }

        gEff->SetPoint(i, kpkmMass, eff * 100);
        gEff_UnLikeCharge->SetPoint(i, kpkmMass, eff_UnLikeCharge * 100);
        gEff_dEdxCut->SetPoint(i, kpkmMass, eff_dEdxCut * 100);
        gEff_PtCut01->SetPoint(i, kpkmMass, eff_PtCut01 * 100);
        gEff_RejectPionSub->SetPoint(i, kpkmMass, eff_RejectPionSub * 100);
        gEff_RejectPionMuonSub->SetPoint(i, kpkmMass, eff_RejectPionMuonSub * 100);
        gEff_RejectAllSub->SetPoint(i, kpkmMass, eff_RejectAllSub * 100);

        gEff->SetPointError(i, 0, deff * 100);
        gEff_UnLikeCharge->SetPointError(i, 0, deff_UnLikeCharge * 100);
        gEff_dEdxCut->SetPointError(i, 0, deff_dEdxCut * 100);
        gEff_PtCut01->SetPointError(i, 0, deff_PtCut01 * 100);
        gEff_RejectPionSub->SetPointError(i, 0, deff_RejectPionSub * 100);
        gEff_RejectPionMuonSub->SetPointError(i, 0, deff_RejectPionMuonSub * 100);
        gEff_RejectAllSub->SetPointError(i, 0, deff_RejectAllSub * 100);

        gkpkmXSection->SetPoint(i, kpkmMass, xsec_kpkm);
        gkpkmXSection->SetPointError(i, 0, dxsec_kpkm_stat);

        gresidual->SetPoint(i, kpkmMass, diffMcTruth);

        // cout<<" i = "<<i<<" | Nsig_truth = "<<Nsig_truth<<" | Nsig_mc = "<<Nsig_mc<<" | Nsig_UnLikeCharge_mc = "<<Nsig_UnLikeCharge_mc<<" | Nsig_dEdxCut_mc = "<<Nsig_dEdxCut_mc<<endl;
    }

    TCanvas *ckpkmEfficiency = new TCanvas("ckpkmEfficiency", "ckpkmEfficiency", 900, 600);
    ckpkmEfficiency->cd();
    auto lkpkmEfficiency = new TLegend(0.18, 0.65, 0.35, 0.95);
    lkpkmEfficiency->SetTextSize(0.03);
    lkpkmEfficiency->SetBorderSize(0);    
    gEff->Draw("AP");
    gEff_UnLikeCharge->Draw("PSAME");
    gEff_dEdxCut->Draw("PSAME");
    gEff_PtCut01->Draw("PSAME");
    gEff_RejectPionSub->Draw("PSAME");
    gEff_RejectPionMuonSub->Draw("PSAME");
    gEff_RejectAllSub->Draw("PSAME");
    lkpkmEfficiency->AddEntry(gEff, "pair tracks", "p");
    lkpkmEfficiency->AddEntry(gEff_UnLikeCharge, "opposite charge", "p");
    lkpkmEfficiency->AddEntry(gEff_dEdxCut, "|#sigma^{K^{#pm}}_{dE/dx}|<3", "p");
    lkpkmEfficiency->AddEntry(gEff_PtCut01, "|#sigma^{K^{#pm}}_{dE/dx}|<3 + p_{T}<0.1 GeV/c", "p");
    lkpkmEfficiency->AddEntry(gEff_RejectPionSub, "p_{T}<0.1. + |#sigma^{(#pi)}_{dE/dx}|>2 + Bkg subtraction", "p");
    lkpkmEfficiency->AddEntry(gEff_RejectPionMuonSub, "p_{T}<0.1 + |#sigma^{(#pi, #mu)}_{dE/dx}|>2 + Bkg subtraction", "p");
    lkpkmEfficiency->AddEntry(gEff_RejectAllSub, "p_{T}<0.1 + |#sigma^{(#pi, #mu, e)}_{dE/dx}|>2 + Bkg subtraction", "p");
    lkpkmEfficiency->Draw();
    gEff_dEdxCut->Write(); // Form("h%s_gEff_dEdxCut", name.Data()), TObject::kWriteDelete
    ckpkmEfficiency->Print(Form("%s/ckpkmEfficiency.root", outDir.Data()),"root");
    ckpkmEfficiency->Print(Form("%s/ckpkmEfficiency.pdf", outDir.Data()),"pdf");
    
    TCanvas *ckpkmXSection = new TCanvas("ckpkmXSection", "ckpkmXSection", 900, 600);
    ckpkmXSection->cd();
    ckpkmXSection->SetLogy();
    gkpkmXSection->Draw("AP");
    gkpkmXSection->Write(); // Form("h%s_gkpkmXSection", name.Data()), TObject::kWriteDelete
    ckpkmXSection->Print(Form("%s/ckpkmXSection.root", outDir.Data()),"root");
    ckpkmXSection->Print(Form("%s/ckpkmXSection.pdf", outDir.Data()),"pdf");

    TCanvas *cresidual = new TCanvas("cresidual", "cresidual", 900, 600);
    cresidual->cd();
    gresidual->Draw("AP");
    gresidual->Write(); // Form("h%s_gresidual", name.Data()), TObject::kWriteDelete
    cresidual->Print(Form("%s/cresidual.root", outDir.Data()),"root");
    cresidual->Print(Form("%s/cresidual.pdf", outDir.Data()),"pdf");

    // ---------- Cross section in rapidity bins

    auto hMassPairKaonVsRapidity_truth = (TH2D *)f_mc->Get("hMassPairKaonVsRapidity_truth");
    auto hMassPairKaonVsRapiditySub_mc = (TH2D *)f_mc->Get("hMassPairKaonVsRapiditySub");
    auto hMassPairKaonVsRapiditySub_data = (TH2D *)f_data->Get("hMassPairKaonVsRapiditySub");

    // TGraphErrors *gdxsectiondy = new TGraphErrors();
    // gdxsectiondy->SetMarkerStyle(20);
    // // gdxsectiondy->SetMarkerSize(1.5);
    // gdxsectiondy->SetMarkerColor(kBlue);
    // gdxsectiondy->SetMinimum(0.0);
    // gdxsectiondy->SetTitle("; y; d\\sigma/dy (mb)");

    TCanvas *cMassPairKaon = new TCanvas("cMassPairKaon", "cMassPairKaon", 1000, 1000);
    cMassPairKaon->Divide(3, 3);

    // =================== Systematic Errors ====================
    const Int_t np = hMassPairKaonVsRapiditySub_data->GetNbinsX();
    Double_t x[np], y[np], exl[np], exh[np], eylstat[np], eyhstat[np], eylsys[np], eyhsys[np];

    double xsec_optimal[]             = {4.085, 5.170, 3.156}; // pt < 0.1; regions: 9, 12.25, 6.25;
    double xsec_pt012[]               = {3.937, 5.222, 3.483}; // pt < 0.12
    double xsec_pt008[]               = {3.566, 4.980, 3.550}; // pt < 0.08
    double xsec_rad35[]               = {4.488, 5.252, 3.124}; // regions: 12.25
    double xsec_rad25[]               = {3.768, 5.052, 3.267}; // regions: 6.25
    double xsec_rejectPion25std[]     = {4.276, 5.479, 3.276}; // regions: 12.25
    double xsec_rejectPion15std[]     = {4.024, 5.165, 3.107}; // regions: 12.25
    double xsec_rejectMuon25std[]     = {5.382, 6.273, 3.549}; // regions: 12.25
    double xsec_rejectMuon15std[]     = {4.938, 4.487, 3.542}; // regions: 12.25
    double xsec_rejectElectron25std[] = {4.935, 6.479, 3.654}; // regions: 12.25
    double xsec_rejectElectron15std[] = {4.882, 4.499, 2.446}; // regions: 12.25

    for (Int_t i = 1; i <= hMassPairKaonVsRapiditySub_data->GetNbinsX(); ++i)
    {
        cMassPairKaon->cd(i);
        TH1D *hMassPairKaon_py_truth = hMassPairKaonVsRapidity_truth->ProjectionY(Form("_truth_%d", i), i, i);
        hMassPairKaon_py_truth->Draw();
        cMassPairKaon->cd(i+3);
        TH1D *hMassPairKaon_py_mc = hMassPairKaonVsRapiditySub_mc->ProjectionY(Form("_mc_%d", i), i, i);
        hMassPairKaon_py_mc->Draw();
        cMassPairKaon->cd(i+6);
        TH1D *hMassPairKaon_py_data = hMassPairKaonVsRapiditySub_data->ProjectionY(Form("_data_%d", i), i, i);
        hMassPairKaon_py_data->Draw();

        double ymin = hMassPairKaonVsRapiditySub_data->GetXaxis()->GetBinLowEdge(i);
        double ymax = hMassPairKaonVsRapiditySub_data->GetXaxis()->GetBinUpEdge(i);
        hMassPairKaon_py_truth->SetTitle(Form("%.2f<y<%.2f;m_{K^{+}K^{-}} (GeV/c^{2});Counts",ymin,ymax));
        hMassPairKaon_py_mc->SetTitle(Form("%.2f<y<%.2f;m_{K^{+}K^{-}} (GeV/c^{2});Counts",ymin,ymax));
        hMassPairKaon_py_data->SetTitle(Form("%.2f<y<%.2f;m_{K^{+}K^{-}} (GeV/c^{2});Counts",ymin,ymax));

        double eff = 0, deff = 0, xsec_kpkm = 0, dxsec_kpkm_stat = 0;
        double lumi_kpkm = 485; // milli barn^{-1}
        double fpileup = 1.111; // from Rho(770) analysis
        double Nsig_truth = hMassPairKaon_py_truth->Integral();
        double dNsig_mc;
        double Nsig_mc = hMassPairKaon_py_mc->IntegralAndError(1, hMassPairKaon_py_mc->GetNbinsX(), dNsig_mc, "");
        double dNsig_data;
        double Nsig_data = hMassPairKaon_py_data->IntegralAndError(1, hMassPairKaon_py_data->GetNbinsX(), dNsig_data, "");
        double pairRapidity = hMassPairKaonVsRapiditySub_data->GetXaxis()->GetBinCenter(i);

        eff = Nsig_mc / Nsig_truth; // Efficiency = N_observed/N_generated
        deff = eff * (dNsig_mc / Nsig_mc);

        // =================== Systematic Errors ====================
        double xsec_pt[]             = {xsec_optimal[i-1], xsec_pt012[i-1], xsec_pt008[i-1]};
        double xsec_rad[]            = {xsec_optimal[i-1], xsec_rad35[i-1], xsec_rad25[i-1]};
        double xsec_rejectPion[]     = {xsec_optimal[i-1], xsec_rejectPion25std[i-1], xsec_rejectPion15std[i-1]};
        double xsec_rejectMuon[]     = {xsec_optimal[i-1], xsec_rejectMuon25std[i-1], xsec_rejectMuon15std[i-1]};
        double xsec_rejectElectron[] = {xsec_optimal[i-1], xsec_rejectElectron25std[i-1], xsec_rejectElectron15std[i-1]};
        double err_xsec_pt             = TMath::RMS(np, xsec_pt)/xsec_optimal[i-1];
        double err_xsec_rad            = TMath::RMS(np, xsec_rad)/xsec_optimal[i-1];
        double err_xsec_rejectPion     = TMath::RMS(np, xsec_rejectPion)/xsec_optimal[i-1];
        double err_xsec_rejectMuon     = TMath::RMS(np, xsec_rejectMuon)/xsec_optimal[i-1];
        double err_xsec_rejectElectron = TMath::RMS(np, xsec_rejectElectron)/xsec_optimal[i-1];

        // from rho0 analysis
        double err_trk_select  = 0.015;
        double err_trk_match  = 0.040;
        double err_trk_eff  = 0.010;
        double err_SPD_chip  = 0.010;
        double err_PileUp  = 0.038;
        double err_Lumi  = 0.050;

        double err_sys = TMath::Sqrt(err_xsec_pt*err_xsec_pt + err_xsec_rad*err_xsec_rad + err_xsec_rejectPion*err_xsec_rejectPion + err_xsec_rejectMuon*err_xsec_rejectMuon + err_xsec_rejectElectron*err_xsec_rejectElectron + err_trk_select*err_trk_select + err_trk_match*err_trk_match + err_trk_eff*err_trk_eff + err_SPD_chip*err_SPD_chip + err_PileUp*err_PileUp + err_Lumi*err_Lumi);

        xsec_kpkm = (Nsig_data * fpileup) / (eff * lumi_kpkm);
        dxsec_kpkm_stat = xsec_kpkm * (dNsig_data / Nsig_data);
        double dxsec_phi2pi_sys = xsec_kpkm * TMath::Sqrt((err_sys*err_sys));

        // gdxsectiondy->SetPoint(i-1, pairRapidity, xsec_kpkm);
        // gdxsectiondy->SetPointError(i-1, 0, dxsec_kpkm_stat);

        // gdxsectiondy->SetPoint(i-1, pairRapidity, xsec_kpkm);
        // gdxsectiondy->SetPointError(i-1, 2, 0, 0, dxsec_kpkm_stat, dxsec_kpkm_stat, dxsec_phi2pi_sys, dxsec_phi2pi_sys);

        x[i-1]       = pairRapidity;
        y[i-1]       = xsec_kpkm;
        exl[i-1]     = hMassPairKaonVsRapiditySub_data->GetXaxis()->GetBinWidth(i)/2;
        exh[i-1]     = hMassPairKaonVsRapiditySub_data->GetXaxis()->GetBinWidth(i)/2;
        eylstat[i-1] = dxsec_kpkm_stat;
        eyhstat[i-1] = dxsec_kpkm_stat;
        eylsys[i-1]  = dxsec_phi2pi_sys;
        eyhsys[i-1]  = dxsec_phi2pi_sys;

        fprintf(tab_sys, "(%0.2f , %0.2f) & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f \\\\ \n", ymin, ymax, err_xsec_pt * 100, err_xsec_rad * 100, err_xsec_rejectPion * 100, err_xsec_rejectMuon * 100, err_xsec_rejectElectron * 100);

        // cout<<" i = "<<i<<" | pairRapidity = "<<pairRapidity<<" | xsec_kpkm = "<<xsec_kpkm<<endl;
        // printf(" i = %i | pairRapidity = %f | xsec_kpkm = %f", i, pairRapidity, xsec_kpkm);

        // printf (" ####### bins: %d, %d, %d, %d, %d, %d \n", hMassPairKaonVsRapiditySub_data->GetNbinsX(), hMassPairKaonVsRapidity_truth->GetNbinsX(), hMassPairKaonVsRapiditySub_mc->GetNbinsX(), hMassPairKaon_py_truth->GetNbinsX(), hMassPairKaon_py_mc->GetNbinsX(), hMassPairKaon_py_data->GetNbinsX());
        // printf("Nsig_truth = %f, Nsig_mc = %f, Nsig_data = %f, eff = %f, xsec_kpkm = %f \n", Nsig_truth, Nsig_mc, Nsig_data, eff, xsec_kpkm);
    }
    
    TCanvas *cdxsectiondy = new TCanvas("cdxsectiondy", "cdxsectiondy", 900, 600);
    cdxsectiondy->cd();

    // cdxsectiondy->SetLogy();
    // gdxsectiondy->Draw("AP");

    TGraphMultiErrors* gdxsectiondy = new TGraphMultiErrors("gdxsectiondy", "", np, x, y, exl, exh, eylstat, eyhstat);
    gdxsectiondy->AddYError(np, eylsys, eyhsys);
    gdxsectiondy->SetMarkerStyle(20);
    gdxsectiondy->SetMinimum(0.0);
    gdxsectiondy->SetTitle("; y; d\\sigma/dy (mb)");
    // gdxsectiondy->SetLineColor(kRed);
    // gdxsectiondy->GetAttLine(0)->SetLineColor(kRed);
    gdxsectiondy->GetAttLine(1)->SetLineColor(kGreen);
    gdxsectiondy->GetAttFill(1)->SetFillStyle(1);
    // Graph and x erros drawn with "APS"
    // Stat Errors drawn with "Z"
    // Sys Errors drawn with "5 s=0.5"
    gdxsectiondy->Draw("APS ; Z ; 5 s=1.0"); //APS ; Z ; 6 s=0.5
    cdxsectiondy->Update();

    gdxsectiondy->Write(); // Form("h%s_gdxsectiondy", name.Data()), TObject::kWriteDelete
    cdxsectiondy->Print(Form("%s/cdxsectiondy.root", outDir.Data()),"root");
    cdxsectiondy->Print(Form("%s/cdxsectiondy.pdf", outDir.Data()),"pdf");

    cMassPairKaon->Print(Form("%s/cMassPairKaon.root", outDir.Data()),"root");
    cMassPairKaon->Print(Form("%s/cMassPairKaon.pdf", outDir.Data()),"pdf");

    // cout<<"| hMassPairKaon_mc = "<<hMassPairKaon_mc->GetXaxis()->GetBinWidth(2)<<" | hMassPairKaon_RejectAllSub_mc = "<<hMassPairKaon_RejectAllSub_mc->GetXaxis()->GetBinWidth(2)<<endl;
    // cout<<"| hMassPairKaon_mc = "<<hMassPairKaon_mc->GetNbinsX()<<" | hMassPairKaon_RejectAllSub_mc = "<<hMassPairKaon_RejectAllSub_mc->GetNbinsX()<<endl;

    fprintf(tab_sys, "\\hline\n \\end{tabular}\n \\end{table}\n \\end{document}\n");
    fclose(tab_sys);
    gSystem->Exec(Form("pdflatex -output-directory %s %s/tab_sys.tex", outDir.Data(), outDir.Data()));

    printf(" ####### xsec = %0.3f, %0.3f, %0.3f \n", y[0], y[1], y[2]);
}