#include "TFile.h"
#include "TH1F.h"
#include "TF1.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TLatex.h"
#include "TText.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TEllipse.h"
#include "TMath.h"
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
#include <algorithm>
#include <vector>

using namespace std;

void Phi0Ana(TString dataORmc)
{
    // Open the file containing the tree.
    TFile *inputFile = NULL;
    if(dataORmc == "data") inputFile = TFile::Open("/Users/ahamdi/lbl/upc_loc/kpkm/output/data/350_20220224-1926/AnalysisResults_350.root"); // home/nacer = Users/ahamdi/lbl
    if(dataORmc == "mc") inputFile = TFile::Open("/Users/ahamdi/lbl/upc_loc/kpkm/output/mc/356_20220505-1003/AnalysisResults_MC_356.root"); // home/nacer = Users/ahamdi/lbl
    TString outDir = "/Users/ahamdi/lbl/upc_loc/Phi0Ana/output_Phi0Ana";

    TTree *inputTree = (TTree *)inputFile->Get("Central/Selected");
    if(inputTree == nullptr)
        cout<<"something went wrong! abort!"<<endl;

    // If list is valid, you can loop over its contents to make sure they are what you expect:
    // for(auto obj: *inputList)
    //     cout << "obj in list: " << obj->GetName() << endl;
    
    //Save histograms to output ROOT file
    TFile *outputFile = new TFile(Form("%s/Phi0Ana_%s.root",outDir.Data(), dataORmc.Data()), "RECREATE");

    // read histograms inside TList
    TH1F *fHistTriggersPerRun = (TH1F*)((TList*)inputFile->Get("Central/Histograms"))->FindObject("fHistTriggersPerRun");
    fHistTriggersPerRun->Write();

    // save results to latex/pdf table
    FILE *table_Phi0Ana = fopen(Form("%s/table_Phi0Ana_%s.tex",outDir.Data(), dataORmc.Data()), "w");
    fprintf(table_Phi0Ana, "\\documentclass[8pt]{extarticle}\n \\usepackage[margin=0.1in]{geometry}\n \\usepackage{tabularx}\n \\usepackage{caption} \n \\captionsetup{labelformat=empty}\n \\begin{document}\n \\begin{table}[!htbp]\n \\centering\n \\caption{$K^{+}K^{-}$ yields}\n \\begin{tabular}{|c|c|c|c|}\n \\hline\n Cuts & Signal & Bkg & Signal - Bkg \\\\ \n \\hline\n");

    gStyle->SetPalette(kRainBow); // choose different Palette color scheme
    // gStyle->SetOptStat("e"); // remove stat box
    gStyle->SetOptStat(0); // remove stat box 
    gStyle->SetOptFit(1);
    gStyle->SetTitleSize(.05, "xyz");
    gStyle->SetTitleOffset(1.4, "xy");
    gStyle->SetLabelSize(.05, "xyz");
    gStyle->SetPadTopMargin(0.12);
    gStyle->SetPadBottomMargin(0.15);
    gStyle->SetPadRightMargin(0.17); // 2d:0.17 , 1d:0.05
    gStyle->SetPadLeftMargin(0.16);  // 2d:0.16, 1d:0.15
    gStyle->SetLineWidth(2);
    gStyle->SetPadGridX(true);
    gStyle->SetPadGridY(true);

    // displace stat box
    // gStyle->SetOptFit();
    // c1->Modified(); c1->Update();
    // TPaveStats *stats =(TPaveStats*)c1->GetPrimitive("stats");
    // stats->SetName("h1stats");
    // stats->SetY1NDC(.4);
    // stats->SetY2NDC(.6);
    // stats->SetTextColor(2);

    // ------------------------------------ Invariant Mass ------------------------------------ 

    //**** Kaons
    auto hMassPairKaon = new TH1F("hMassPairKaon", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
    // 2 pions of different charge 
    auto hMassPairKaonUnLikeCharge = new TH1F("hMassPairKaonUnLikeCharge", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
    hMassPairKaonUnLikeCharge->SetLineColor(6);
    // 2 pions of same charge
    auto hMassPairKaonLikeCharge = new TH1F("hMassPairKaonLikeCharge", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
    hMassPairKaonLikeCharge->SetLineColor(7);
    // dE/dx cut
    auto hMassPairKaondEdxCut = new TH1F("hMassPairKaondEdxCut", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
    hMassPairKaondEdxCut->SetLineColor(28);
    // Pt cut < 0.1 GeV
    auto hMassPairKaonPtCut01 = new TH1F("hMassPairKaonPtCut01", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
    hMassPairKaonPtCut01->SetLineColor(1);

    // **** Pions
    auto hMassPairPion = new TH1F("hMassPairPion", "; m_{#pi^{+}#pi^{-}} (GeV/c^{2}); Counts", 300, 0.3, 1.4);
    // 2 pions of different charge 
    auto hMassPairPionUnLikeCharge = new TH1F("hMassPairPionUnLikeCharge", "; m_{#pi^{+}#pi^{-}} (GeV/c^{2}); Counts", 300, 0.3, 1.4);
    hMassPairPionUnLikeCharge->SetLineColor(6);
    // 2 pions of same charge
    auto hMassPairPionLikeCharge = new TH1F("hMassPairPionLikeCharge", "; m_{#pi^{+}#pi^{-}} (GeV/c^{2}); Counts", 300, 0.3, 1.4);
    hMassPairPionLikeCharge->SetLineColor(7);
    // dE/dx cut
    auto hMassPairPiondEdxCut = new TH1F("hMassPairPiondEdxCut", "; m_{#pi^{+}#pi^{-}} (GeV/c^{2}); Counts", 300, 0.3, 1.4);
    hMassPairPiondEdxCut->SetLineColor(28);
    // Pt cut < 0.1 GeV
    auto hMassPairPionPtCut01 = new TH1F("hMassPairPionPtCut01", "; m_{#pi^{+}#pi^{-}} (GeV/c^{2}); Counts", 300, 0.3, 1.4);
    hMassPairPionPtCut01->SetLineColor(1);
    // Pt cut > 0.1 GeV
    // auto hMassPairPionPtCut01bkg = new TH1F("hMassPairPionPtCut01bkg", "; m_{#pi^{+}#pi^{-}} (GeV/c^{2}); Counts", 300, 0.3, 1.4);
    // hMassPairPionPtCut01bkg->SetLineColor(2);
 
    // ------------- select kaon with other particles rejected 

    // s.t.d dE/dx cut to reject Pion
    auto hMassPairKaonRejectPion = new TH1F("hMassPairKaonRejectPion", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
    hMassPairKaonRejectPion->SetLineColor(kBlue);

    // s.t.d dE/dx cut to reject Pion + Muon
    auto hMassPairKaonRejectPionMuon = new TH1F("hMassPairKaonRejectPionMuon", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
    hMassPairKaonRejectPionMuon->SetLineColor(7);

    // s.t.d dE/dx cut to reject Pion + Muon + Electron
    auto hMassPairKaonRejectAll = new TH1F("hMassPairKaonRejectAll", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
    hMassPairKaonRejectAll->SetLineColor(28);

    // s.t.d dE/dx cut to reject Pion + Muon + Electron and keep kaon
    auto hMassPairKaon2Std = new TH1F("hMassPairKaon2Std", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
    hMassPairKaon2Std->SetLineColor(1);

    // ------------- Signal, Bkg and their subtraction

    // Select Kaon
    auto hMassPairKaonSig = new TH1F("hMassPairKaonSig", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonSig->SetLineColor(kBlue);
    auto hMassPairKaonBkg = new TH1F("hMassPairKaonBkg", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonBkg->SetLineColor(kRed);

    // Select Kaon & Reject Pion

    auto hMassPairKaonRejectPionSig = new TH1F("hMassPairKaonRejectPionSig", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonRejectPionSig->SetLineColor(kBlue);
    auto hMassPairKaonRejectPionBkg = new TH1F("hMassPairKaonRejectPionBkg", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonRejectPionBkg->SetLineColor(kRed);

    // Select Kaon & Reject (Pion + Muon)

    auto hMassPairKaonRejectPionMuonSig = new TH1F("hMassPairKaonRejectPionMuonSig", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonRejectPionMuonSig->SetLineColor(kBlue);
    auto hMassPairKaonRejectPionMuonBkg = new TH1F("hMassPairKaonRejectPionMuonBkg", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonRejectPionMuonBkg->SetLineColor(kRed);

    // Select Kaon & Reject (Pion + Muon + Electron)

    auto hMassPairKaonRejectAllSig = new TH1F("hMassPairKaonRejectAllSig", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonRejectAllSig->SetLineColor(kBlue);
    auto hMassPairKaonRejectAllBkg = new TH1F("hMassPairKaonRejectAllBkg", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonRejectAllBkg->SetLineColor(kRed);

    // -------------- Invariant Mass vs. Pair Rapidity --------------
    auto hMassPairKaonVsRapiditySig = new TH2D("hMassPairKaonVsRapiditySig", ";Counts;pair y;m_{K^{+}K^{-}} (GeV/c^{2})", 3,-0.8,0.8, 100, 0.95, 2.0);
    auto hMassPairKaonVsRapidityBkg = new TH2D("hMassPairKaonVsRapidityBkg", ";Counts;pair y;m_{K^{+}K^{-}} (GeV/c^{2})", 3,-0.8,0.8, 100, 0.95, 2.0);
    
    // ------------------------------------ Pair transverse momentum ------------------------------------ 

    auto hPtPair = new TH1F("hPtPair", "; pair p_{T} (GeV/c); Counts", 300, 0., 3.);
    // 2 tracks of different charge 
    auto hPtUnLikeCharge = new TH1F("hPtUnLikeCharge", "; pair p_{T} (GeV/c); Counts", 300, 0., 3.);
    hPtUnLikeCharge->SetLineColor(6);
    // 2 tracks of same charge
    auto hPtLikeCharge = new TH1F("hPtLikeCharge", "; pair p_{T} (GeV/c); Counts", 300, 0., 3.);
    hPtLikeCharge->SetLineColor(7);
    // Pion dE/dx cut
    auto hPtPiondEdxCut = new TH1F("hPtPiondEdxCut", "; pair p_{T} (GeV/c); Counts", 300, 0., 3.);
    hPtPiondEdxCut->SetLineColor(28);
    // Kaon dE/dx cut
    auto hPtKaondEdxCut = new TH1F("hPtKaondEdxCut", "; pair p_{T} (GeV/c); Counts", 300, 0.0, 3.0);
    hPtKaondEdxCut->SetLineColor(46);

    // ----------- Signal, Bkg and their subtraction

    // Select Kaon
    auto hPtPairKaonSig = new TH1F("hPtPairKaonSig", "; pair p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonSig->SetLineColor(kBlue);
    auto hPtPairKaonBkg = new TH1F("hPtPairKaonBkg", "; pair p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonBkg->SetLineColor(kRed);

    // Select Kaon & Reject Pion

    auto hPtPairKaonRejectPionSig = new TH1F("hPtPairKaonRejectPionSig", "; pair p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonRejectPionSig->SetLineColor(kBlue);
    auto hPtPairKaonRejectPionBkg = new TH1F("hPtPairKaonRejectPionBkg", "; pair p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonRejectPionBkg->SetLineColor(kRed);

    // Select Kaon & Reject (Pion + Muon)

    auto hPtPairKaonRejectPionMuonSig = new TH1F("hPtPairKaonRejectPionMuonSig", "; pair p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonRejectPionMuonSig->SetLineColor(kBlue);
    auto hPtPairKaonRejectPionMuonBkg = new TH1F("hPtPairKaonRejectPionMuonBkg", "; pair p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonRejectPionMuonBkg->SetLineColor(kRed);

    // Select Kaon & Reject (Pion + Muon + Electron)

    auto hPtPairKaonRejectAllSig = new TH1F("hPtPairKaonRejectAllSig", "; pair p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonRejectAllSig->SetLineColor(kBlue);
    auto hPtPairKaonRejectAllBkg = new TH1F("hPtPairKaonRejectAllBkg", "; pair p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonRejectAllBkg->SetLineColor(kRed);

    // --------------------------------------------------------------------------------------------------

    // -------------- dE/dx sigma vs. track momentum
    auto hDEdxVsP = new TH2D("hDEdxVsP", "; p(GeV/c); dE/dx", 1000,0.,5., 1000,0.,600.);
    auto hStdDEdxVsPKaon = new TH2D("hStdDEdxVsPKaon", "; p(GeV/c); #sigma^{K^{#pm}}_{dE/dx};Counts", 1000,0.,5., 1000,-20.,50.);
    auto hStdDEdxVsPPion = new TH2D("hStdDEdxVsPPion", "; p(GeV/c); #sigma^{#pi^{#pm}}_{dE/dx};Counts", 1000,0.,5., 1000,-10.,50.);
    auto hStdDEdxVsPMuon = new TH2D("hStdDEdxVsPMuon", "; p(GeV/c); #sigma^{#mu^{#pm}}_{dE/dx};Counts", 1000,0.,5., 1000,-10.,50.);
    auto hStdDEdxVsPElectron = new TH2D("hStdDEdxVsPElectron", "; p(GeV/c); #sigma^{e^{#pm}}_{dE/dx};Counts", 1000,0.,5., 1000,-10.,50.);

    //------ dE/dx sigma 1 vs. dE/dx sigma 2

    // Kaons
    auto hStdDEdx2Kaon = new TH2D("hStdDEdx2Kaon", ";#sigma^{K^{1}}_{dE/dx}; #sigma^{K^{2}}_{dE/dx};Counts", 1000,-20.,10., 1000,-20.,10.);

    // reject Pion
    auto hStdDEdx2KaonRejectPion = new TH2D("hStdDEdx2KaonRejectPion", ";#sigma^{K^{1}}_{dE/dx}; #sigma^{K^{2}}_{dE/dx};Counts", 1000,-20.,10., 1000,-20.,10.);

    // reject Pion + Muon
    auto hStdDEdx2KaonRejectPionMuon = new TH2D("hStdDEdx2KaonRejectPionMuon", ";#sigma^{K^{1}}_{dE/dx}; #sigma^{K^{2}}_{dE/dx};Counts", 1000,-20.,10., 1000,-20.,10.); // 1000,-30.,50., 1000,-30.,50.

    //reject Pion + Muon + Electron
    auto hStdDEdx2KaonRejectAll = new TH2D("hStdDEdx2KaonRejectAll", ";#sigma^{K^{1}}_{dE/dx}; #sigma^{K^{2}}_{dE/dx};Counts", 1000,-20.,10., 1000,-20.,10.);

    // reject Pion + Muon + Electron while keeping kaon
    auto hStdDEdx2Kaon2Std = new TH2D("hStdDEdx2Kaon2Std", ";#sigma^{K^{1}}_{dE/dx}; #sigma^{K^{2}}_{dE/dx};Counts", 1000,-20.,10., 1000,-20.,10.);

    // ------ dE/dx sigma in different pT and Eta regions
    // Pt
    auto hStdDEdxKaon_Pt025 = new TH1F("hStdDEdxKaon_Pt025", ";#sigma^{K^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    hStdDEdxKaon_Pt025->SetLineColor(1);
    auto hStdDEdxKaon_Pt050 = new TH1F("hStdDEdxKaon_Pt050", ";#sigma^{K^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    hStdDEdxKaon_Pt050->SetLineColor(2);
    auto hStdDEdxKaon_Pt075 = new TH1F("hStdDEdxKaon_Pt075", ";#sigma^{K^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    hStdDEdxKaon_Pt075->SetLineColor(4);
    auto hStdDEdxKaon_Pt100 = new TH1F("hStdDEdxKaon_Pt100", ";#sigma^{K^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    hStdDEdxKaon_Pt100->SetLineColor(6);

    // Eta
    auto hStdDEdxKaon_Eta025 = new TH1F("hStdDEdxKaon_Eta025", ";#sigma^{K^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    hStdDEdxKaon_Eta025->SetLineColor(1);
    auto hStdDEdxKaon_Eta050 = new TH1F("hStdDEdxKaon_Eta050", ";#sigma^{K^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    hStdDEdxKaon_Eta050->SetLineColor(2);
    auto hStdDEdxKaon_Eta075 = new TH1F("hStdDEdxKaon_Eta075", ";#sigma^{K^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    hStdDEdxKaon_Eta075->SetLineColor(4);
    auto hStdDEdxKaon_Eta100 = new TH1F("hStdDEdxKaon_Eta100", ";#sigma^{K^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    hStdDEdxKaon_Eta100->SetLineColor(6);

    // Kaon
    auto hStdDEdxKaon = new TH1F("hStdDEdxKaon", ";#sigma^{K^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    // KaonNotSelected
    auto hStdDEdxKaonNotSelected = new TH1F("hStdDEdxKaonNotSelected", ";#sigma^{K^{1 Track}}_{dE/dx}; Counts", 200,-30.,30.);
    // Pion
    auto hStdDEdxPion = new TH1F("hStdDEdxPion", ";#sigma^{#pi^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    // Muon
    auto hStdDEdxMuon = new TH1F("hStdDEdxMuon", ";#sigma^{#mu^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);
    // Electron
    auto hStdDEdxElectron = new TH1F("hStdDEdxElectron", ";#sigma^{e^{#pm}}_{dE/dx}; Counts", 500,-30.,30.);

    // -------------- Track PseudoRapidity --------------
    auto hTrack1vs2PseudoRapidity = new TH2D("hTrack1vs2PseudoRapidity", ";#eta_{1};#eta_{2};Counts", 100,-1.,1., 100,-1.,1.);

    // -------------- Track Azimuthal Angle --------------
    auto hTrackAzimuthal = new TH1F("hTrackAzimuthal", ";#phi (rad);Counts", 300,-TMath::Pi(), TMath::Pi());

    // -------------- Pair Azimuthal Angle --------------
    auto hAzimuthal = new TH1F("hAzimuthal", ";#phi (rad);Counts", 300,-TMath::Pi(), TMath::Pi());

    // -------------- Primary Vertex Z --------------
    auto hVertexZ = new TH1F("hVertexZ", ";Vertex z-axis (cm);Counts", 300,-20., 20);

    // -------------- Pair Rapidity --------------
    auto hRapidity = new TH1F("hRapidity", ";pair y;Counts", 300,-1.,1.);
   
    // --------- Count sig and Bkg events after different PID cuts
    Int_t NumbKaonSig1 = 0, NumbKaonSig2 = 0, NumbKaonSig3 = 0, NumbKaonSig4 = 0;
    Int_t NumbKaonBkg1 = 0, NumbKaonBkg2 = 0, NumbKaonBkg3 = 0, NumbKaonBkg4 = 0;
    Int_t NumbKaon1PID = 0, NumbKaon2PID = 0;

    // -------- Events per run
    auto hEvents = new TH1F("hEvents","hEvents", 50000, 240000.5, 290000.5); //244975 - 246994 (83 runs)

    // get the variables inside input TTree branches
    const Int_t Maxtrk = 7;
    Int_t nTrack_T;
    Int_t RunNum_T;
    Bool_t ITSSA_T[Maxtrk];
    Bool_t TPCrefit_T[Maxtrk] = {0};
    Bool_t ITSrefit_T[Maxtrk] = {0};
    Int_t TPCNcls_T[Maxtrk];
    Int_t ITSNcls_T[Maxtrk];
    Bool_t HasPointOnITSLayer_0_T[Maxtrk];
	Bool_t HasPointOnITSLayer_1_T[Maxtrk];
    Float_t dca_0_T[Maxtrk];
    Float_t dca_1_T[Maxtrk];
    Int_t TrackQ_T[Maxtrk];
    Float_t Pt_T = -999;
	Float_t Rapidity_T = -999;
	Float_t Phi_T = -999;
	Bool_t LikeSign_T = -999;
    Float_t TrackPx_T[Maxtrk];
    Float_t TrackPy_T[Maxtrk];
    Float_t TrackPz_T[Maxtrk];
    Float_t TrackP_T[Maxtrk];
    Float_t TrackPt_T[Maxtrk];
    Float_t TrackEta_T[Maxtrk];
    Float_t TrackPhi_T[Maxtrk];
    Int_t TPCsignal_T[Maxtrk];
    Float_t PIDTPCKaon_T[Maxtrk];
    Float_t PIDTPCPion_T[Maxtrk];
    Float_t PIDTPCMuon_T[Maxtrk];
    Float_t PIDTPCElectron_T[Maxtrk];
    Float_t VtxZ_T;
    
    inputTree->SetBranchAddress("nTrack_T", &nTrack_T);
    inputTree->SetBranchAddress("RunNum_T", &RunNum_T);
    inputTree->SetBranchAddress("ITSSA_T", ITSSA_T);
    inputTree->SetBranchAddress("TPCrefit_T", TPCrefit_T);
    inputTree->SetBranchAddress("ITSrefit_T", ITSrefit_T);
    inputTree->SetBranchAddress("TPCNcls_T", TPCNcls_T);
    inputTree->SetBranchAddress("ITSNcls_T", ITSNcls_T);
    inputTree->SetBranchAddress("HasPointOnITSLayer_0_T", HasPointOnITSLayer_0_T);
    inputTree->SetBranchAddress("HasPointOnITSLayer_1_T", HasPointOnITSLayer_1_T);
    inputTree->SetBranchAddress("dca_0_T", dca_0_T);
    inputTree->SetBranchAddress("dca_1_T", dca_1_T);
    inputTree->SetBranchAddress("TrackQ_T", TrackQ_T);
    inputTree->SetBranchAddress("Pt_T", &Pt_T);
    inputTree->SetBranchAddress("Rapidity_T", &Rapidity_T);
    inputTree->SetBranchAddress("Phi_T", &Phi_T);
    inputTree->SetBranchAddress("VtxZ_T", &VtxZ_T);
    inputTree->SetBranchAddress("LikeSign_T", &LikeSign_T);
    inputTree->SetBranchAddress("TrackPx_T", TrackPx_T);
    inputTree->SetBranchAddress("TrackPy_T", TrackPy_T);
    inputTree->SetBranchAddress("TrackPz_T", TrackPz_T);
    inputTree->SetBranchAddress("TrackP_T", TrackP_T);
    inputTree->SetBranchAddress("TrackPt_T", TrackPt_T);
    inputTree->SetBranchAddress("TrackEta_T", TrackEta_T);
    inputTree->SetBranchAddress("TrackPhi_T", TrackPhi_T);
    inputTree->SetBranchAddress("TPCsignal_T", TPCsignal_T);
    inputTree->SetBranchAddress("PIDTPCKaon_T", PIDTPCKaon_T);
    inputTree->SetBranchAddress("PIDTPCPion_T", PIDTPCPion_T);
    inputTree->SetBranchAddress("PIDTPCMuon_T", PIDTPCMuon_T);
    inputTree->SetBranchAddress("PIDTPCElectron_T", PIDTPCElectron_T);

    // grab the pion mass from PDG
    TDatabasePDG *pdgdat = TDatabasePDG::Instance(); 
	TParticlePDG *partKaon = pdgdat->GetParticle(321); // kaon
  	TParticlePDG *partPion = pdgdat->GetParticle(211); // pion
  	TParticlePDG *partMuon = pdgdat->GetParticle(13); // Muon
  	TParticlePDG *partElectron = pdgdat->GetParticle(11); // electron
  	Double_t kaonMass = partKaon->Mass();
  	Double_t pionMass = partPion->Mass();
  	Double_t muonMass = partMuon->Mass();
  	Double_t electronMass = partElectron->Mass();

    TLorentzVector lvKaon[2];
    TLorentzVector lvPion[2];
    TLorentzVector lvMuon[2];
    TLorentzVector lvElectron[2];
    TLorentzVector lvSumKaon;
    TLorentzVector lvSumPion; // pair-4vector
    TLorentzVector lvSumMuon;
    TLorentzVector lvSumElectron;

	Bool_t likeSign = -999;
    Int_t TrackQ[2];
    Float_t PIDTPCKaon[2];
    Float_t PIDTPCPion[2];
    Float_t PIDTPCMuon[2];
    Float_t PIDTPCElectron[2];
    Float_t TrackEta[2];
    Float_t TrackPhi[2];

    // if center of dEdx s.t.d is shifted in MC
    double x0 = 0;
    double y0 = 0;

    if(dataORmc == "data")
    {
        x0 = -0.23; // 1.82
        y0 = -0.36; // 1.7
    }
    if(dataORmc == "mc")
    {
        x0 = 0.0; // 1.82
        y0 = 0.0; // 1.62
    }

    list<int> runList_Rho0 = {244975, 244980, 244982, 244983, 245064, 245066, 245068, 245683, 245692, 245702, 245705, 245829, 245831, 245833, 245923, 245949, 245952, 245954, 245963, 246001, 246003, 246012, 246036, 246037, 246042, 246048, 246049, 246052, 246053, 246087, 246089, 246113, 246115, 246148, 246151, 246152, 246153, 246178, 246180, 246181, 246182, 246185, 246217, 246222, 246225, 246271, 246272, 246275, 246276, 246424, 246431, 246434, 246487, 246488, 246493, 246495, 246750, 246751, 246757, 246758, 246759, 246760, 246763, 246765, 246766, 246804, 246805, 246807, 246808, 246809, 246810, 246844, 246845, 246846, 246847, 246851, 246945, 246948, 246982, 246984, 246989, 246991, 246994};

    list<int> runList_Phi0 = {244917, 244918, 244975, 244980, 244982, 244983, 245061, 245064, 245066, 245068, 246390, 246391, 246392, 245145, 245146, 245148, 245151, 245152, 245231, 245232, 245233, 245259, 245343, 245345, 245346, 245347, 245349, 245353, 245396, 245397, 245401, 245407, 245409, 245410, 245411, 245439, 245441, 245446, 245450, 245452, 245453, 245454, 245496, 245497, 245501, 245504, 245505, 245507, 245535, 245540, 245542, 245543, 245544, 245545, 245554, 245683, 245692, 245700, 245702, 245705, 245729, 245731, 245738, 245752, 245759, 245766, 245775, 245785, 245793, 245829, 245831, 245833, 245923, 245949, 245952, 245954, 245963, 246001, 246003, 246012, 246036, 246037, 246042, 246048, 246049, 246052, 246053, 246087, 246089, 246113, 246115, 246148, 246151, 246152, 246153, 246178, 246180, 246181, 246182, 246185, 246217, 246222, 246225, 246271, 246272, 246275, 246276, 246424, 246428, 246431, 246434, 246487, 246488, 246493, 246495, 246540, 246543, 246553, 246567, 246568, 246575, 246583, 246648, 246671, 246675, 246676, 246750, 246751, 246757, 246758, 246759, 246760, 246763, 246765, 246766, 246804, 246805, 246807, 246808, 246809, 246810, 246844, 246845, 246846, 246847, 246851, 246855, 246858, 246859, 246864, 246865, 246867, 246870, 246871, 246928, 246930, 246937, 246942, 246945, 246948, 246949, 246980, 246982, 246984, 246989, 246991, 246994};

    list<int> runList_diff = {244917, 244918, 245061, 246390, 246391, 246392, 245145, 245146, 245148, 245151, 245152, 245231, 245232, 245233, 245259, 245343, 245345, 245346, 245347, 245349, 245353, 245396, 245397, 245401, 245407, 245409, 245410, 245411, 245439, 245441, 245446, 245450, 245452, 245453, 245454, 245496, 245497, 245501, 245504, 245505, 245507, 245535, 245540, 245542, 245543, 245544, 245545, 245554, 245700, 245729, 245731, 245738, 245752, 245759, 245766, 245775, 245785, 245793, 246428, 246540, 246543, 246553, 246567, 246568, 246575, 246583, 246648, 246671, 246675, 246676, 246855, 246858, 246859, 246864, 246865, 246867, 246870, 246871, 246928, 246930, 246937, 246942, 246949, 246980};

    bool isValidatedRun = false;

    // loop over all the events
    for (int i = 0; i < inputTree->GetEntries(); i++)
    {
        inputTree->GetEntry(i);

        hEvents->Fill(RunNum_T); // number of events per run
         
        // Long64_t nentries_mc = inputTree->GetEntries();
        // cout<<" nentries_mc = "<<nentries_mc<<endl;

        // skip runs that are not used in Rho analysis
        isValidatedRun = (std::find(runList_Rho0.begin(), runList_Rho0.end(), RunNum_T) != runList_Rho0.end());
        if(!isValidatedRun) continue;

        // if(i%100000==0) cout<<"event = "<<i<<" | RunNum_T = "<<RunNum_T<<endl;

        // hEvents->Fill(RunNum_T); // number of events per run

        Int_t nGoodTracks=0;
        Int_t TrackIndex[Maxtrk] = {-1};

        // loop over all tracks
        for(Int_t iTrack=0; iTrack<nTrack_T; iTrack++)
        {
            // keep tracks with TPC and ITS-only tracks as well
            if(ITSSA_T[iTrack]) continue; // to remove ITS standalone tracks
            if(! TPCrefit_T[iTrack]) continue;
            if(! ITSrefit_T[iTrack]) continue;
            if(TPCNcls_T[iTrack] < 50) continue; // to remove TPC only tracks
            // if(ITSNcls_T[iTrack] < 3) continue; // to remove TPC only tracks
            if(!(HasPointOnITSLayer_0_T[iTrack] && HasPointOnITSLayer_1_T[iTrack])) continue;
            if(TMath::Abs(dca_1_T[iTrack]) > 2) continue;
            Double_t cut_DCAxy = (0.0182 + 0.0350/TMath::Power(TrackPt_T[iTrack],1.01));
            if(TMath::Abs(dca_0_T[iTrack]) > cut_DCAxy) continue;

	        // store good track index
	        TrackIndex[nGoodTracks] = iTrack;
            nGoodTracks++;

            // cout<<"################  NO PROBLEM UP TO HERE !"<<endl;
            //if(i%100000==0) cout<<"event = "<<i<<" | nTrack_T = "<<nTrack_T<<" | nGoodTracks = "<<nGoodTracks<<" | TrackIndex["<<nGoodTracks-1<<"] = "<<TrackIndex[nGoodTracks-1]<<endl;
        }

        if(nGoodTracks == 2) // loop over two good tracks
        {
            // Float_t PIDTPCKaonPlus = -30.0, PIDTPCKaonMinus = 30.0;

            for(Int_t iTrack=0; iTrack<nGoodTracks; iTrack++)
            {
                // creat 4-momenta using Pt, Eta, Phi, Mass per track
                Int_t itr = TrackIndex[iTrack];
                lvKaon[iTrack].SetPtEtaPhiM(TrackPt_T[itr], TrackEta_T[itr], TrackPhi_T[itr], kaonMass);
                lvPion[iTrack].SetPtEtaPhiM(TrackPt_T[itr], TrackEta_T[itr], TrackPhi_T[itr], pionMass);
                lvMuon[iTrack].SetPtEtaPhiM(TrackPt_T[itr], TrackEta_T[itr], TrackPhi_T[itr], muonMass);
                lvElectron[iTrack].SetPtEtaPhiM(TrackPt_T[itr], TrackEta_T[itr], TrackPhi_T[itr], electronMass);

                TrackQ[iTrack] = TrackQ_T[itr];
                PIDTPCKaon[iTrack] = PIDTPCKaon_T[itr];
                PIDTPCPion[iTrack] = PIDTPCPion_T[itr];
                PIDTPCMuon[iTrack] = PIDTPCMuon_T[itr];
                PIDTPCElectron[iTrack] = PIDTPCElectron_T[itr];
                TrackEta[iTrack] = TrackEta_T[itr];
                TrackPhi[iTrack] = TrackPhi_T[itr];

                // -------------- S.t.d dE/dx vs. track momentum
                hDEdxVsP->Fill(TrackP_T[itr], TPCsignal_T[itr]);
                hStdDEdxVsPKaon->Fill(TrackP_T[itr], PIDTPCKaon[iTrack]);
                hStdDEdxVsPPion->Fill(TrackP_T[itr], PIDTPCPion[iTrack]);
                hStdDEdxVsPMuon->Fill(TrackP_T[itr], PIDTPCMuon[iTrack]);
                hStdDEdxVsPElectron->Fill(TrackP_T[itr], PIDTPCElectron[iTrack]);

                hTrackAzimuthal->Fill(TrackPhi[iTrack]);

                // if(TrackQ[iTrack] == 1) PIDTPCKaonPlus = PIDTPCKaon[iTrack];
                // if(TrackQ[iTrack] == -1) PIDTPCKaonMinus = PIDTPCKaon[iTrack];

                // if(i%10000==0 && TrackQ[iTrack] == 1) cout<<"event = "<<i<<" | TrackQ["<<iTrack<<"] = "<<TrackQ[iTrack]<<" | PIDTPCKaon["<<iTrack<<"] = "<<PIDTPCKaon[iTrack]<<" | PIDTPCKaonPlus = "<<PIDTPCKaonPlus<<" | PIDTPCKaonMinus = "<<PIDTPCKaonMinus<<endl;
            }

	        lvSumKaon = lvKaon[0] + lvKaon[1];
            lvSumPion = lvPion[0] + lvPion[1];
            lvSumMuon = lvMuon[0] + lvMuon[1];
	        lvSumElectron = lvElectron[0] + lvElectron[1];

            if (TrackQ[0] * TrackQ[1] > 0)
                likeSign = 1;
            else
                likeSign = 0;

            // -------- Invariant Mass
            
            hMassPairKaon->Fill(lvSumKaon.M()); // pair kaons invariant mass
            hMassPairPion->Fill(lvSumPion.M()); // pair pions invariant mass
            hPtPair->Fill(lvSumKaon.Pt()); // Pair transverse momentum 

            if(likeSign == 1)
            {
                hMassPairKaonLikeCharge->Fill(lvSumKaon.M()); // pair kaons invariant mass
                hMassPairPionLikeCharge->Fill(lvSumPion.M()); // pair pions invariant mass
                hPtLikeCharge->Fill(lvSumKaon.Pt()); // Pair transverse momentum 
            }               

            if(likeSign == 1) continue; //save only events with opposite charge track pairs

            hMassPairKaonUnLikeCharge->Fill(lvSumKaon.M()); // pair kaons invariant mass
            hMassPairPionUnLikeCharge->Fill(lvSumPion.M()); // pair pions invariant mass
            hPtUnLikeCharge->Fill(lvSumKaon.Pt()); // Pair transverse momentum

            // Kinematics
            hTrack1vs2PseudoRapidity->Fill(TrackEta[0], TrackEta[1]);
            hAzimuthal->Fill(lvSumKaon.Phi());
            hVertexZ->Fill(VtxZ_T);
            hRapidity->Fill(lvSumKaon.Rapidity());

            // pair kaons invariant mass
            if(abs(PIDTPCKaon[0]) < 3 && abs(PIDTPCKaon[1]) < 3)
            {
                hMassPairKaondEdxCut->Fill(lvSumKaon.M());
                hPtKaondEdxCut->Fill(lvSumKaon.Pt()); // Pair transverse monemtum

                if(lvSumKaon.Pt() < 0.1)
                {
                    hMassPairKaonPtCut01->Fill(lvSumKaon.M());
                }
            }

            // pair pions invariant mass
            if(abs(PIDTPCPion[0]) < 3 && abs(PIDTPCPion[1]) < 3)
            {
                hMassPairPiondEdxCut->Fill(lvSumPion.M());
                hPtPiondEdxCut->Fill(lvSumPion.Pt()); // Pair transverse monemtum

                if(lvSumPion.Pt() < 0.1)
                {
                    hMassPairPionPtCut01->Fill(lvSumPion.M());
                }
            }

            // ------ select kaons with other particles rejected

            if(abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2)
            {
                hMassPairKaonRejectPion->Fill(lvSumKaon.M());

                if(abs(PIDTPCMuon[0]) > 2 && abs(PIDTPCMuon[1]) > 2)
                {
                    hMassPairKaonRejectPionMuon->Fill(lvSumKaon.M());

                    if(abs(PIDTPCElectron[0]) > 2 && abs(PIDTPCElectron[1]) > 2)
                    {
                        hMassPairKaonRejectAll->Fill(lvSumKaon.M());

                        if(abs(PIDTPCKaon[0]) < 2 && abs(PIDTPCKaon[1]) < 2)
                        {
                            hMassPairKaon2Std->Fill(lvSumKaon.M());
                        }
                    }
                }
            }

            // ------ dE/dx sigma 1 in different pT and Eta regions

            for(Int_t iTrack=0; iTrack<nGoodTracks; iTrack++)
            {
                // -------------- S.t.d dE/dx in fifferet Pt and Eta regions
                if(lvSumKaon.Pt() < 0.25) hStdDEdxKaon_Pt025->Fill(PIDTPCKaon[iTrack]);
                if(lvSumKaon.Pt() > 0.25 && lvSumKaon.Pt() < 0.50) hStdDEdxKaon_Pt050->Fill(PIDTPCKaon[iTrack]);
                if(lvSumKaon.Pt() > 0.50 && lvSumKaon.Pt() < 0.75) hStdDEdxKaon_Pt075->Fill(PIDTPCKaon[iTrack]);
                if(lvSumKaon.Pt() > 0.75 && lvSumKaon.Pt() < 1.0) hStdDEdxKaon_Pt100->Fill(PIDTPCKaon[iTrack]);

                if(abs(lvSumKaon.Eta()) < 0.25) hStdDEdxKaon_Eta025->Fill(PIDTPCKaon[iTrack]);
                if(abs(lvSumKaon.Eta()) < 0.50) hStdDEdxKaon_Eta050->Fill(PIDTPCKaon[iTrack]);
                if(abs(lvSumKaon.Eta()) < 0.75) hStdDEdxKaon_Eta075->Fill(PIDTPCKaon[iTrack]);
                if(abs(lvSumKaon.Eta()) < 1.0) hStdDEdxKaon_Eta100->Fill(PIDTPCKaon[iTrack]);

                hStdDEdxKaon->Fill(PIDTPCKaon[iTrack]);
                hStdDEdxPion->Fill(PIDTPCPion[iTrack]);
                hStdDEdxMuon->Fill(PIDTPCMuon[iTrack]);
                hStdDEdxElectron->Fill(PIDTPCElectron[iTrack]);
            }

            if(abs(PIDTPCKaon[0]) < 3) hStdDEdxKaonNotSelected->Fill(PIDTPCKaon[1]);
            // if(abs(PIDTPCKaonPlus) < 3) hStdDEdxKaonNotSelected->Fill(PIDTPCKaonMinus);

            // --------- Count sig and Bkg events after different PID cuts

            if(lvSumKaon.Pt() > 0.1) continue; // 0.1, 0.12, 0.08

            // ------ dE/dx sigma 1 vs. dE/dx sigma 2
            hStdDEdx2Kaon->Fill(PIDTPCKaon[0], PIDTPCKaon[1]);
            if (abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2)
            {
                hStdDEdx2KaonRejectPion->Fill(PIDTPCKaon[0], PIDTPCKaon[1]);
            }
            if (abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2 && abs(PIDTPCMuon[0]) > 2 && abs(PIDTPCMuon[1]) > 2)
            {
                hStdDEdx2KaonRejectPionMuon->Fill(PIDTPCKaon[0], PIDTPCKaon[1]);
            }
            if (abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2 && abs(PIDTPCMuon[0]) > 2 && abs(PIDTPCMuon[1]) > 2 && abs(PIDTPCElectron[0]) > 2 && abs(PIDTPCElectron[1]) > 2)
            {
                hStdDEdx2KaonRejectAll->Fill(PIDTPCKaon[0], PIDTPCKaon[1]);
            }
            if (abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2 && abs(PIDTPCMuon[0]) > 2 && abs(PIDTPCMuon[1]) > 2 && abs(PIDTPCElectron[0]) > 2 && abs(PIDTPCElectron[1]) > 2 && abs(PIDTPCKaon[0]) < 2 && abs(PIDTPCKaon[1]) < 2)
            {
                hStdDEdx2Kaon2Std->Fill(PIDTPCKaon[0], PIDTPCKaon[1]);
            }

            // ------ Sig events
            if((TMath::Power((PIDTPCKaon[0] - x0), 2) + TMath::Power((PIDTPCKaon[1] - y0), 2)) < 9) // 9, 12.25, 6.25 
            {
                NumbKaonSig1++;
                hMassPairKaonSig->Fill(lvSumKaon.M());
                hPtPairKaonSig->Fill(lvSumKaon.Pt());
                
                if(abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2)
                {
                    NumbKaonSig2++;
                    hMassPairKaonRejectPionSig->Fill(lvSumKaon.M());
                    hPtPairKaonRejectPionSig->Fill(lvSumKaon.Pt());
                }
                if(abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2 && abs(PIDTPCMuon[0]) > 2 && abs(PIDTPCMuon[1]) > 2)
                {
                    NumbKaonSig3++;
                    hMassPairKaonRejectPionMuonSig->Fill(lvSumKaon.M());
                    hPtPairKaonRejectPionMuonSig->Fill(lvSumKaon.Pt());
                }
                if(abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2 && abs(PIDTPCMuon[0]) > 2 && abs(PIDTPCMuon[1]) > 2 && abs(PIDTPCElectron[0]) > 2 && abs(PIDTPCElectron[1]) > 2) // 2, 2.5, 1.5
                {
                    NumbKaonSig4++;
                    hMassPairKaonRejectAllSig->Fill(lvSumKaon.M());
                    hPtPairKaonRejectAllSig->Fill(lvSumKaon.Pt());
                    hMassPairKaonVsRapiditySig->Fill(lvSumKaon.Rapidity(), lvSumKaon.M());
                }
            }

            // ------ Bkg events
            if((TMath::Power((PIDTPCKaon[0] - x0), 2) + TMath::Power((PIDTPCKaon[1] - y0), 2)) > 16 && (TMath::Power((PIDTPCKaon[0] - x0), 2) + TMath::Power((PIDTPCKaon[1] - y0), 2)) < 25) // (16, 25), (16, 28.25), (16, 22.25)
            {
                hMassPairKaonBkg->Fill(lvSumKaon.M());
                hPtPairKaonBkg->Fill(lvSumKaon.Pt());
                NumbKaonBkg1++;

                if(abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2)
                {
                    NumbKaonBkg2++;
                    hMassPairKaonRejectPionBkg->Fill(lvSumKaon.M());
                    hPtPairKaonRejectPionBkg->Fill(lvSumKaon.Pt());
                }
                if(abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2 && abs(PIDTPCMuon[0]) > 2 && abs(PIDTPCMuon[1]) > 2)
                {
                    NumbKaonBkg3++;
                    hMassPairKaonRejectPionMuonBkg->Fill(lvSumKaon.M());
                    hPtPairKaonRejectPionMuonBkg->Fill(lvSumKaon.Pt());
                }
                if(abs(PIDTPCPion[0]) > 2 && abs(PIDTPCPion[1]) > 2 && abs(PIDTPCMuon[0]) > 2 && abs(PIDTPCMuon[1]) > 2 && abs(PIDTPCElectron[0]) > 2 && abs(PIDTPCElectron[1]) > 2) // 2, 2.5, 1.5
                {
                    NumbKaonBkg4++;
                    hMassPairKaonRejectAllBkg->Fill(lvSumKaon.M());
                    hPtPairKaonRejectAllBkg->Fill(lvSumKaon.Pt());
                    hMassPairKaonVsRapidityBkg->Fill(lvSumKaon.Rapidity(), lvSumKaon.M());
                }
            }

            if(abs(PIDTPCKaon[0]) < 3 && abs(PIDTPCKaon[1]) < 3)
            {
                NumbKaon2PID++;
            }
            if(abs(PIDTPCKaon[0]) < 3 || abs(PIDTPCKaon[1]) < 3)
            {
                NumbKaon1PID++;
            }
        }
    }

    //  // --------------- Generated
    
    if(dataORmc == "mc")
    {
        TTree *inputTree_truth = (TTree *)inputFile->Get("Central/Generated");
        if(inputTree_truth == nullptr)
            cout<<"something went wrong! abort!"<<endl;

        // Long64_t nentries_truth = inputTree_truth->GetEntries();
        // cout<<" nentries_truth = "<<nentries_truth<<endl;

        // Pair Invariant Mass
        auto hMassPairKaon_truth = new TH1F("hMassPairKaon_truth", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
        // Pair Transverse Momentum
        auto hPtPair_truth = new TH1F("hPtPair_truth", "; p_{T} (GeV/c); Counts", 300, 0., 3.);
        // Invariant Mass vs. Pair Rapidity --------------
        auto hMassPairKaonVsRapidity_truth = new TH2D("hMassPairKaonVsRapidity_truth", ";pair y;m_{K^{+}K^{-}} (GeV/c^{2});Counts", 3,-0.8,0.8, 100, 0.95, 2.0);

        Float_t Mass_MC_T;
        Float_t Pt_MC_T;
        Float_t Rapidity_MC_T;
        Float_t Phi_MC_T;
        Int_t RunNum_MC_T;

        inputTree_truth->SetBranchAddress("Mass_MC_T", &Mass_MC_T);
        inputTree_truth->SetBranchAddress("Pt_MC_T", &Pt_MC_T);
        inputTree_truth->SetBranchAddress("Rapidity_MC_T", &Rapidity_MC_T);
        inputTree_truth->SetBranchAddress("Phi_MC_T", &Phi_MC_T);
        inputTree_truth->SetBranchAddress("RunNum_MC_T", &RunNum_MC_T);

        for (int i = 0; i < inputTree_truth->GetEntries(); i++)
        {
            inputTree_truth->GetEntry(i);

            // if(i%100000==0) cout<<"------ Before : event = "<<i<<" | RunNum_MC_T = "<<RunNum_MC_T<<endl;

            hEvents->Fill(RunNum_MC_T);

            // skip runs that are not used in Rho analysis
            isValidatedRun = (std::find(runList_Rho0.begin(), runList_Rho0.end(), RunNum_MC_T) != runList_Rho0.end());
            if(!isValidatedRun) continue;

            hMassPairKaon_truth->Fill(Mass_MC_T);
            hPtPair_truth->Fill(Pt_MC_T);
            hMassPairKaonVsRapidity_truth->Fill(Rapidity_MC_T, Mass_MC_T);
        }
    
        // Pair Invariant Mass
        auto cMassPairKaon_truth = new TCanvas("cMassPairKaon_truth", "cMassPairKaon_truth", 600, 400);
        cMassPairKaon_truth->cd();
        // auto hMassPairKaon_truth = new TH1F("hMassPairKaon_truth", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 300, 0.95, 2.0);
        // inputTree_truth->Project("hMassPairKaon_truth", "Mass_MC_T");
        hMassPairKaon_truth->Draw();
        hMassPairKaon_truth->Write();
        cMassPairKaon_truth->Print(Form("%s/cMassPairKaon_truth.root", outDir.Data()),"root");
        cMassPairKaon_truth->Print(Form("%s/cMassPairKaon_truth.pdf", outDir.Data()),"pdf");
        // Pair Transverse Momentum
        auto cPtPair_truth = new TCanvas("cPtPair_truth", "cPtPair_truth", 600, 400);
        cPtPair_truth->SetLogy();
        cPtPair_truth->cd();
        // auto hPtPair_truth = new TH1F("hPtPair_truth", "; p_{T} (GeV/c); Counts", 300, 0., 3.);
        // inputTree_truth->Project("hPtPair_truth", "Pt_MC_T");
        hPtPair_truth->Draw("e");
        hPtPair_truth->Write();
        cPtPair_truth->Print(Form("%s/cPtPair_truth.root", outDir.Data()),"root");
        cPtPair_truth->Print(Form("%s/cPtPair_truth.pdf", outDir.Data()),"pdf");

        // -------------- Invariant Mass vs. Pair Rapidity --------------
        auto cMassPairKaonVsRapidity_truth = new TCanvas("cMassPairKaonVsRapidity_truth", "cMassPairKaonVsRapidity_truth", 600, 400);
        cMassPairKaonVsRapidity_truth->cd();
        hMassPairKaonVsRapidity_truth->Draw("colz");
        hMassPairKaonVsRapidity_truth->Write();
        cMassPairKaonVsRapidity_truth->Print(Form("%s/cMassPairKaonVsRapidity_truth.root", outDir.Data()),"root");
        cMassPairKaonVsRapidity_truth->Print(Form("%s/cMassPairKaonVsRapidity_truth.pdf", outDir.Data()),"pdf");
    }

    // -------- Events per run
    auto cEvents = new TCanvas("cEvents", "cEvents", 1200, 500);
    cEvents->cd();
    hEvents->LabelsOption("v", "X");
    hEvents->Draw();
    hEvents->Write();
    cEvents->Print(Form("%s/cEvents_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cEvents->Print(Form("%s/cEvents_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // -------------- Track PseudoRapidity --------------
    auto cTrack1vs2PseudoRapidity = new TCanvas("cTrack1vs2PseudoRapidity", "cTrack1vs2PseudoRapidity", 600, 400);
    cTrack1vs2PseudoRapidity->cd();
    hTrack1vs2PseudoRapidity->Draw("colz");
    hTrack1vs2PseudoRapidity->Write();
    cTrack1vs2PseudoRapidity->Print(Form("%s/cTrack1vs2PseudoRapidity_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cTrack1vs2PseudoRapidity->Print(Form("%s/cTrack1vs2PseudoRapidity_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // -------------- Track Azimuthal Angle --------------
    auto cTrackAzimuthal = new TCanvas("cTrackAzimuthal", "cTrackAzimuthal", 600, 400);
    cTrackAzimuthal->cd();
    hTrackAzimuthal->SetMinimum(0.);
    // h_TrackAzimuthal->GetXaxis()->SetNdivisions(520);
    hTrackAzimuthal->Draw();
    hTrackAzimuthal->Write();
    cTrackAzimuthal->Print(Form("%s/cTrackAzimuthal_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cTrackAzimuthal->Print(Form("%s/cTrackAzimuthal_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // -------------- Pair Azimuthal Angle --------------
    auto cAzimuthal = new TCanvas("cAzimuthal", "cAzimuthal", 600, 400);
    cAzimuthal->cd();
    hAzimuthal->SetMinimum(0.);
    // h_Azimuthal->GetXaxis()->SetNdivisions(520);
    hAzimuthal->Draw();
    hAzimuthal->Write();
    cAzimuthal->Print(Form("%s/cAzimuthal_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cAzimuthal->Print(Form("%s/cAzimuthal_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // -------------- Primary Vertex Z --------------
    auto cVertexZ = new TCanvas("cVertexZ", "cVertexZ", 600, 400);
    cVertexZ->cd();
    hVertexZ->SetMinimum(0.);
    // inputTree->Project("hVertexZ", "VtxZ_T");
    // h_VertexZ->GetXaxis()->SetNdivisions(520);
    hVertexZ->Draw();
    hVertexZ->Write();
    cVertexZ->Print(Form("%s/cVertexZ_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cVertexZ->Print(Form("%s/cVertexZ_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // -------------- Pair Rapidity --------------
    auto cRapidity = new TCanvas("cRapidity", "cRapidity", 600, 400);
    cRapidity->cd();
    hRapidity->SetMinimum(0.);
    // inputTree->Project("hRapidity", "VtxZ_T");
    // h_Rapidity->GetXaxis()->SetNdivisions(520);
    hRapidity->Draw();
    hRapidity->Write();
    cRapidity->Print(Form("%s/cRapidity_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cRapidity->Print(Form("%s/cRapidity_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // ------------------------------------ Invariant Mass ------------------------------------ 

    // ##### Kaons

    auto cMassPairKaon = new TCanvas("cMassPairKaon", "cMassPairKaon", 600, 400);
    cMassPairKaon->cd();
    cMassPairKaon->SetLogy();
    auto lMassPairKaon = new TLegend(0.6, 0.7, 0.8, 0.9);
    lMassPairKaon->SetTextSize(0.03);
    lMassPairKaon->SetBorderSize(0);
    hMassPairKaon->SetMinimum(1.0);
    hMassPairKaon->Draw();
    hMassPairKaonUnLikeCharge->Draw("same");
    hMassPairKaonLikeCharge->Draw("same");
    hMassPairKaondEdxCut->Draw("same");
    hMassPairKaonPtCut01->Draw("esame");
    // hMassPairKaonPtCut01bkg->Draw("same");
    lMassPairKaon->AddEntry(hMassPairKaon, "2 track events", "l");
    lMassPairKaon->AddEntry(hMassPairKaonUnLikeCharge, "different charge", "l");
    lMassPairKaon->AddEntry(hMassPairKaonLikeCharge, "same charge", "l");
    lMassPairKaon->AddEntry(hMassPairKaondEdxCut, "-3<#sigma^{K^{#pm}}_{dE/dx}<3 + opposite charge", "lep");
    lMassPairKaon->AddEntry(hMassPairKaonPtCut01, "-3<#sigma^{K^{#pm}}_{dE/dx}<3 + opposite charge + p_{T}<0.1 GeV/c", "lep");
    // lMassPairKaon->AddEntry(hMassPairKaonPtCut01bkg, "-3<#sigma^{K^{#pm}}_{dE/dx}<3 + opposite charge + p_{T}>0.1 GeV/c", "lep");
    lMassPairKaon->Draw();
    hMassPairKaon->Write();
    hMassPairKaonUnLikeCharge->Write();
    hMassPairKaonLikeCharge->Write();
    hMassPairKaondEdxCut->Write();
    hMassPairKaonPtCut01->Write();
    // hMassPairKaonPtCut01bkg->Write();
    cMassPairKaon->Print(Form("%s/cMassPairKaon_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cMassPairKaon->Print(Form("%s/cMassPairKaon_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // ##### Pions

    auto cMassPairPion = new TCanvas("cMassPairPion", "cMassPairPion", 600, 400);
    cMassPairPion->cd();
    // cMassPairPion->SetLogy();
    auto lMassPairPion = new TLegend(0.5, 0.5, 0.7, 0.85);
    lMassPairPion->SetTextSize(0.03);
    lMassPairPion->SetBorderSize(0);
    // hMassPairPionPairPion->SetMinimum(1.0);
    hMassPairPion->Draw();
    hMassPairPionUnLikeCharge->Draw("same");
    hMassPairPionLikeCharge->Draw("same");
    hMassPairPiondEdxCut->Draw("same");
    hMassPairPionPtCut01->Draw("esame");
    // hMassPairPionPtCut01bkg->Draw("same");
    lMassPairPion->AddEntry(hMassPairPion, "2 track events", "l");
    lMassPairPion->AddEntry(hMassPairPionUnLikeCharge, "different charge", "l");
    lMassPairPion->AddEntry(hMassPairPionLikeCharge, "same charge", "l");
    lMassPairPion->AddEntry(hMassPairPiondEdxCut, "-3<#sigma^{#pi^{#pm}}_{dE/dx}<3 + opposite charge", "l");
    lMassPairPion->AddEntry(hMassPairPionPtCut01, "-3<#sigma^{#pi^{#pm}}_{dE/dx}<3 + opposite charge + p_{T}<0.1 GeV/c", "lep");
    // lMassPairPion->AddEntry(hMassPairPionPtCut01bkg, "-3<#sigma^{#pi^{#pm}}_{dE/dx}<3 + opposite charge + p_{T}>0.1 GeV/c", "l");
    lMassPairPion->Draw();
    hMassPairPion->Write();
    hMassPairPionUnLikeCharge->Write();
    hMassPairPionLikeCharge->Write();
    hMassPairPiondEdxCut->Write();
    hMassPairPionPtCut01->Write();
    // hMassPairPionPtCut01bkg->Write();
    cMassPairPion->Print(Form("%s/cMassPairPion_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cMassPairPion->Print(Form("%s/cMassPairPion_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // ------------- select kaon with other particles rejected

    auto cMassPairKaonSelect = new TCanvas("cMassPairKaonSelect", "cMassPairKaonSelect", 600, 400);
    cMassPairKaonSelect->cd();
    cMassPairKaonSelect->SetLogy();
    auto lMassPairKaonSelect = new TLegend(0.6, 0.6, 0.8, 0.9);
    lMassPairKaonSelect->SetTextSize(0.03);
    lMassPairKaonSelect->SetBorderSize(0);
    hMassPairKaonUnLikeCharge->SetMinimum(1.0);
    // hMassPairKaonUnLikeCharge->Rebin(2);
    hMassPairKaonUnLikeCharge->Draw();
    hMassPairKaonRejectPion->Draw("same");
    hMassPairKaonRejectPionMuon->Draw("same");
    hMassPairKaonRejectAll->Draw("same");
    hMassPairKaon2Std->Draw("same");
    lMassPairKaonSelect->AddEntry(hMassPairKaonUnLikeCharge, "opposite charge", "l");
    lMassPairKaonSelect->AddEntry(hMassPairKaonRejectPion, "-2>#sigma^{#pi}_{dE/dx}>2", "l");
    lMassPairKaonSelect->AddEntry(hMassPairKaonRejectPionMuon, "-2>#sigma^{(#pi, #mu)}_{dE/dx}>2", "l");
    lMassPairKaonSelect->AddEntry(hMassPairKaonRejectAll, "-2>#sigma^{(#pi, #mu, e)}_{dE/dx}>2", "l");
    lMassPairKaonSelect->AddEntry(hMassPairKaon2Std, "-2>#sigma^{(#pi, #mu, e)}_{dE/dx}>2 &  -2<#sigma^{K}_{dE/dx}<2", "l");
    lMassPairKaonSelect->Draw();
    hMassPairKaonRejectPion->Write();
    hMassPairKaonRejectPionMuon->Write();
    hMassPairKaonRejectAll->Write();
    hMassPairKaon2Std->Write();
    cMassPairKaonSelect->Print(Form("%s/cMassPairKaonSelect_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cMassPairKaonSelect->Print(Form("%s/cMassPairKaonSelect_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // ------------- Signal, Bkg and their subtraction

    // Select Kaon
    
    auto cMassPairKaonSub = new TCanvas("cMassPairKaonSub", "cMassPairKaonSub", 600, 400);
    cMassPairKaonSub->cd();
    auto hMassPairKaonSub = new TH1F("hMassPairKaonSub", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonSub->Add(hMassPairKaonSig, hMassPairKaonBkg, 1., -1.);
    hMassPairKaonSub->SetLineColor(1); 
    hMassPairKaonSub->SetMarkerStyle(kOpenCircle);
    // cMassPairKaon->SetLogy();
    auto lMassPairKaonSub = new TLegend(0.6, 0.7, 0.8, 0.9);
    lMassPairKaonSub->SetTextSize(0.03);
    lMassPairKaonSub->SetBorderSize(0);
    // hMassPairKaonSig->SetMinimum(1.0);
    hMassPairKaonSig->Draw("e");
    hMassPairKaonBkg->Draw("esame");
    hMassPairKaonSub->Draw("esame");
    lMassPairKaonSub->AddEntry(hMassPairKaonSig, "#sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 9", "l");
    lMassPairKaonSub->AddEntry(hMassPairKaonBkg, "16 < #sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 25", "l");
    lMassPairKaonSub->AddEntry(hMassPairKaonSub, "Subtracted", "p");
    lMassPairKaonSub->Draw();
    hMassPairKaonSig->Write();
    hMassPairKaonBkg->Write();
    hMassPairKaonSub->Write();
    cMassPairKaonSub->Print(Form("%s/cMassPairKaonSub_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cMassPairKaonSub->Print(Form("%s/cMassPairKaonSub_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // Select Kaon & Reject Pion

    auto cMassPairKaonRejectPionSub = new TCanvas("cMassPairKaonRejectPionSub", "cMassPairKaonRejectPionSub", 600, 400);
    cMassPairKaonRejectPionSub->cd();
    auto hMassPairKaonRejectPionSub = new TH1F("hMassPairKaonRejectPionSub", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonRejectPionSub->Add(hMassPairKaonRejectPionSig, hMassPairKaonRejectPionBkg, 1., -1.);
    hMassPairKaonRejectPionSub->SetLineColor(1); 
    hMassPairKaonRejectPionSub->SetMarkerStyle(kOpenCircle);
    // cMassPairKaonRejectPion->SetLogy();
    auto lMassPairKaonRejectPionSub = new TLegend(0.6, 0.7, 0.8, 0.9);
    lMassPairKaonRejectPionSub->SetTextSize(0.03);
    lMassPairKaonRejectPionSub->SetBorderSize(0);
    // hMassPairKaonRejectPionSig->SetMinimum(1.0);
    hMassPairKaonRejectPionSig->Draw("e");
    hMassPairKaonRejectPionBkg->Draw("esame");
    hMassPairKaonRejectPionSub->Draw("esame");
    lMassPairKaonRejectPionSub->AddEntry(hMassPairKaonRejectPionSig, "#sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 9", "l");
    lMassPairKaonRejectPionSub->AddEntry(hMassPairKaonRejectPionBkg, "16 < #sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 25", "l");
    lMassPairKaonRejectPionSub->AddEntry(hMassPairKaonRejectPionSub, "Subtracted", "p");
    lMassPairKaonRejectPionSub->Draw();
    hMassPairKaonRejectPionSig->Write();
    hMassPairKaonRejectPionBkg->Write();
    hMassPairKaonRejectPionSub->Write();
    cMassPairKaonRejectPionSub->Print(Form("%s/cMassPairKaonRejectPionSub_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cMassPairKaonRejectPionSub->Print(Form("%s/cMassPairKaonRejectPionSub_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // Select Kaon & Reject (Pion + Muon)
    
    auto cMassPairKaonRejectPionMuonSub = new TCanvas("cMassPairKaonRejectPionMuonSub", "cMassPairKaonRejectPionMuonSub", 600, 400);
    cMassPairKaonRejectPionMuonSub->cd();
    auto hMassPairKaonRejectPionMuonSub = new TH1F("hMassPairKaonRejectPionMuonSub", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonRejectPionMuonSub->Add(hMassPairKaonRejectPionMuonSig, hMassPairKaonRejectPionMuonBkg, 1., -1.);
    hMassPairKaonRejectPionMuonSub->SetLineColor(1); 
    hMassPairKaonRejectPionMuonSub->SetMarkerStyle(kOpenCircle);
    // cMassPairKaonRejectPionMuon->SetLogy();
    auto lMassPairKaonRejectPionMuonSub = new TLegend(0.6, 0.7, 0.8, 0.9);
    lMassPairKaonRejectPionMuonSub->SetTextSize(0.03);
    lMassPairKaonRejectPionMuonSub->SetBorderSize(0);
    // hMassPairKaonRejectPionMuonSig->SetMinimum(1.0);
    hMassPairKaonRejectPionMuonSig->Draw("e");
    hMassPairKaonRejectPionMuonBkg->Draw("esame");
    hMassPairKaonRejectPionMuonSub->Draw("esame");
    lMassPairKaonRejectPionMuonSub->AddEntry(hMassPairKaonRejectPionMuonSig, "#sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 9", "l");
    lMassPairKaonRejectPionMuonSub->AddEntry(hMassPairKaonRejectPionMuonBkg, "16 < #sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 25", "l");
    lMassPairKaonRejectPionMuonSub->AddEntry(hMassPairKaonRejectPionMuonSub, "Subtracted", "p");
    lMassPairKaonRejectPionMuonSub->Draw();
    hMassPairKaonRejectPionMuonSig->Write();
    hMassPairKaonRejectPionMuonBkg->Write();
    hMassPairKaonRejectPionMuonSub->Write();
    cMassPairKaonRejectPionMuonSub->Print(Form("%s/cMassPairKaonRejectPionMuonSub_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cMassPairKaonRejectPionMuonSub->Print(Form("%s/cMassPairKaonRejectPionMuonSub_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // Select Kaon & Reject (Pion + Muon + Electron)
    
    auto cMassPairKaonRejectAllSub = new TCanvas("cMassPairKaonRejectAllSub", "cMassPairKaonRejectAllSub", 600, 400);
    cMassPairKaonRejectAllSub->cd();
    auto hMassPairKaonRejectAllSub = new TH1F("hMassPairKaonRejectAllSub", "; m_{K^{+}K^{-}} (GeV/c^{2}); Counts", 100, 0.95, 2.0);
    hMassPairKaonRejectAllSub->Add(hMassPairKaonRejectAllSig, hMassPairKaonRejectAllBkg, 1., -1.);
    hMassPairKaonRejectAllSub->SetLineColor(1); 
    hMassPairKaonRejectAllSub->SetMarkerStyle(kOpenCircle);
    // cMassPairKaonRejectAll->SetLogy();
    auto lMassPairKaonRejectAllSub = new TLegend(0.6, 0.7, 0.8, 0.9);
    lMassPairKaonRejectAllSub->SetTextSize(0.03);
    lMassPairKaonRejectAllSub->SetBorderSize(0);
    // hMassPairKaonRejectAllSig->SetMinimum(1.0);
    hMassPairKaonRejectAllSig->Draw("e");
    hMassPairKaonRejectAllBkg->Draw("esame");
    hMassPairKaonRejectAllSub->Draw("esame");
    lMassPairKaonRejectAllSub->AddEntry(hMassPairKaonRejectAllSig, "#sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 9", "l");
    lMassPairKaonRejectAllSub->AddEntry(hMassPairKaonRejectAllBkg, "16 < #sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 25", "l");
    lMassPairKaonRejectAllSub->AddEntry(hMassPairKaonRejectAllSub, "Subtracted", "p");
    lMassPairKaonRejectAllSub->Draw();
    hMassPairKaonRejectAllSig->Write();
    hMassPairKaonRejectAllBkg->Write();
    hMassPairKaonRejectAllSub->Write();
    cMassPairKaonRejectAllSub->Print(Form("%s/cMassPairKaonRejectAllSub_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cMassPairKaonRejectAllSub->Print(Form("%s/cMassPairKaonRejectAllSub_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // in different rapidity regions

    auto cMassPairKaonVsRapiditySub = new TCanvas("cMassPairKaonVsRapiditySub", "cMassPairKaonVsRapiditySub", 600, 400);
    cMassPairKaonVsRapiditySub->cd();
    auto hMassPairKaonVsRapiditySub = new TH2D("hMassPairKaonVsRapiditySub", ";pair y;m_{K^{+}K^{-}} (GeV/c^{2});Counts", 3,-0.8,0.8, 100, 0.95, 2.0);
    hMassPairKaonVsRapiditySub->Add(hMassPairKaonVsRapiditySig, hMassPairKaonVsRapidityBkg, 1., -1.);
    hMassPairKaonVsRapiditySub->Draw("colz");
    hMassPairKaonVsRapiditySub->Write();
    cMassPairKaonVsRapiditySub->Print(Form("%s/cMassPairKaonVsRapiditySub_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cMassPairKaonVsRapiditySub->Print(Form("%s/cMassPairKaonVsRapiditySub_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    //  ------------------------------------ Pair transverse momentum ------------------------------------ 

    auto cPt = new TCanvas("cPt", "cPt", 600, 400);
    cPt->cd();
    cPt->SetLogy();
    auto lPt = new TLegend(0.5, 0.5, 0.7, 0.85);
    lPt->SetTextSize(0.03);
    lPt->SetBorderSize(0);
    // hPtPair->SetMinimum(1.0);
    hPtPair->Draw();
    hPtUnLikeCharge->Draw("same");
    hPtLikeCharge->Draw("same");
    // hPtPiondEdxCut->Draw("same");
    hPtKaondEdxCut->Draw("same");
    lPt->AddEntry(hPtPair, "2 track events", "l");
    lPt->AddEntry(hPtUnLikeCharge, "different charge", "l");
    lPt->AddEntry(hPtLikeCharge, "same charge", "l");
    // lPt->AddEntry(hPtPiondEdxCut, "-3<#sigma^{#pi^{#pm}}_{dE/dx}<3 + opposite charge", "l");
    lPt->AddEntry(hPtKaondEdxCut, "-3<#sigma^{K^{#pm}}_{dE/dx}<3 + opposite charge", "l");
    lPt->Draw();
    hPtPair->Write();
    hPtUnLikeCharge->Write();
    hPtLikeCharge->Write();
    // hPtPiondEdxCut->Write();
    hPtKaondEdxCut->Write();
    cPt->Print(Form("%s/cPt_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cPt->Print(Form("%s/cPt_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // ------------- Signal, Bkg and their subtraction

    // Select Kaon
    
    auto cPtPairKaonSub = new TCanvas("cPtPairKaonSub", "cPtPairKaonSub", 600, 400);
    cPtPairKaonSub->cd();
    auto hPtPairKaonSub = new TH1F("hPtPairKaonSub", "; p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonSub->Add(hPtPairKaonSig, hPtPairKaonBkg, 1., -1.);
    hPtPairKaonSub->SetLineColor(1); 
    hPtPairKaonSub->SetMarkerStyle(kOpenCircle);
    // cPtPairKaon->SetLogy();
    auto lPtPairKaonSub = new TLegend(0.6, 0.7, 0.8, 0.9);
    lPtPairKaonSub->SetTextSize(0.03);
    lPtPairKaonSub->SetBorderSize(0);
    // hPtPairKaonSig->SetMinimum(1.0);
    hPtPairKaonSig->Draw("e");
    hPtPairKaonBkg->Draw("esame");
    hPtPairKaonSub->Draw("esame");
    lPtPairKaonSub->AddEntry(hPtPairKaonSig, "#sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 9", "l");
    lPtPairKaonSub->AddEntry(hPtPairKaonBkg, "16 < #sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 25", "l");
    lPtPairKaonSub->AddEntry(hPtPairKaonSub, "Subtracted", "p");
    lPtPairKaonSub->Draw();
    hPtPairKaonSig->Write();
    hPtPairKaonBkg->Write();
    hPtPairKaonSub->Write();
    cPtPairKaonSub->Print(Form("%s/cPtPairKaonSub_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cPtPairKaonSub->Print(Form("%s/cPtPairKaonSub_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // Select Kaon & Reject Pion

    auto cPtPairKaonRejectPionSub = new TCanvas("cPtPairKaonRejectPionSub", "cPtPairKaonRejectPionSub", 600, 400);
    cPtPairKaonRejectPionSub->cd();
    auto hPtPairKaonRejectPionSub = new TH1F("hPtPairKaonRejectPionSub", "; p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonRejectPionSub->Add(hPtPairKaonRejectPionSig, hPtPairKaonRejectPionBkg, 1., -1.);
    hPtPairKaonRejectPionSub->SetLineColor(1); 
    hPtPairKaonRejectPionSub->SetMarkerStyle(kOpenCircle);
    // cPtPairKaonRejectPion->SetLogy();
    auto lPtPairKaonRejectPionSub = new TLegend(0.6, 0.7, 0.8, 0.9);
    lPtPairKaonRejectPionSub->SetTextSize(0.03);
    lPtPairKaonRejectPionSub->SetBorderSize(0);
    // hPtPairKaonRejectPionSig->SetMinimum(1.0);
    hPtPairKaonRejectPionSig->Draw("e");
    hPtPairKaonRejectPionBkg->Draw("esame");
    hPtPairKaonRejectPionSub->Draw("esame");
    lPtPairKaonRejectPionSub->AddEntry(hPtPairKaonRejectPionSig, "#sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 9", "l");
    lPtPairKaonRejectPionSub->AddEntry(hPtPairKaonRejectPionBkg, "16 < #sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 25", "l");
    lPtPairKaonRejectPionSub->AddEntry(hPtPairKaonRejectPionSub, "Subtracted", "p");
    lPtPairKaonRejectPionSub->Draw();
    hPtPairKaonRejectPionSig->Write();
    hPtPairKaonRejectPionBkg->Write();
    hPtPairKaonRejectPionSub->Write();
    cPtPairKaonRejectPionSub->Print(Form("%s/cPtPairKaonRejectPionSub_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cPtPairKaonRejectPionSub->Print(Form("%s/cPtPairKaonRejectPionSub_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // Select Kaon & Reject (Pion + Muon)
    
    auto cPtPairKaonRejectPionMuonSub = new TCanvas("cPtPairKaonRejectPionMuonSub", "cPtPairKaonRejectPionMuonSub", 600, 400);
    cPtPairKaonRejectPionMuonSub->cd();
    auto hPtPairKaonRejectPionMuonSub = new TH1F("hPtPairKaonRejectPionMuonSub", "; p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonRejectPionMuonSub->Add(hPtPairKaonRejectPionMuonSig, hPtPairKaonRejectPionMuonBkg, 1., -1.);
    hPtPairKaonRejectPionMuonSub->SetLineColor(1); 
    hPtPairKaonRejectPionMuonSub->SetMarkerStyle(kOpenCircle);
    // cPtPairKaonRejectPionMuon->SetLogy();
    auto lPtPairKaonRejectPionMuonSub = new TLegend(0.6, 0.7, 0.8, 0.9);
    lPtPairKaonRejectPionMuonSub->SetTextSize(0.03);
    lPtPairKaonRejectPionMuonSub->SetBorderSize(0);
    // hPtPairKaonRejectPionMuonSig->SetMinimum(1.0);
    hPtPairKaonRejectPionMuonSig->Draw("e");
    hPtPairKaonRejectPionMuonBkg->Draw("esame");
    hPtPairKaonRejectPionMuonSub->Draw("esame");
    lPtPairKaonRejectPionMuonSub->AddEntry(hPtPairKaonRejectPionMuonSig, "#sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 9", "l");
    lPtPairKaonRejectPionMuonSub->AddEntry(hPtPairKaonRejectPionMuonBkg, "16 < #sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 25", "l");
    lPtPairKaonRejectPionMuonSub->AddEntry(hPtPairKaonRejectPionMuonSub, "Subtracted", "p");
    lPtPairKaonRejectPionMuonSub->Draw();
    hPtPairKaonRejectPionMuonSig->Write();
    hPtPairKaonRejectPionMuonBkg->Write();
    hPtPairKaonRejectPionMuonSub->Write();
    cPtPairKaonRejectPionMuonSub->Print(Form("%s/cPtPairKaonRejectPionMuonSub_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cPtPairKaonRejectPionMuonSub->Print(Form("%s/cPtPairKaonRejectPionMuonSub_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // Select Kaon & Reject (Pion + Muon + Electron)
    
    auto cPtPairKaonRejectAllSub = new TCanvas("cPtPairKaonRejectAllSub", "cPtPairKaonRejectAllSub", 600, 400);
    cPtPairKaonRejectAllSub->cd();
    auto hPtPairKaonRejectAllSub = new TH1F("hPtPairKaonRejectAllSub", "; p_{T} (GeV/c); Counts", 100, 0., 0.1);
    hPtPairKaonRejectAllSub->Add(hPtPairKaonRejectAllSig, hPtPairKaonRejectAllBkg, 1., -1.);
    hPtPairKaonRejectAllSub->SetLineColor(1); 
    hPtPairKaonRejectAllSub->SetMarkerStyle(kOpenCircle);
    // cPtPairKaonRejectAll->SetLogy();
    auto lPtPairKaonRejectAllSub = new TLegend(0.6, 0.7, 0.8, 0.9);
    lPtPairKaonRejectAllSub->SetTextSize(0.03);
    lPtPairKaonRejectAllSub->SetBorderSize(0);
    // hPtPairKaonRejectAllSig->SetMinimum(1.0);
    hPtPairKaonRejectAllSig->Draw("e");
    hPtPairKaonRejectAllBkg->Draw("esame");
    hPtPairKaonRejectAllSub->Draw("esame");
    lPtPairKaonRejectAllSub->AddEntry(hPtPairKaonRejectAllSig, "#sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 9", "l");
    lPtPairKaonRejectAllSub->AddEntry(hPtPairKaonRejectAllBkg, "16 < #sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 25", "l");
    lPtPairKaonRejectAllSub->AddEntry(hPtPairKaonRejectAllSub, "Subtracted", "p");
    lPtPairKaonRejectAllSub->Draw();
    hPtPairKaonRejectAllSig->Write();
    hPtPairKaonRejectAllBkg->Write();
    hPtPairKaonRejectAllSub->Write();
    cPtPairKaonRejectAllSub->Print(Form("%s/cPtPairKaonRejectAllSub_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cPtPairKaonRejectAllSub->Print(Form("%s/cPtPairKaonRejectAllSub_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // -------------- S.t.d dE/dx vs. track momentum

    // dE/dx vs p
    auto cDEdxVsP = new TCanvas("cDEdxVsP", "cDEdxVsP", 600, 400);
    cDEdxVsP->cd();
    hDEdxVsP->Draw("colz");
    hDEdxVsP->Write();
    cDEdxVsP->Print(Form("%s/cDEdxVsP_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cDEdxVsP->Print(Form("%s/cDEdxVsP_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");
    // Kaon
    auto cStdDEdxVsPKaon = new TCanvas("cStdDEdxVsPKaon", "cStdDEdxVsPKaon", 600, 400);
    cStdDEdxVsPKaon->cd();
    hStdDEdxVsPKaon->Draw("colz");
    hStdDEdxVsPKaon->Write();
    cStdDEdxVsPKaon->Print(Form("%s/cStdDEdxVsPKaon_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxVsPKaon->Print(Form("%s/cStdDEdxVsPKaon_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");
    // Pion
    auto cStdDEdxVsPPion = new TCanvas("cStdDEdxVsPPion", "cStdDEdxVsPPion", 600, 400);
    cStdDEdxVsPPion->cd();
    hStdDEdxVsPPion->Draw("colz");
    hStdDEdxVsPPion->Write();
    cStdDEdxVsPPion->Print(Form("%s/cStdDEdxVsPPion_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxVsPPion->Print(Form("%s/cStdDEdxVsPPion_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");
    // Muon
    auto cStdDEdxVsPMuon = new TCanvas("cStdDEdxVsPMuon", "cStdDEdxVsPMuon", 600, 400);
    cStdDEdxVsPMuon->cd();
    hStdDEdxVsPMuon->Draw("colz");
    hStdDEdxVsPMuon->Write();
    cStdDEdxVsPMuon->Print(Form("%s/cStdDEdxVsPMuon_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxVsPMuon->Print(Form("%s/cStdDEdxVsPMuon_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");
    // Electron
    auto cStdDEdxVsPElectron = new TCanvas("cStdDEdxVsPElectron", "cStdDEdxVsPElectron", 600, 400);
    cStdDEdxVsPElectron->cd();
    hStdDEdxVsPElectron->Draw("colz");
    hStdDEdxVsPElectron->Write();
    cStdDEdxVsPElectron->Print(Form("%s/cStdDEdxVsPElectron_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxVsPElectron->Print(Form("%s/cStdDEdxVsPElectron_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    //------ dE/dx sigma 1 vs. dE/dx sigma 2

    // Kaons

    auto cStdDEdx2Kaon = new TCanvas("cStdDEdx2Kaon", "cStdDEdx2Kaon", 600, 400);
    cStdDEdx2Kaon->cd();
    hStdDEdx2Kaon->Draw("colz");
    hStdDEdx2Kaon->Write();
    cStdDEdx2Kaon->Print(Form("%s/cStdDEdx2Kaon_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdx2Kaon->Print(Form("%s/cStdDEdx2Kaon_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // reject Pion
    auto cStdDEdx2KaonRejectPion = new TCanvas("cStdDEdx2KaonRejectPion", "cStdDEdx2KaonRejectPion", 600, 400);
    cStdDEdx2KaonRejectPion->cd();
    hStdDEdx2KaonRejectPion->Draw("colz");
    auto el1 = new TEllipse(x0, y0, 3., 3.);
    el1->SetFillStyle(0);
    el1->SetLineColor(4);
    auto el2 = new TEllipse(x0, y0, 4.,4.);
    el2->SetFillStyle(0);
    el2->SetLineColor(2);
    auto el3 = new TEllipse(x0, y0, 5.,5.);
    el3->SetFillStyle(0);
    el3->SetLineColor(2);
    el1->Draw("same");
    el2->Draw("same");
    el3->Draw("same");
    hStdDEdx2KaonRejectPion->Write();
    cStdDEdx2KaonRejectPion->Print(Form("%s/cStdDEdx2KaonRejectPion_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdx2KaonRejectPion->Print(Form("%s/cStdDEdx2KaonRejectPion_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");
    
    // reject Pion + Muon
    auto cStdDEdx2KaonRejectPionMuon = new TCanvas("cStdDEdx2KaonRejectPionMuon", "cStdDEdx2KaonRejectPionMuon", 600, 400);
    cStdDEdx2KaonRejectPionMuon->cd();
    hStdDEdx2KaonRejectPionMuon->Draw("colz");
    hStdDEdx2KaonRejectPionMuon->Write();
    cStdDEdx2KaonRejectPionMuon->Print(Form("%s/cStdDEdx2KaonRejectPionMuon_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdx2KaonRejectPionMuon->Print(Form("%s/cStdDEdx2KaonRejectPionMuon_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    //reject Pion + Muon + Electron
    auto cStdDEdx2KaonRejectAll = new TCanvas("cStdDEdx2KaonRejectAll", "cStdDEdx2KaonRejectAll", 600, 400);
    cStdDEdx2KaonRejectAll->cd();
    hStdDEdx2KaonRejectAll->Draw("colz");
    hStdDEdx2KaonRejectAll->Write();
    cStdDEdx2KaonRejectAll->Print(Form("%s/cStdDEdx2KaonRejectAll_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdx2KaonRejectAll->Print(Form("%s/cStdDEdx2KaonRejectAll_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // reject Pion + Muon + Electron while keeping kaon
    auto cStdDEdx2Kaon2Std = new TCanvas("cStdDEdx2Kaon2Std", "cStdDEdx2Kaon2Std", 600, 400);
    cStdDEdx2Kaon2Std->cd();
    hStdDEdx2Kaon2Std->Draw("colz");
    hStdDEdx2Kaon2Std->Write();
    cStdDEdx2Kaon2Std->Print(Form("%s/cStdDEdx2Kaon2Std_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdx2Kaon2Std->Print(Form("%s/cStdDEdx2Kaon2Std_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // --------- Count sig and Bkg events after different PID cuts
    Int_t cuts[4] = {1, 2, 3, 4};
    Int_t Nsig[4] = {NumbKaonSig1, NumbKaonSig2, NumbKaonSig3, NumbKaonSig4};
    Int_t Nbkg[4] = {NumbKaonBkg1, NumbKaonBkg2, NumbKaonBkg3, NumbKaonBkg4};
    auto cNumbKaon = new TCanvas("cNumbKaon", "cNumbKaon", 600, 400);
    cNumbKaon->cd();
    // auto h = new TH1F("h","test",4,cuts[0],cuts[4-1]);
    // for(int i=1;i<=4;i++) h->GetXaxis()->SetBinLabel(i,cutsLabel[i-1]);
    // h->Draw();
    auto grNumbKaonSig = new TGraph(4, cuts, Nsig);
    grNumbKaonSig->SetMarkerStyle(20);
    grNumbKaonSig->SetMarkerColor(kBlue);
    grNumbKaonSig->SetMinimum(0.0);
    grNumbKaonSig->SetTitle(";Cuts;Number of kaon pair events");
    auto grNumbKaonBkg = new TGraph(4, cuts, Nbkg);
    grNumbKaonBkg->SetMarkerStyle(20);
    grNumbKaonBkg->SetMarkerColor(kRed);
    auto lNumbKaon = new TLegend(0.68, 0.78, 0.77, 0.87);
    lNumbKaon->SetTextSize(0.03);
    lNumbKaon->SetBorderSize(0);
    grNumbKaonSig->Draw("AP");
    grNumbKaonBkg->Draw("PSAME");
    lNumbKaon->AddEntry(grNumbKaonSig, "#sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 9", "p");
    lNumbKaon->AddEntry(grNumbKaonBkg, "16 < #sigma^{2}_{K^{+}} + #sigma^{2}_{K^{-}} < 25", "p");
    lNumbKaon->Draw();
    grNumbKaonSig->Write();
    grNumbKaonBkg->Write();
    cNumbKaon->Print(Form("%s/cNumbKaon_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cNumbKaon->Print(Form("%s/cNumbKaon_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    fprintf(table_Phi0Ana, "None & %0.d & %0.d & %0.d \\\\ \n", Nsig[0], Nbkg[0], Nsig[0]-Nbkg[0]);
    fprintf(table_Phi0Ana, "$|\\sigma^{\\pi}_{dE/dx}| > 2$ & %0.d & %0.d & %0.d \\\\ \n", Nsig[1], Nbkg[1], Nsig[1]-Nbkg[1]);
    fprintf(table_Phi0Ana, "$|\\sigma^{(\\pi, \\mu)}_{dE/dx}| > 2$ & %0.d & %0.d & %0.d \\\\ \n", Nsig[2], Nbkg[2], Nsig[2]-Nbkg[2]);
    fprintf(table_Phi0Ana, "$|\\sigma^{(\\pi, \\mu, e)}_{dE/dx}| > 2$ & %0.d & %0.d & %0.d \\\\ \n", Nsig[3], Nbkg[3], Nsig[3]-Nbkg[3]);

    fprintf(table_Phi0Ana, "\\hline\n \\end{tabular}\n \\end{table}\n \\end{document}\n");
    fclose(table_Phi0Ana);
    gSystem->Exec(Form("pdflatex -output-directory %s %s/table_Phi0Ana_%s.tex", outDir.Data(), outDir.Data(), dataORmc.Data()));

    // cout<<"Nsig[0] = "<<Nsig[0]<<" | Nbkg[0] = "<<Nbkg[0]<<endl;
    // cout<<"Nsig[1] = "<<Nsig[1]<<" | Nbkg[1] = "<<Nbkg[1]<<endl;
    // cout<<"Nsig[2] = "<<Nsig[2]<<" | Nbkg[2] = "<<Nbkg[2]<<endl;
    // cout<<"Nsig[3] = "<<Nsig[3]<<" | Nbkg[3] = "<<Nbkg[3]<<endl;
    // // count number of events pairs with only one of them is kaon
    // cout<<"-------- Number of event pairs with two of them is kaon = "<<NumbKaon2PID<<endl;
    // cout<<"-------- Number of events pairs with only one of them is kaon = "<<NumbKaon1PID<<endl;

    
    // ------ dE/dx sigma 1 in different pT and Eta regions
    // Pt
    auto cStdDEdxKaon_PtCut = new TCanvas("cStdDEdxKaon_PtCut", "cStdDEdxKaon_PtCut", 600, 400);
    cStdDEdxKaon_PtCut->cd();
    cStdDEdxKaon_PtCut->SetLogy();
    auto lStdDEdxKaon_PtCut = new TLegend(0.6, 0.7, 0.8, 0.9); // 0.2, 0.7, 0.4, 0.9,  0.6, 0.7, 0.8, 0.9
    lStdDEdxKaon_PtCut->SetTextSize(0.03);
    lStdDEdxKaon_PtCut->SetBorderSize(0);
    hStdDEdxKaon_Pt025->SetMinimum(1.0);
    hStdDEdxKaon_Pt025->Draw();
    hStdDEdxKaon_Pt050->Draw("same");
    hStdDEdxKaon_Pt075->Draw("same");
    hStdDEdxKaon_Pt100->Draw("same");
    lStdDEdxKaon_PtCut->AddEntry(hStdDEdxKaon_Pt025, "Pair-P_{T} < 0.25 GeV", "l");
    lStdDEdxKaon_PtCut->AddEntry(hStdDEdxKaon_Pt050, "0.25 < Pair P_{T} < 0.50 GeV", "l");
    lStdDEdxKaon_PtCut->AddEntry(hStdDEdxKaon_Pt075, "0.50 < Pair P_{T} < 0.75 GeV", "l");
    lStdDEdxKaon_PtCut->AddEntry(hStdDEdxKaon_Pt100, "0.75 < Pair P_{T} < 1 GeV", "l");
    lStdDEdxKaon_PtCut->Draw();
    hStdDEdxKaon_Pt025->Write();
    hStdDEdxKaon_Pt050->Write();
    hStdDEdxKaon_Pt075->Write();
    hStdDEdxKaon_Pt100->Write();
    cStdDEdxKaon_PtCut->Print(Form("%s/cStdDEdxKaon_PtCut_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxKaon_PtCut->Print(Form("%s/cStdDEdxKaon_PtCut_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // Eta
    auto cStdDEdxKaon_EtaCut = new TCanvas("cStdDEdxKaon_EtaCut", "cStdDEdxKaon_EtaCut", 600, 400);
    cStdDEdxKaon_EtaCut->cd();
    cStdDEdxKaon_EtaCut->SetLogy();
    auto lStdDEdxKaon_EtaCut = new TLegend(0.6, 0.7, 0.8, 0.9);
    lStdDEdxKaon_EtaCut->SetTextSize(0.03);
    lStdDEdxKaon_EtaCut->SetBorderSize(0);
    hStdDEdxKaon_Eta100->SetMinimum(1.0);
    hStdDEdxKaon_Eta100->Draw();
    hStdDEdxKaon_Eta075->Draw("same");
    hStdDEdxKaon_Eta050->Draw("same");
    hStdDEdxKaon_Eta025->Draw("same");
    lStdDEdxKaon_EtaCut->AddEntry(hStdDEdxKaon_Eta025, "|Pair #eta| < 0.25", "l");
    lStdDEdxKaon_EtaCut->AddEntry(hStdDEdxKaon_Eta050, "|Pair #eta| < 0.50", "l");
    lStdDEdxKaon_EtaCut->AddEntry(hStdDEdxKaon_Eta075, "|Pair #eta| < 0.75", "l");
    lStdDEdxKaon_EtaCut->AddEntry(hStdDEdxKaon_Eta100, "|Pair #eta| < 1", "l");
    lStdDEdxKaon_EtaCut->Draw();
    hStdDEdxKaon_Eta025->Write();
    hStdDEdxKaon_Eta050->Write();
    hStdDEdxKaon_Eta075->Write();
    hStdDEdxKaon_Eta100->Write();
    cStdDEdxKaon_EtaCut->Print(Form("%s/cStdDEdxKaon_EtaCut_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxKaon_EtaCut->Print(Form("%s/cStdDEdxKaon_EtaCut_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    // Kaon
    auto cStdDEdxKaon = new TCanvas("cStdDEdxKaon", "cStdDEdxKaon", 600, 400);
    cStdDEdxKaon->cd();
    hStdDEdxKaon->SetMinimum(1.0);
    hStdDEdxKaon->Draw();
    hStdDEdxKaon->Write();
    cStdDEdxKaon->Print(Form("%s/cStdDEdxKaon_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxKaon->Print(Form("%s/cStdDEdxKaon_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");
    // KaonNotSelected
    auto cStdDEdxKaonNotSelected = new TCanvas("cStdDEdxKaonNotSelected", "cStdDEdxKaonNotSelected", 600, 400);
    cStdDEdxKaonNotSelected->cd();
    hStdDEdxKaonNotSelected->SetMinimum(1.0);
    hStdDEdxKaonNotSelected->Draw();
    hStdDEdxKaonNotSelected->Fit("gaus","","", -2.0, 3.0);
    hStdDEdxKaonNotSelected->Write();
    cStdDEdxKaonNotSelected->Print(Form("%s/cStdDEdxKaonNotSelected_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxKaonNotSelected->Print(Form("%s/cStdDEdxKaonNotSelected_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");
    // Pion
    auto cStdDEdxPion = new TCanvas("cStdDEdxPion", "cStdDEdxPion", 600, 400);
    cStdDEdxPion->cd();
    hStdDEdxPion->SetMinimum(1.0);
    hStdDEdxPion->Draw();
    hStdDEdxPion->Write();
    cStdDEdxPion->Print(Form("%s/cStdDEdxPion_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxPion->Print(Form("%s/cStdDEdxPion_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");
    // Muon
    auto cStdDEdxMuon = new TCanvas("cStdDEdxMuon", "cStdDEdxMuon", 600, 400);
    cStdDEdxMuon->cd();
    hStdDEdxMuon->SetMinimum(1.0);
    hStdDEdxMuon->Draw();
    hStdDEdxMuon->Write();
    cStdDEdxMuon->Print(Form("%s/cStdDEdxMuon_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxMuon->Print(Form("%s/cStdDEdxMuon_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");
    // Electron
    auto cStdDEdxElectron = new TCanvas("cStdDEdxElectron", "cStdDEdxElectron", 600, 400);
    cStdDEdxElectron->cd();
    hStdDEdxElectron->SetMinimum(1.0);
    hStdDEdxElectron->Draw();
    hStdDEdxElectron->Write();
    cStdDEdxElectron->Print(Form("%s/cStdDEdxElectron_%s.root", outDir.Data(), dataORmc.Data()),"root");
    cStdDEdxElectron->Print(Form("%s/cStdDEdxElectron_%s.pdf", outDir.Data(), dataORmc.Data()),"pdf");

    cout<<"triggers all runs = "<<fHistTriggersPerRun->Integral()<<endl;
    cout<<"triggers selected runs = "<<hEvents->Integral()<<endl;
}