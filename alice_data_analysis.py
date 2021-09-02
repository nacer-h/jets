import uproot
import pandas as pd
import fastjet as fj
import fjcontrib
import fjext
import os
from pyjetty.mputils import perror, pwarning, pinfo, pdebug
from pyjetty.mputils import treewriter
from tqdm import tqdm

class AnalyzeDataFile(object):
    """This class hass a few features:

    open an output file in the contructor (define the output - tree in this case) - in the .init_output()
    for each file one can call .analyze_file(<input_file_path>)
    at the end one has to call .finish() to write to the output file properly
    now the analysis is implemented in the .analyze_event()

    So the short code looks like:

    an = AnalyzeDataFile('output.root')
    an.analyze_file('some_alice_input_file.root')
    an.analyze_file('some_other_alice_input_file.root')
    ...
    an.finish()"""
    
    def __init__(self, output_filename):
        self.event_tree_name = 'PWGHF_TreeCreator/tree_event_char'
        self.track_tree_name = 'PWGHF_TreeCreator/tree_Particle'
        if not self.init_output(output_filename):
            perror('unable to initialize with', output_filename, 'file as output.')
            return
        
    def get_pandas_from_a_file_with_query(self, file_name, tree_name, squery=None):
        try:
            tree = uproot.open(file_name)[tree_name]
        except:
            pwarning('error getting', tree_name, 'from file:', file_name)
            return None
        if not tree:
            perror('tree {} not found in file {}'.format(tree_name, file_name))
            return None
        if uproot.version.version_info[0] == '4':
            df = tree.arrays(library="pd")
        else:
            df = tree.pandas.df()
        if squery:
            #df.query(squery, inplace=True)
            df = df.query(squery)
            df.reset_index(drop=True)
        return df
    
    def analyze_file(self, input_filename):
        if not os.path.isfile(input_filename):
            perror('file', input_filename, 'does not exists.')
            return
        _event_df = self.get_pandas_from_a_file_with_query(input_filename, self.event_tree_name, 'is_ev_rej == 0')
        if _event_df is None:
            perror('unable to continue...')
            return
        _track_df_unmerged = self.get_pandas_from_a_file_with_query(input_filename, self.track_tree_name)
        if _track_df_unmerged is None:
            perror('unable to continue...')
            return
        # Merge event info into track tree
        if 'ev_id_ext' in list(_event_df):
            _track_df = pd.merge(_track_df_unmerged, _event_df, on=['run_number', 'ev_id', 'ev_id_ext'])
        else:
            # older data formats - should not be used
            pwarning('using the old data format - without ev_id_ext ...')
            _track_df = pd.merge(_track_df_unmerged, _event_df, on=['run_number', 'ev_id'])
        # group the track dataframe by event
        # track_df_grouped is a DataFrameGroupBy object with one track dataframe per event
        _track_df_grouped = _track_df.groupby(['run_number','ev_id'])
        # transform -> analyze the "events==dataframes" 
        # use a progress bar
        self.pbar = tqdm(range(len(_track_df_grouped)))
        _ = _track_df_grouped.apply(self.analyze_event)
        self.pbar.close()

    def get_unique_fname(fn):
        counter = 0
        outfn = fn.replace('.root', '_{}.root'.format(counter))
        while os.path.exists(outfn):
            counter += 1
            outfn = fn.replace('.root', '_{}.root'.format(counter))
        return outfn

    def init_output(self, output_filename):
        self.output_filename = output_filename
        self.output_tree = treewriter.RTreeWriter(name = 'jets', file_name = self.output_filename)
        if self.output_tree:
            return True
        return False
        
    def analyze_event(self, df):
        #df here is a merged dataframe from the event and track trees
        self.pbar.update(1)

        #make FastJet PseudoJet vectors from tracks describbed by Pt,Eta,Phi 
        _particles = fjext.vectorize_pt_eta_phi(df['ParticlePt'].values, df['ParticleEta'].values, df['ParticlePhi'].values)

        #define a jet finder on the event
        jet_R0 = 0.4
        jet_definition = fj.JetDefinition(fj.antikt_algorithm, jet_R0)
        # select jets in a pT window and fully contained within ALICE acceptance (eta tracks < 0.9)
        jet_selector = fj.SelectorPtMin(10.0) & fj.SelectorPtMax(1000.0) & fj.SelectorAbsEtaMax(0.9 - jet_R0)
        # get jets == actually run the jet finder
        jets = fj.sorted_by_pt(jet_selector(jet_definition(_particles)))        
        #define a soft drop algorithm
        sd01 = fjcontrib.SoftDrop(0, 0.1, 1.0)
        
        # do something for each jet
        for j in jets:
            # get the jet after soft drop
            j_sd01 = sd01.result(j)
            # get the kinematics of the soft drop selected split
            j_sd01_info = fjcontrib.get_SD_jet_info(j_sd01)

            # get parents of SD jet (leading and sub-leading prongs)
            pe1 = fj.PseudoJet()
            pe2 = fj.PseudoJet()
            has_parents = j_sd01.has_parents(pe1, pe2)
            # compute jet momentum fraction of the leading (highest pt) and sub-leading (lowest pt) prongs (zg = Pt_sub-leading/(Pt_sub-leading + Pt_leading))
            zg = 0.0
            if not has_parents:
                zg = -1000
                continue
            if (pe1.pt() + pe2.pt()) == 0:
                zg = -1000
                continue
            else:
                if pe1.pt() > pe2.pt():
                    zg = pe2.pt() / (pe1.pt() + pe2.pt())
                else:
                    zg = pe1.pt() / (pe1.pt() + pe2.pt())
                    
            # get the subjets kinematics
            # sjets04 = fj.sorted_by_pt(jet_definition(j.constituents()))
            # stream to the output tree
            self.output_tree.fill_branches( j         = j,
                                            j_sd01    = j_sd01, 
                                            j_sd01_z  = j_sd01_info.z, 
                                            j_sd01_mu = j_sd01_info.mu, 
                                            j_sd01_Rg = j_sd01_info.dR,
                                            zg = zg
                                            )
            self.output_tree.fill_tree()
            
    def finish(self):
        # need to call to close the output file properly
        self.output_tree.write_and_close()

# Process the data trees with jet algorithms to produce the analyzed tree
## Pb-Pb 5.02 TeV data - /rstorage/alice/data/LHC18qr/570/LHC18q/000296433/0068/AnalysisResults.root
## pp 13 TeV data - /rstorage/alice/data/LHC18bdeghijkklnop/436/alice/data/2018/LHC18b/000285064/pass1/AOD208/PWGHF/HF_TreeCreator/436_20200520-0351_child_1/0001/AnalysisResults.root
## pp 5.02 TeV data - /rstorage/alice/data/LHC18b8/449/child_1/TrainOutput/1/282008/0001/AnalysisResults.root
an_data_file = AnalyzeDataFile('alice_data_analysis_output.root')
fname = '/rstorage/alice/data/LHC18bdeghijkklnop/436/alice/data/2018/LHC18b/000285064/pass1/AOD208/PWGHF/HF_TreeCreator/436_20200520-0351_child_1/0001/AnalysisResults.root'
an_data_file.analyze_file(fname)
an_data_file.finish()